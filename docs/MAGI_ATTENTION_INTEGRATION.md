# MagiAttention Integration into turbo-alignment

## 1. Rationale: Why Replace Ulysses with MagiAttention

### 1.1 The Problem with Ulysses for RM Training

The current Ulysses sequence parallelism implementation has fundamental limitations for Reward Model training:

**Communication overhead scales with model depth.** Every transformer layer executes **four** `all_to_all` collective operations — two before attention (scatter heads, gather sequence for Q/K/V) and two after (scatter sequence, gather heads for output). For Qwen3-32B with 64 layers, this means **256 all_to_all calls per forward pass**, and another 256 during backward. Each `all_to_all` is a synchronization point even with overlap.

The current implementation in `turbo_alignment/modeling/ulysses_attn.py` is a simple `torch.autograd.Function`:

```python
class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, scatter_idx, gather_idx):
        input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_idx).contiguous()
```

This creates intermediate tensor copies (`.contiguous()`) for every call. With 512 calls per step, this adds measurable overhead.

**Ulysses requires forking the entire model.** The current codebase maintains a complete fork of Qwen3's model code in `turbo_alignment/modeling/qwen3.py` (~500 lines). This fork:
- Rewrites `Qwen3Attention.forward()` to inject `_SeqAllToAll` before and after attention
- Rewrites `Qwen3ModelWithMPU._update_causal_mask()` to handle SP-adjusted sequence lengths
- Rewrites `Qwen3ModelWithMPU.forward()` to handle SP position IDs and cache positions
- Adds `GatherAllLogits.apply()` calls in `Qwen3ForSequenceClassificationWithMPU`

Every HuggingFace Transformers update to Qwen3 requires manually re-applying these patches.

**Ulysses cannot load-balance heterogeneous masks.** The RM attention pattern is:
```
[context=FULL | chosen=CAUSAL | rejected=CAUSAL_with_blocking]
```
Ulysses splits the sequence into equal chunks across GPUs. If the context is 40% of the sequence and chosen/rejected are 30% each, GPU 0 gets mostly FULL attention (more compute) while later GPUs get mostly masked regions (less compute). There's no mechanism to rebalance.

**The 4D attention mask is enormous.** `PairPreferenceDataCollator._get_attn_mask()` creates a `[B, 1, S, S]` boolean mask, then `RMTrainer.compute_loss()` converts it to bf16:
```python
attention_mask = torch.finfo(model.dtype).min * (attention_mask == 0).to(model.dtype)
```
At seq_len=24,576: `24576² × 2 bytes ≈ 1.15 GB` per sample. This mask must be stored on every GPU.

### 1.2 How MagiAttention Solves These Problems

| Problem | Ulysses | MagiAttention |
|---|---|---|
| **Communication per layer** | 4× all_to_all (scatter/gather Q,K,V,O) | 0 (dispatch/undispatch happens once, outside the model) |
| **Communication total** | 512 all_to_all per fwd+bwd | 2 operations total (1 dispatch + 1 undispatch) |
| **Model code changes** | Fork entire model (~500 LOC) | Register one attention function (~15 LOC) |
| **Mask memory** | 1.15 GB dense matrix | ~100 bytes (q_ranges/k_ranges integers) |
| **Load balancing** | None | Dispatch solver balances compute per iteration |
| **Mask support** | Generic 4D (any pattern, but no kernel optimization) | Native heterogeneous mask with FFA kernels |

MagiAttention's architecture is fundamentally different:
1. **Dispatch once**: Before the model forward pass, tokens are distributed across GPUs based on the dispatch solver's load-balanced assignment
2. **Distributed attention per layer**: Each layer computes attention on its local tokens, communicating only the needed K/V via GroupCast/GroupReduce (zero-redundant)
3. **Undispatch once**: After the model forward pass, tokens are gathered back to global order

This means the model itself doesn't need to know about parallelism — it just receives tokens and computes attention using MagiAttention's registered backend.

### 1.3 Why MagiAttention is Ideal for the RM Mask Pattern

The RM mask has a specific structure:

```
Position:     0...ctx_end...chosen_end...rejected_end
              |  context  |   chosen   |   rejected  |
              
Context rows:  attend to [0, ctx_end) with FULL mask
Chosen rows:   attend to [0, chosen_end) with CAUSAL mask  
Rejected rows: attend to [0, ctx_end) with FULL mask
               + attend to [chosen_end, rejected_end) with CAUSAL mask
               (BLOCKED from attending to [ctx_end, chosen_end))
```

This maps directly to MagiAttention's `q_ranges/k_ranges/attn_type_map`:

```python
# 4 attention slices per sample:
q_ranges = [[0, ctx_end],                    # context queries
            [ctx_end, chosen_end],            # chosen queries  
            [chosen_end, rejected_end],       # rejected → context
            [chosen_end, rejected_end]]       # rejected → self
k_ranges = [[0, ctx_end],                    # context attends to context
            [0, chosen_end],                  # chosen attends to context+chosen
            [0, ctx_end],                     # rejected attends to context
            [chosen_end, rejected_end]]       # rejected attends to self
attn_types = [FULL, CAUSAL, FULL, CAUSAL]
```

MagiAttention's FFA kernel processes these slices natively without materializing a dense mask.

---

## 2. Detailed Codebase Changes

### 2.1 Files to CREATE

#### `turbo_alignment/modeling/magi_attn.py` (NEW — ~40 lines)

Registers the MagiAttention backend with HuggingFace Transformers:

```python
"""MagiAttention backend for HuggingFace Transformers.

Registers a custom attention function that delegates to MagiAttention's
distributed calc_attn API. This replaces the Ulysses all_to_all approach
with MagiAttention's dispatch/undispatch paradigm.
"""
import torch
from einops import rearrange
from torch import nn
from typing import Optional

from magi_attention.api import calc_attn, get_most_recent_key
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def magi_attention_forward(
    module: nn.Module,
    query: torch.Tensor,     # [batch=1, num_heads, seq_len, head_dim]
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Drop-in attention function for HuggingFace's ALL_ATTENTION_FUNCTIONS registry.
    
    MagiAttention expects tensors in [total_seq, num_heads, head_dim] format.
    The dispatch/undispatch is handled outside the model, so here we just
    call calc_attn on the local (already dispatched) tokens.
    """
    magi_attn_key = get_most_recent_key()
    dtype = query.dtype
    
    # HF format: [B, heads, seq, dim] → MagiAttn format: [seq, heads, dim]
    q = rearrange(query, "1 nh s hd -> s nh hd").to(torch.bfloat16)
    k = rearrange(key, "1 nh s hd -> s nh hd").to(torch.bfloat16)
    v = rearrange(value, "1 nh s hd -> s nh hd").to(torch.bfloat16)
    
    o, meta = calc_attn(q, k, v, key=magi_attn_key)
    
    # Back to HF format: [seq, heads, dim] → [B=1, seq, heads*dim]
    o = rearrange(o, "s nh hd -> 1 s (nh hd)").to(dtype)
    return o, None


ALL_ATTENTION_FUNCTIONS.register("magi_attention", magi_attention_forward)
```

#### `turbo_alignment/modeling/magi_mask_converter.py` (NEW — ~60 lines)

Converts RM boundary format to MagiAttention's range format:

```python
"""Convert PairPreference boundaries to MagiAttention q_ranges/k_ranges.

The RM training uses [context | chosen | rejected] sequential packing with
a rectangular exclusion mask (rejected cannot attend to chosen).
This module translates those boundaries into MagiAttention's compact
q_ranges/k_ranges/attn_type_map representation.
"""
import torch
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType


def boundaries_to_magi_ranges(ctx_end: int, chosen_end: int, rejected_end: int):
    """
    Convert [context_end, chosen_end, rejected_end] boundaries to MagiAttention ranges.
    
    The RM attention pattern:
      - Context [0, ctx_end): attends to context (FULL)
      - Chosen [ctx_end, chosen_end): attends to context+chosen (CAUSAL)
      - Rejected [chosen_end, rejected_end): attends to context (FULL) + self (CAUSAL)
        but NOT to chosen (rectangular exclusion)
    
    Returns:
        q_ranges: AttnRanges
        k_ranges: AttnRanges  
        attn_mask_types: list[AttnMaskType]
    """
    q_ranges_list = []
    k_ranges_list = []
    attn_types = []
    
    # Slice 1: Context queries → Context keys (FULL bidirectional)
    q_ranges_list.append([0, ctx_end])
    k_ranges_list.append([0, ctx_end])
    attn_types.append(AttnMaskType.FULL)
    
    # Slice 2: Chosen queries → Context+Chosen keys (CAUSAL)
    q_ranges_list.append([ctx_end, chosen_end])
    k_ranges_list.append([0, chosen_end])
    attn_types.append(AttnMaskType.CAUSAL)
    
    # Slice 3: Rejected queries → Context keys (FULL — rejected can see all context)
    q_ranges_list.append([chosen_end, rejected_end])
    k_ranges_list.append([0, ctx_end])
    attn_types.append(AttnMaskType.FULL)
    
    # Slice 4: Rejected queries → Rejected keys (CAUSAL — rejected attends to self)
    q_ranges_list.append([chosen_end, rejected_end])
    k_ranges_list.append([chosen_end, rejected_end])
    attn_types.append(AttnMaskType.CAUSAL)
    
    q_ranges = AttnRanges.from_ranges(q_ranges_list)
    k_ranges = AttnRanges.from_ranges(k_ranges_list)
    
    return q_ranges, k_ranges, attn_types


def batch_boundaries_to_cu_seqlens(boundaries_tensor: torch.Tensor, max_seq_len: int):
    """
    Convert batch of boundaries to cu_seqlens format for varlen dispatch.
    boundaries_tensor: [batch_size, 3] — (ctx_end, chosen_end, rejected_end)
    """
    batch_size = boundaries_tensor.shape[0]
    # Each sample occupies rejected_end tokens
    seqlens = boundaries_tensor[:, 2]  # rejected_end = total length per sample
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    return cu_seqlens, cu_seqlens.clone()  # q and k have same lengths
```

### 2.2 Files to MODIFY

#### `turbo_alignment/trainers/rm.py` — Major Changes

**Current state**: `RMTrainer.compute_loss()` does:
1. Manually calls `model.model(...)` to get hidden_states
2. Calls `model.score(hidden_states)` on ALL positions
3. Has SP-aware reward extraction with `all_reduce` for distributed indices
4. The `prediction_step` has the dimension bug that caused the original crash

**With MagiAttention**: 
- Dispatch happens in `_prepare_inputs` (before `compute_loss`)
- Undispatch happens after model forward (in `compute_loss`)
- Reward extraction becomes simple indexing (no SP-aware logic needed)
- The model is called normally — no manual `model.model(...)` + `model.score(...)` split needed

Specific changes to `compute_loss`:

```python
# CURRENT (with Ulysses SP):
def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    # Manual forward through model internals
    hidden_states = model.model(input_ids=..., attention_mask=..., position_ids=...).last_hidden_state
    logits = model.score(hidden_states)
    
    # SP-aware reward extraction — complex, error-prone
    if parallel_states.sequence_parallel_is_initialized():
        rank = parallel_states.get_sequence_parallel_rank()
        seq_len_chunk = logits.size(1)
        offset = rank * seq_len_chunk
        def get_rewards(indices):
            is_local = (indices >= offset) & (indices < offset + seq_len_chunk)
            local_indices = indices - offset
            safe_indices = torch.where(is_local, local_indices, torch.zeros_like(local_indices))
            rewards = logits[batch_idx, safe_indices].squeeze(-1)
            rewards = rewards * is_local.to(rewards.dtype)
            dist.all_reduce(rewards, op=dist.ReduceOp.SUM, group=parallel_states.get_sequence_parallel_group())
            return rewards
        rewards_w = get_rewards(chosen_indices)
        rewards_l = get_rewards(rejected_indices)
    else:
        rewards_w = logits[batch_idx, chosen_indices]
        rewards_l = logits[batch_idx, rejected_indices]

# WITH MAGIATTENTION:
def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    from magi_attention.api import get_most_recent_key, undispatch
    
    # Standard model forward (model uses registered magi_attention backend internally)
    outputs = model(input_ids=inputs['input_ids'], 
                    attention_mask=None,  # No 4D mask needed!
                    position_ids=inputs['position_ids'])
    hidden_states = outputs.last_hidden_state
    
    # Undispatch: gather tokens back to global order
    magi_key = get_most_recent_key()
    if magi_key is not None:
        hidden_states = undispatch(hidden_states.squeeze(0), magi_key).unsqueeze(0)
    
    logits = model.score(hidden_states)
    
    # Simple indexing — works because undispatch restored global order
    batch_size = inputs['input_ids'].shape[0]
    rewards_w = logits[torch.arange(batch_size), inputs['chosen_indices']]
    rewards_l = logits[torch.arange(batch_size), inputs['rejected_indices']]
    
    loss = -torch.nn.functional.logsigmoid(rewards_w - rewards_l).mean()
```

The SP-aware `get_rewards` function with `is_local` masking and `all_reduce` (20+ lines) is replaced by 2 lines of direct indexing.

#### `turbo_alignment/dataset/pair_preferences/collators.py` — Medium Changes

**Current state**: `PairPreferenceDataCollator.__call__()` produces:
- `input_ids`: padded sequences
- `attention_mask`: **4D mask** `[B, 1, S, S]` (1.15 GB at 24K)
- `position_ids`: symmetric positions
- `chosen_indices`, `rejected_indices`

**With MagiAttention**:
- `input_ids`: same as before
- `attention_mask`: **removed** — MagiAttention uses q_ranges/k_ranges instead
- `boundaries`: `[B, 3]` tensor of (ctx_end, chosen_end, rejected_end) — used for dispatch
- `position_ids`: still needed for symmetric positioning (computed before dispatch)
- `chosen_indices`, `rejected_indices`: same as before

The `_get_attn_mask()` method (which builds the expensive `[S, S]` causal mask with rectangular exclusion) can be **completely removed** when using MagiAttention. The `_get_position_ids()` method is still needed.

The `__call__` method changes:

```python
# CURRENT:
batch['attention_mask'] = self._get_attn_mask(boundaries_tensor, max_seq_len, device)  # 1.15 GB
batch['position_ids'] = self._get_position_ids(boundaries_tensor, max_seq_len)

# WITH MAGIATTENTION:
batch['boundaries'] = boundaries_tensor  # [B, 3] — ~24 bytes
batch['position_ids'] = self._get_position_ids(boundaries_tensor, max_seq_len)
# No attention_mask — saves 1.15 GB per sample
```

#### `turbo_alignment/modeling/qwen3.py` — Can Be Largely Removed

**Current state**: 500+ lines of forked Qwen3 code. The key modifications are:
1. `Qwen3Attention.forward()`: Injects `_SeqAllToAll` calls (lines 99-115 in the class)
2. `Qwen3ModelWithMPU._update_causal_mask()`: SP-adjusted sequence lengths
3. `Qwen3ModelWithMPU.forward()`: SP position ID handling
4. `Qwen3ForSequenceClassificationWithMPU`: `GatherAllLogits.apply()` calls

**With MagiAttention**: 
- `Qwen3Attention` can use the **standard HuggingFace implementation** — no `_SeqAllToAll` needed. The attention is handled by the registered `magi_attention` backend via `ALL_ATTENTION_FUNCTIONS`.
- `_update_causal_mask()` override can be removed — MagiAttention doesn't use 4D masks
- `GatherAllLogits.apply()` in the SequenceClassification model is replaced by `undispatch()` in the trainer

The entire `qwen3.py` could be replaced by using `AutoModelForSequenceClassification` with:
```python
model.config._attn_implementation = "magi_attention"
```

However, to maintain backward compatibility, the file should keep a simplified version without Ulysses-specific code, or add a config flag to choose the backend.

#### `turbo_alignment/modeling/parallel_states.py` — Replaced by MagiAttention's Groups

**Current state**: 500+ lines managing global process groups:
- `_SEQUENCE_PARALLEL_GROUP`
- `_SEQUENCE_DATA_PARALLEL_GROUP`  
- `_DATA_PARALLEL_GROUP`
- Complex `initialize_model_parallel()` that creates DP×SP mesh

**With MagiAttention**: Process groups are created via PyTorch's `DeviceMesh`:

```python
# MagiAttention's approach (from magi_trainer.py):
device_mesh = DeviceMesh(
    device_type="cuda",
    mesh=torch.arange(0, world_size).reshape(world_size // cp_size, cp_size),
    mesh_dim_names=("dp", "cp"),
)
cp_group = device_mesh.get_group("cp")
```

The `parallel_states.py` module can be kept for backward compatibility but MagiAttention's CP group is passed directly to `magi_attn_varlen_dispatch()`.

#### `turbo_alignment/sequence_parallel/collator.py` — No Longer Needed for MagiAttention

**Current state**: `DataCollatorForSequenceParallism` wraps the base collator to split batches across SP ranks:

```python
# In pipelines/train/base.py:
if experiment_settings.trainer_settings.sequence_parallel > 1:
    data_collator = DataCollatorForSequenceParallism.create_with_tokenizer(
        data_collator,
        seq_p_rank=get_sequence_parallel_rank(),
        seq_p_world_size=get_sequence_parallel_world_size(),
        tokenizer=self.tokenizer,
        fields_not_to_split=['attention_mask', 'chosen_indices', 'rejected_indices'],
    )
```

**With MagiAttention**: This wrapper is not needed. MagiAttention's dispatch handles token distribution. The collator produces full sequences, and the trainer dispatches them before the model forward pass.

#### `turbo_alignment/sequence_parallel/trainer.py` — Simplified

**Current state**: Custom `_inner_training_loop` with SP-aware batch size calculations, gradient scaling, etc.

**With MagiAttention**: The trainer only needs:
1. Dispatch in `_prepare_inputs`
2. Undispatch in `compute_loss`
3. Loss scaling by CP size in backward (as shown in MagiAttention's `magi_trainer.py`)

Most of the complex SP logic in `trainer.py` becomes unnecessary.

#### `turbo_alignment/pipelines/train/base.py` — Minor Changes

Remove the `DataCollatorForSequenceParallism` wrapping when using MagiAttention:

```python
# CURRENT:
if experiment_settings.trainer_settings.sequence_parallel > 1:
    data_collator = DataCollatorForSequenceParallism.create_with_tokenizer(...)

# WITH MAGIATTENTION:
if experiment_settings.trainer_settings.sp_backend == "magi_attention":
    pass  # No wrapping needed
elif experiment_settings.trainer_settings.sequence_parallel > 1:
    data_collator = DataCollatorForSequenceParallism.create_with_tokenizer(...)
```

#### `turbo_alignment/settings/model.py` — Add MagiAttention Config

```python
class ModelType(str, Enum):
    CAUSAL = 'causal'
    SEQ_CLS = 'seq_cls'
    # ... existing types ...
    SEQ_CLS_QWEN3_WITH_ULYSSES = 'seq_cls_qwen3_with_ulysses'
    # NEW: MagiAttention doesn't need a separate model type — 
    # it uses standard seq_cls with a registered attention backend
```

#### `turbo_alignment/settings/tf/trainer.py` — Add SP Backend Choice

```python
class TrainerSettings(BaseModel):
    sequence_parallel: int = 1
    sp_backend: str = "ulysses"  # NEW: "ulysses" | "magi_attention"
```

### 2.3 Files UNCHANGED

These files don't need modification:
- `turbo_alignment/dataset/pair_preferences/pair_preference.py` — Dataset reading/tokenization is unaffected
- `turbo_alignment/dataset/chat/chat.py` — Chat tokenization is unaffected
- `turbo_alignment/metrics/reward.py` — Metrics computation works on gathered outputs
- `turbo_alignment/modeling/ulysses_attn.py` — Kept as fallback for non-Hopper GPUs

---

## 3. Data Flow Comparison

### 3.1 Current Flow (Ulysses SP)

```
1. PairPreferenceDataCollator.__call__()
   ├── Concatenate [context | chosen | rejected]           → input_ids [B, S]
   ├── Build 4D mask with rectangular exclusion            → attention_mask [B, 1, S, S] (1.15 GB!)
   ├── Compute symmetric position IDs                      → position_ids [B, S]
   └── Extract boundary indices                            → chosen_indices, rejected_indices

2. DataCollatorForSequenceParallism wraps the batch
   ├── Splits input_ids along seq dim per SP rank          → input_ids [B, S/SP]
   ├── Does NOT split attention_mask (full mask on every GPU)
   └── Does NOT split chosen/rejected indices

3. RMTrainer.compute_loss()
   ├── Convert mask: attention_mask = finfo.min * (mask==0) → bf16 mask (1.15 GB)
   ├── model.model(input_ids, attention_mask, position_ids)
   │   └── Per layer:
   │       ├── _SeqAllToAll(Q, scatter=heads, gather=seq)  → all_to_all #1
   │       ├── _SeqAllToAll(K, scatter=heads, gather=seq)  → all_to_all #2
   │       ├── _SeqAllToAll(V, scatter=heads, gather=seq)  → all_to_all #3
   │       ├── attention_forward(Q, K, V, mask)            → local attention
   │       └── _SeqAllToAll(O, scatter=seq, gather=heads)  → all_to_all #4
   ├── model.score(hidden_states)                          → score ALL S positions
   ├── SP-aware get_rewards with is_local + all_reduce     → extract rewards
   └── Compute ranking loss
```

**Total communication**: 4 × all_to_all × 64 layers × 2 (fwd+bwd) = **512 all_to_all** operations

### 3.2 Proposed Flow (MagiAttention)

```
1. PairPreferenceDataCollator.__call__() [SIMPLIFIED]
   ├── Concatenate [context | chosen | rejected]           → input_ids [B, S]
   ├── NO 4D mask (saves 1.15 GB)
   ├── Compute symmetric position IDs                      → position_ids [B, S]
   ├── Extract boundary indices                            → chosen_indices, rejected_indices
   └── Output boundaries tensor                            → boundaries [B, 3]

2. NO DataCollatorForSequenceParallism wrapping

3. MagiRMTrainer._prepare_inputs()
   ├── Convert boundaries to q_ranges/k_ranges/attn_types  → compact mask descriptor
   ├── magi_attn_varlen_dispatch(input_ids, ...)           → distributed tokens (ONCE)
   ├── Dispatch position_ids alongside input_ids
   └── Store magi_attn_runtime_key for later undispatch

4. MagiRMTrainer.compute_loss()
   ├── model(input_ids, position_ids)                      → standard model forward
   │   └── Per layer:
   │       └── magi_attention_forward(Q, K, V)             → calc_attn with GroupCast/GroupReduce
   │           (zero-redundant communication, load-balanced)
   ├── undispatch(hidden_states, magi_key)                 → gather to global order (ONCE)
   ├── model.score(hidden_states)                          → score ALL positions
   ├── Simple indexing: logits[batch_idx, chosen_indices]  → extract rewards (NO all_reduce!)
   └── Compute ranking loss
   
5. MagiRMTrainer.training_step()
   └── loss * cp_size → backward                           → scale loss for CP
```

**Total communication**: 1 dispatch + N distributed attention operations (optimally overlapped) + 1 undispatch. The per-layer communication uses GroupCast/GroupReduce which is zero-redundant and overlapped with compute.

---

## 4. Constraints and Risks

### 4.1 Hopper-Only Limitation
MagiAttention's FFA kernels **only run on Hopper GPUs** (H100/H200). The current cluster uses H100-8x nodes, so this is compatible. However:
- Local development on consumer GPUs requires Ulysses fallback
- CI pipelines need Hopper access or must skip MagiAttention tests
- **Recommendation**: Keep Ulysses as fallback, select backend via config

### 4.2 DeepSpeed Compatibility
MagiAttention creates its own CP process groups via `DeviceMesh`. These must be compatible with DeepSpeed's data parallel groups. With CP_SIZE=N:
- DeepSpeed DP world = total_GPUs / N
- MagiAttention CP group = N GPUs
- These are orthogonal and should work with ZeRO-2/3

The `MagiAccelerator` class from the transformers example shows how to create the correct mesh.

### 4.3 Batch Size Must Be 1
MagiAttention's `squash_batch_dim` flattens the batch dimension. With the RM's `per_device_train_batch_size: 1`, this is natural. For `per_device_eval_batch_size > 1`, varlen dispatch handles multiple samples via `cu_seqlens`.

### 4.4 Position ID Symmetry
The RM uses symmetric position IDs (rejected mirrors chosen). MagiAttention's `get_position_ids()` returns standard sequential positions. The symmetric positions must be computed before dispatch and dispatched alongside `input_ids`.

---

## 5. Implementation Plan

| Phase | Task | Files | Effort | Risk |
|---|---|---|---|---|
| **1** | Create `magi_attn.py` attention backend | NEW file | 1 day | Low |
| **2** | Create `magi_mask_converter.py` | NEW file | 1 day | Medium |
| **3** | Add `sp_backend` config option | `settings/tf/trainer.py`, `settings/model.py` | 0.5 day | Low |
| **4** | Modify collator to skip 4D mask when using MagiAttention | `collators.py` | 0.5 day | Low |
| **5** | Create `MagiRMTrainer` with dispatch/undispatch/simplified compute_loss | `trainers/rm.py` or NEW file | 2 days | Medium |
| **6** | Skip `DataCollatorForSequenceParallism` wrapping | `pipelines/train/base.py` | 0.5 day | Low |
| **7** | Integration test on H100 cluster | Test configs | 2-3 days | Medium |

**Total: 7-9 days** including testing.

### Dependencies
- `magi_attention` package (pip install from source, requires NGC docker)
- `einops` (for tensor reshaping in attention function)
- Hopper GPU cluster access for testing

### Backward Compatibility
All changes are additive. The `sp_backend: "ulysses"` default preserves existing behavior. MagiAttention is opt-in via `sp_backend: "magi_attention"`.
