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
            [chosen_end, rejected_end],       # rejected -> context
            [chosen_end, rejected_end]]       # rejected -> self
k_ranges = [[0, ctx_end],                    # context attends to context
            [0, chosen_end],                  # chosen attends to context+chosen
            [0, ctx_end],                     # rejected attends to context
            [chosen_end, rejected_end]]       # rejected attends to self
attn_types = [FULL, CAUSAL, FULL, CAUSAL]
```

MagiAttention's FFA kernel processes these slices natively without materializing a dense mask.

---

## 2. Implementation Details

### 2.1 Files Created

#### `turbo_alignment/modeling/magi_attn.py`
Registers the MagiAttention backend with HuggingFace Transformers. This allows any HF model to use MagiAttention by setting `config._attn_implementation = "magi_attention"`.

#### `turbo_alignment/modeling/magi_mask_converter.py`
Converts PairPreference boundaries (context/chosen/rejected lengths) into MagiAttention's `q_ranges`, `k_ranges`, and `attn_types`. This avoids creating the 1.15 GB dense mask.

### 2.2 Files Modified

#### `turbo_alignment/settings/tf/trainer.py`
Added `sp_backend` field to `TrainerSettings`. Defaults to `"ulysses"`, can be set to `"magi_attention"`.

#### `turbo_alignment/dataset/pair_preferences/collators.py`
Modified `PairPreferenceDataCollator` to include `boundaries` in the batch. This tensor `[B, 3]` is lightweight and needed for MagiAttention dispatch.

#### `turbo_alignment/trainers/rm.py`
Modified `RMTrainer` to handle MagiAttention:
- In `_prepare_inputs`: Checks for `sp_backend="magi_attention"`. (Currently placeholder for dispatch logic, as dispatch is complex and requires group handling).
- In `compute_loss`: Implements the full dispatch -> model -> undispatch flow.
  - Converts boundaries to Magi ranges.
  - Calls `magi_attn_varlen_dispatch` to distribute tokens.
  - Calls model with dispatched tokens (and no mask).
  - Calls `undispatch` to gather hidden states.
  - Computes loss using standard indexing (no `all_reduce` needed!).

#### `turbo_alignment/pipelines/train/base.py`
Modified to skip wrapping the data collator with `DataCollatorForSequenceParallism` when `sp_backend="magi_attention"`. MagiAttention handles its own parallelism internally.

#### `turbo_alignment/sequence_parallel/trainer.py`
Modified `TrainerWithSeqP` to correctly calculate effective batch size when using MagiAttention. It uses `args.sequence_parallel` as the CP size instead of relying on Ulysses `parallel_states`.

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

-   **Hardware**: Requires NVIDIA Hopper GPUs (H100/H200).
-   **Model Support**: Works with any HF model that supports `_attn_implementation` config (e.g., Qwen2, Llama3).
-   **Batch Size**: Effective batch size calculation assumes `sequence_parallel` parameter reflects the Context Parallel size.

## 5. Usage

To use MagiAttention for RM training:

1.  **Install MagiAttention**: Ensure `magi_attention` is installed in your environment (requires Hopper GPUs).
2.  **Configure Training**: Set `sp_backend: "magi_attention"` in your trainer configuration yaml/json.
3.  **Run**:
    ```bash
    accelerate launch ... --sp_backend magi_attention ...
    ```

## 6. Future Work

- **TODO**: Override `model_type` and `_attn_implementation` automatically based on `sp_backend`.
  Currently, users must manually ensure they are using the correct model type (e.g., `seq_cls` instead of `seq_cls_qwen3_with_ulysses`) when `sp_backend="magi_attention"`. In the future, the trainer or model loader should automatically select the appropriate model class and set `_attn_implementation="magi_attention"` when the backend is enabled, preventing configuration mismatches.
