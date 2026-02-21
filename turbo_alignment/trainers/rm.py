import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from peft import PeftModel
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging

try:
    from magi_attention.api import (
        get_most_recent_key, 
        undispatch, 
        magi_attn_flex_dispatch, 
        dispatch,
        compute_pad_size,
        squash_batch_dim,
        get_position_ids
    )
    from magi_attention.config import DistAttnConfig
    from torch.distributed.device_mesh import DeviceMesh
except ImportError:
    get_most_recent_key = None
    undispatch = None
    magi_attn_flex_dispatch = None
    dispatch = None
    compute_pad_size = None
    squash_batch_dim = None
    get_position_ids = None
    DistAttnConfig = None
    DeviceMesh = None

from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer
from turbo_alignment.modeling import parallel_states
from turbo_alignment.modeling.magi_attn import boundaries_to_magi_ranges

logger = logging.get_logger(__name__)


class RMTrainer(MultiGPUCherryPicksTrainer):
    """
    Reward Model Trainer using sequential packing: [context | chosen | rejected].

    Processes all segments in a single forward pass. Extracts rewards at chosen_indices
    and rejected_indices, then computes ranking loss: -log(sigmoid(reward_chosen - reward_rejected)).

    Expected batch from PairPreferenceDataCollator:
        - 'input_ids': Sequentially packed sequences
        - 'attention_mask': 4D masks with chosen/rejected isolation
        - 'position_ids': Symmetric positions
        - 'chosen_indices', 'rejected_indices': Reward extraction positions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if getattr(self.args, "sp_backend", "ulysses") == "magi_attention":
            # Import magi_attn to register the backend
            import turbo_alignment.modeling.magi_attn  # noqa: F401

            # Force the model to use magi_attention backend
            if hasattr(self.model, "config"):
                self.model.config._attn_implementation = "magi_attention"

    def _build_cp_group(self):
        # cp_group do not change during training step.
        if hasattr(self, "cp_group"):
            return self.cp_group

        cp_size = getattr(self.args, "sequence_parallel", 1)
        if cp_size <= 1:
            # Fallback or error if using Magi without SP
            return None
            
        device_mesh = torch.arange(0, torch.distributed.get_world_size()).reshape(
            torch.distributed.get_world_size() // cp_size,  # dp_size
            cp_size,
        )

        device_mesh = DeviceMesh(
            device_type="cuda",
            mesh=device_mesh,
            mesh_dim_names=("dp", "cp"),  # set dp-cp 2-dim parallel
        )

        cp_group = device_mesh.get_group("cp")
        self.cp_group = cp_group

        return cp_group

    def _prepare_inputs(self, inputs: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        """
        Prepare inputs for training, handling MagiAttention dispatch if enabled.
        """
        inputs = super()._prepare_inputs(inputs)
        
        if getattr(self.args, "sp_backend", "ulysses") == "magi_attention":
            if magi_attn_flex_dispatch is None:
                raise ImportError("MagiAttention is required for sp_backend='magi_attention'")
                
            input_ids = inputs["input_ids"]
            position_ids = inputs["position_ids"]
            boundaries = inputs.get("boundaries")
            
            if boundaries is None:
                raise ValueError("Boundaries tensor required for MagiAttention but not found in inputs")
                
            batch_size, max_seq_len = input_ids.shape
            
            # 1. Flatten inputs (remove padding manually based on boundaries)
            # We can't use squash_batch_dim directly because it assumes no padding or specific padding
            # Here we have padding at end, but boundaries tell us valid length
            
            flat_input_ids_list = []
            flat_position_ids_list = []
            
            q_ranges_list = []
            k_ranges_list = []
            attn_types_list = []
            
            total_seqlen = 0
            
            for i in range(batch_size):
                length = boundaries[i, 2].item()
                flat_input_ids_list.append(input_ids[i, :length])
                flat_position_ids_list.append(position_ids[i, :length])
                
                # Get ranges for this sample
                q_r, k_r, a_t = boundaries_to_magi_ranges(
                    boundaries[i, 0].item(), 
                    boundaries[i, 1].item(), 
                    boundaries[i, 2].item()
                )
                q_ranges_list.append(q_r)
                k_ranges_list.append(k_r)
                attn_types_list.append(a_t)
                
                total_seqlen += length
            
            flat_input_ids = torch.cat(flat_input_ids_list)
            flat_position_ids = torch.cat(flat_position_ids_list)
            
            # 2. Prepare dispatch params
            cp_group = self._build_cp_group()
            cp_size = getattr(self.args, "sequence_parallel", 1)
            chunk_size = 512 # Default from examples
            pad_size = compute_pad_size(total_seqlen, cp_size, chunk_size)
            
            # Combine ranges for batch
            # magi_attn_flex_dispatch expects single AttnRanges object if we treat it as one long sequence
            # But the API takes q_ranges (AttnRanges) which can hold multiple ranges
            # We need to offset ranges for subsequent samples in the batch
            
            offset = 0
            combined_q_ranges = []
            combined_k_ranges = []
            combined_attn_types = []
            
            for i in range(batch_size):
                length = boundaries[i, 2].item()
                # q_ranges_list[i] is an AttnRanges object
                # We need to extract its ranges and add offset
                for r in q_ranges_list[i]:
                    combined_q_ranges.append([r.start + offset, r.end + offset])
                for r in k_ranges_list[i]:
                    combined_k_ranges.append([r.start + offset, r.end + offset])
                combined_attn_types.extend(attn_types_list[i])
                
                offset += length
                
            from magi_attention.common import AttnRanges
            q_ranges = AttnRanges.from_ranges(combined_q_ranges)
            k_ranges = AttnRanges.from_ranges(combined_k_ranges)
            
            # 3. Dispatch
            local_input_ids, magi_key = magi_attn_flex_dispatch(
                x=flat_input_ids,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=combined_attn_types,
                total_seqlen_q=total_seqlen,
                total_seqlen_k=total_seqlen,
                pad_size=pad_size,
                chunk_size=chunk_size,
                cp_group_or_mesh=cp_group,
                dist_attn_config=DistAttnConfig(),
                is_same_source=True,
                is_q_permutable=True,
                is_k_permutable=True
            )
            
            # Dispatch position ids using the key
            local_position_ids = dispatch(flat_position_ids, magi_key)
            
            # 4. Update inputs
            # Model expects [B, S], so unsqueeze to [1, S_local]
            inputs["input_ids"] = local_input_ids.unsqueeze(0)
            inputs["position_ids"] = local_position_ids.unsqueeze(0)
            inputs["attention_mask"] = None # No mask needed for Magi
            
            # We don't need boundaries anymore for model forward, but we might need for loss?
            # Actually we need original boundaries to reconstruct batch for reward indexing
            # inputs["boundaries"] is preserved

        return inputs

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,  # pylint: disable=unused-argument
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        Compute ranking loss from sequentially packed inputs.

        Extracts rewards at chosen_indices and rejected_indices from model outputs.

        Returns:
            loss or (loss, {'rewards_w': chosen_rewards, 'rewards_l': rejected_rewards})
        """
        device = self.accelerator.device
        
        # Check if using MagiAttention
        use_magi = getattr(self.args, "sp_backend", "ulysses") == "magi_attention"
        
        if use_magi:
            # Inputs are already dispatched in _prepare_inputs
            # input_ids and position_ids are [1, S_local]
            
            outputs = model.model(
                input_ids=inputs["input_ids"],
                attention_mask=None,
                position_ids=inputs["position_ids"],
                use_cache=False,
            )
            hidden_states = outputs.last_hidden_state
            
            # Undispatch
            magi_key = get_most_recent_key()
            if magi_key is not None:
                # Undispatch returns [total_seq_len, hidden_dim]
                hidden_states = undispatch(hidden_states.squeeze(0), magi_key)
                
                # Reconstruct batch from flat hidden_states using boundaries
                boundaries = inputs['boundaries'].to(device)
                batch_size = boundaries.shape[0]
                # We need max_seq_len from boundaries to know how much to pad
                # But typically RM training uses packed sequences up to max_seq_len
                # Let's assume max_seq_len is sufficient to hold the longest sequence in batch
                # or we can infer it from inputs (but inputs are local now)
                # We can use boundaries[:, 2].max()
                
                max_seq_len = boundaries[:, 2].max().item()
                
                padded_hidden_states = torch.zeros(
                    batch_size, max_seq_len, hidden_states.shape[-1], 
                    dtype=hidden_states.dtype, device=device
                )
                
                start_idx = 0
                for i in range(batch_size):
                    length = boundaries[i, 2].item()
                    padded_hidden_states[i, :length] = hidden_states[start_idx:start_idx+length]
                    start_idx += length
                
                hidden_states = padded_hidden_states
            
            logits = model.score(hidden_states) # [B, S, 1]
            
            # Compute Loss (Standard Indexing)
            chosen_indices = inputs['chosen_indices'].to(device)
            rejected_indices = inputs['rejected_indices'].to(device)
            batch_size = logits.shape[0]
            
            batch_idx = torch.arange(batch_size, device=device)
            rewards_w = logits[batch_idx, chosen_indices].squeeze(-1)
            rewards_l = logits[batch_idx, rejected_indices].squeeze(-1)
            
        else:
            # Original Ulysses / Standard Path
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            position_ids = inputs['position_ids'].to(device)

            attention_mask = torch.finfo(model.dtype).min * (attention_mask == 0).to(model.dtype)

            hidden_states = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).last_hidden_state
            logits = model.score(hidden_states)  # [batch_size, seq_len, 1]

            # Extract rewards at segment boundaries from the sequentially packed sequence
            chosen_indices = inputs['chosen_indices'].to(device)
            rejected_indices = inputs['rejected_indices'].to(device)

            if parallel_states.sequence_parallel_is_initialized():
                rank = parallel_states.get_sequence_parallel_rank()
                seq_len_chunk = logits.size(1)
                offset = rank * seq_len_chunk
                
                def get_rewards(indices):
                    is_local = (indices >= offset) & (indices < offset + seq_len_chunk)
                    local_indices = indices - offset
                    safe_indices = torch.where(is_local, local_indices, torch.zeros_like(local_indices))

                    batch_idx = torch.arange(indices.size(0), device=logits.device)
                    rewards = logits[batch_idx, safe_indices].squeeze(-1)
                    rewards = rewards * is_local.to(rewards.dtype)

                    dist.all_reduce(rewards, op=dist.ReduceOp.SUM, group=parallel_states.get_sequence_parallel_group())
                    return rewards

                rewards_w = get_rewards(chosen_indices)
                rewards_l = get_rewards(rejected_indices)
            else:
                batch_size = input_ids.shape[0]
                rewards_w = logits[torch.arange(batch_size, device=logits.device), chosen_indices]
                rewards_l = logits[torch.arange(batch_size, device=logits.device), rejected_indices]

        loss = -torch.nn.functional.logsigmoid(rewards_w - rewards_l).mean()

        # Scale loss by CP size for MagiAttention
        # This is required because MagiAttention uses distributed autograd which averages gradients across CP group.
        # To match the global batch size semantics, we need to scale up the loss.
        if getattr(self.args, "sp_backend", "ulysses") == "magi_attention":
            cp_size = getattr(self.args, "sequence_parallel", 1)
            loss = loss * cp_size

        if return_outputs:
            return loss, {'rewards_w': rewards_w, 'rewards_l': rewards_l}
        return loss

    def prediction_step(  # type: ignore[override]  #  pylint: disable=signature-differs
            self,
            model: PreTrainedModel | nn.Module,
            inputs: dict[str, dict[str, torch.Tensor]],
            prediction_loss_only: bool,
            ignore_keys: list[str] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        inputs = self._prepare_inputs(inputs)  # type: ignore[arg-type]
        if ignore_keys is None:
            if hasattr(self.model, 'config'):
                ignore_keys = getattr(self.model.config, 'keys_to_ignore_at_inference', [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        logits = torch.stack(logits)
        if logits.dim() == 3:
            logits = logits.mean(dim=2)
        logits = logits.T

        labels = logits[:, 0] > logits[:, 1]

        labels = labels.long()

        return loss, logits, labels

    def _save_checkpoint(self, model, trial):
        if isinstance(model, PeftModel) and is_deepspeed_zero3_enabled():
            logger.info('Running custom _save_checkpoint')
            checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
            run_dir = self._get_output_dir(trial=trial)
            output_dir = Path(os.path.join(run_dir, checkpoint_folder))

            (output_dir / 'cls_head').mkdir(parents=True, exist_ok=True)

            torch.save(model.base_model.model.score.state_dict(), output_dir / 'cls_head' / 'cls_head.pt')

        return super()._save_checkpoint(model=model, trial=trial)  # pylint: disable=no-member
