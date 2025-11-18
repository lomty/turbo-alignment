from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from turbo_alignment.constants import DISABLE_LOSS_LABEL


class PairPreferenceDataCollator(DataCollatorForSeq2Seq):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            add_labels: bool = True,
            pad_to_multiple_of: int | None = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors='pt',
            label_pad_token_id=DISABLE_LOSS_LABEL,
        )
        self.add_labels = add_labels

    def _process_example(self, ex: dict) -> tuple[torch.Tensor, dict, float | None]:
        """Process a single example into input_ids, labels, boundaries, and margin."""
        context, chosen, rejected = ex['inputs_context'], ex['inputs_chosen'], ex['inputs_rejected']

        # Sequential arrangement: [context | chosen | rejected]
        input_ids = torch.cat([context, chosen, rejected])

        boundaries = {
            'context_end': len(context),
            'chosen_end': len(context) + len(chosen),
            'rejected_end': len(context) + len(chosen) + len(rejected)
        }

        return input_ids, boundaries, ex.get('precomputed_margin')

    def _process_stack_example(self, ex: dict) -> tuple[torch.Tensor, torch.Tensor, dict, float | None]:
        """Process a single example into input_ids, labels, boundaries, and margin."""
        context, chosen, rejected = ex['inputs_context'], ex['inputs_chosen'], ex['inputs_rejected']

        # Clone tensors to avoid shared memory references that cause pin_memory errors
        chosen_input_ids = torch.cat([context, chosen]).clone()
        rejected_input_ids = torch.cat([context, rejected]).clone()

        boundaries = {
            'context_end': len(context),
            'chosen_end': len(context) + len(chosen),
            'rejected_end': len(context) + len(rejected)
        }

        return chosen_input_ids, rejected_input_ids, boundaries, ex.get('precomputed_margin')

    def _get_attn_mask(self, boundaries_tensor: torch.Tensor, max_seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates 4D attention mask with proper isolation between chosen/rejected segments.
        Uses vectorized operations for efficiency.

        Args:
            boundaries_tensor: Tensor[batch_size, 3] with [context_end, chosen_end, rejected_end]
            max_seq_len: Maximum sequence length
            device: Device for tensor creation

        Returns:
            Tensor[batch_size, 1, max_seq_len, max_seq_len] attention mask
        """
        batch_size = boundaries_tensor.shape[0]
        mask = torch.zeros((batch_size, 1, max_seq_len, max_seq_len), dtype=torch.bool, device=device)

        # Create position indices for vectorized operations
        positions = torch.arange(max_seq_len, device=device)
        row_idx = positions.unsqueeze(1)  # [max_seq_len, 1]
        col_idx = positions.unsqueeze(0)  # [1, max_seq_len]

        for i, (ctx_end, chosen_end, rejected_end) in enumerate(boundaries_tensor):
            # Context: causal attention (lower triangular)
            context_mask = (row_idx < ctx_end) & (col_idx < ctx_end) & (col_idx <= row_idx)

            # Chosen: attend to context + causal within chosen segment
            chosen_to_context = (row_idx >= ctx_end) & (row_idx < chosen_end) & (col_idx < ctx_end)
            chosen_causal = (row_idx >= ctx_end) & (row_idx < chosen_end) & \
                            (col_idx >= ctx_end) & (col_idx < chosen_end) & (col_idx <= row_idx)

            # Rejected: attend to context + causal within rejected segment (isolated from chosen)
            rejected_to_context = (row_idx >= chosen_end) & (row_idx < rejected_end) & (col_idx < ctx_end)
            rejected_causal = (row_idx >= chosen_end) & (row_idx < rejected_end) & \
                              (col_idx >= chosen_end) & (col_idx < rejected_end) & (col_idx <= row_idx)

            # Combine all patterns
            mask[i, 0] = (context_mask | chosen_to_context | chosen_causal |
                          rejected_to_context | rejected_causal).float()

        return mask

    def _get_position_ids(self, boundaries_tensor: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """
        Compute position IDs with context-relative positioning using vectorized operations.

        Position Scheme:
            Context:  [0, 1, 2, ..., N-1]
            Chosen:   [N, N+1, N+2, ...]
            Rejected: [N, N+1, N+2, ...]  # Mirrors chosen positions for symmetry

        Args:
            boundaries_tensor: Tensor[batch_size, 3] with [context_end, chosen_end, rejected_end]
            max_seq_len: Maximum sequence length

        Returns:
            Tensor[batch_size, max_seq_len] position IDs
        """
        batch_size = boundaries_tensor.shape[0]
        base_positions = torch.arange(max_seq_len, dtype=torch.long)
        position_ids = base_positions.unsqueeze(0).expand(batch_size, max_seq_len).clone()

        for i, (ctx_end, chosen_end, rejected_end) in enumerate(boundaries_tensor):
            # Overwrite rejected segment with positions starting from ctx_end
            position_ids[i, chosen_end:rejected_end] = base_positions[ctx_end:ctx_end + rejected_end - chosen_end]

        return position_ids

    def _get_stacked_4d_attention_mask(self, boundaries_list: list[dict], max_seq_len: int,
                                       device: torch.device) -> torch.Tensor:

        chosen_masks, rejected_masks = [], []
        # Create position indices for vectorized operations
        positions = torch.arange(max_seq_len, device=device)
        row_idx = positions.unsqueeze(1)  # [max_seq_len, 1]
        col_idx = positions.unsqueeze(0)  # [1, max_seq_len]

        for boundaries in boundaries_list:
            chosen_end = boundaries['chosen_end']
            rejected_end = boundaries['rejected_end']

            # Start with all positions masked (0 = cannot attend)
            chosen_mask = torch.full((1, max_seq_len, max_seq_len), 0, dtype=torch.bool, device=device)
            # Set causal attention pattern for valid tokens (0.0 = can attend)
            causal_pattern = (row_idx < chosen_end) & (col_idx < chosen_end) & (col_idx <= row_idx)
            chosen_mask[0, causal_pattern] = 1

            # Start with all positions masked (0 = cannot attend)
            rejected_mask = torch.full((1, max_seq_len, max_seq_len), 0, dtype=torch.bool, device=device)
            # Set causal attention pattern for valid tokens (0.0 = can attend)
            causal_pattern = (row_idx < rejected_end) & (col_idx < rejected_end) & (col_idx <= row_idx)
            rejected_mask[0, causal_pattern] = 1

            chosen_masks.append(chosen_mask)
            rejected_masks.append(rejected_mask)

        return torch.stack(chosen_masks + rejected_masks)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:  # type: ignore[override]
        """
        Collate batch of examples from PairPreferenceDataset for reward model training.
        """

        processed = [self._process_example(ex) for ex in examples]
        concat_input_ids, boundaries_list, _ = zip(*processed)

        st_processed = [self._process_stack_example(ex) for ex in examples]
        chosen, rejected, st_boundaries_list, _ = zip(*st_processed)

        # Use parent class for padding
        batch = super().__call__(
            [{'input_ids': input_ids}
             for input_ids in concat_input_ids]
        )
        batch['chosen_idxs'] = len(chosen)
        device = batch['input_ids'].device

        batch1 = super().__call__(
            [{'input_ids': input_ids}
             for input_ids in list(chosen) + list(rejected)]
        )
        batch['st_input_ids'] = batch1['input_ids']

        batch_size, st_max_seq_len = batch1['input_ids'].shape[:2]
        # batch['st_attention_mask'] = batch1['attention_mask']
        batch['st_attention_mask'] = self._get_stacked_4d_attention_mask(
            list(st_boundaries_list), st_max_seq_len, device
        )

        batch['st_position_ids'] = torch.arange(st_max_seq_len, dtype=torch.long) \
            .unsqueeze(0).expand(batch_size, -1).clone()

        boundaries_tensor = torch.tensor(
            [[b['context_end'], b['chosen_end'], b['rejected_end']] for b in boundaries_list],
            dtype=torch.long, device=device
        )

        st_boundaries_tensor = torch.tensor(
            [[b['context_end'], b['chosen_end'], b['rejected_end']] for b in st_boundaries_list],
            dtype=torch.long, device=device
        )

        batch_size, max_seq_len = batch['input_ids'].shape[:2]
        batch['attention_mask'] = self._get_attn_mask(boundaries_tensor, max_seq_len, device)
        batch['position_ids'] = self._get_position_ids(boundaries_tensor, max_seq_len)

        batch['chosen_indices'], batch['rejected_indices'] = \
            boundaries_tensor[:, 1] - 1, boundaries_tensor[:, 2] - 1
        batch['st_chosen_indices'], batch['st_rejected_indices'] = \
            st_boundaries_tensor[:, 1] - 1, st_boundaries_tensor[:, 2] - 1


        return batch
