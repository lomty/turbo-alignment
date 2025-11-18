from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from turbo_alignment.constants import DISABLE_LOSS_LABEL


class PairPreferenceDataCollator(DataCollatorForSeq2Seq):
    """
    Data collator using sequential packing: [context | chosen | rejected].

    Packs all segments into a single sequence for efficient processing. Attention masks
    enforce isolation between chosen/rejected segments. Position IDs are symmetric
    (rejected mirrors chosen) for fair comparison.

    Args:
        tokenizer: Tokenizer for padding operations
        add_labels: Whether to add labels (kept for API compatibility)
        pad_to_multiple_of: Pad sequences to multiple of this value
    """

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

    def _process_example(self, ex: dict) -> tuple[torch.Tensor, tuple[int, int, int], float | None]:
        """
        Pack example into sequential format: [context | chosen | rejected].

        Returns:
            - input_ids: Concatenated tensor
            - boundaries: Tuple (context_end, chosen_end, rejected_end)
            - precomputed_margin: Optional margin value
        """
        context, chosen, rejected = ex['inputs_context'], ex['inputs_chosen'], ex['inputs_rejected']

        # Sequential arrangement: [context | chosen | rejected]
        input_ids = torch.cat([context, chosen, rejected])

        boundaries = (
            len(context),                               # context_end
            len(context) + len(chosen),                 # chosen_end
            len(context) + len(chosen) + len(rejected)  # rejected_end
        )

        return input_ids, boundaries, ex.get('precomputed_margin')

    def _get_attn_mask(self, boundaries: list[tuple[int, int, int]], max_seq_len: int,
                       device: torch.device) -> torch.Tensor:
        """
        Create 4D attention mask enforcing chosen/rejected isolation.

        Attention patterns: context (causal), chosen (to context + causal), rejected (to context + causal).
        Chosen and rejected cannot attend to each other.

        Returns:
            Tensor[batch_size, 1, max_seq_len, max_seq_len] where 1 = attend, 0 = mask
        """
        batch_size = len(boundaries)
        mask = torch.zeros((batch_size, 1, max_seq_len, max_seq_len), dtype=torch.bool, device=device)

        # Create position indices for vectorized operations
        positions = torch.arange(max_seq_len, device=device)
        row_idx = positions.unsqueeze(1)  # [max_seq_len, 1]
        col_idx = positions.unsqueeze(0)  # [1, max_seq_len]

        for i, (ctx_end, chosen_end, rejected_end) in enumerate(boundaries):
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

    def _get_position_ids(self, boundaries: list[tuple[int, int, int]], max_seq_len: int) -> torch.Tensor:
        """
        Compute symmetric position IDs: rejected segment mirrors chosen positions.

        Position scheme: context [0..N-1], chosen [N..N+M-1], rejected [N..N+K-1].

        Returns:
            Tensor[batch_size, max_seq_len] position IDs
        """
        batch_size = len(boundaries)
        base_positions = torch.arange(max_seq_len, dtype=torch.long)
        position_ids = base_positions.unsqueeze(0).expand(batch_size, max_seq_len).clone()

        for i, (ctx_end, chosen_end, rejected_end) in enumerate(boundaries):
            # Overwrite rejected segment with positions starting from ctx_end
            position_ids[i, chosen_end:rejected_end] = base_positions[ctx_end:ctx_end + rejected_end - chosen_end]

        return position_ids

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:  # type: ignore[override]
        """
        Collate batch using sequential packing: [context | chosen | rejected].

        Returns:
            - 'input_ids': Padded sequences
            - 'attention_mask': 4D masks with chosen/rejected isolation
            - 'position_ids': Symmetric positions
            - 'chosen_indices': Last token positions for chosen segments
            - 'rejected_indices': Last token positions for rejected segments
        """

        processed = [self._process_example(ex) for ex in examples]
        concat_input_ids, boundaries, _ = zip(*processed)

        # Use parent class for padding
        batch = super().__call__([{'input_ids': input_ids}
                                  for input_ids in concat_input_ids])
        device = batch['input_ids'].device

        batch_size, max_seq_len = batch['input_ids'].shape[:2]
        batch['attention_mask'] = self._get_attn_mask(boundaries, max_seq_len, device)
        batch['position_ids'] = self._get_position_ids(boundaries, max_seq_len)

        batch['chosen_indices'] = torch.tensor([b[1] - 1 for b in boundaries], dtype=torch.long, device=device)
        batch['rejected_indices'] = torch.tensor([b[2] - 1 for b in boundaries], dtype=torch.long, device=device)

        return batch
