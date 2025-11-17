from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from turbo_alignment.constants import DISABLE_LOSS_LABEL


class PairPreferenceDataCollator(DataCollatorForSeq2Seq):
    """
    Data collator for pair preference datasets (reward model training).
    Processes chosen/rejected answer pairs by concatenating context with each answer,
    then batching them together for efficient training.
    
    Input format (from dataset):
        {
            'inputs_context': Tensor[context_len],
            'inputs_chosen': Tensor[chosen_len],
            'inputs_rejected': Tensor[rejected_len],
            'precomputed_margin': Optional[float]
        }
    
    Output format (batch):
        {
            'input_ids': Tensor[2*batch_size, max_seq_len],  # chosen first, then rejected
            'attention_mask': Tensor[2*batch_size, max_seq_len],
            'labels': Tensor[2*batch_size, max_seq_len],  # -100 for context, actual tokens for answers
            'chosen_idxs': int,  # batch_size, used to split chosen/rejected in trainer
            'precomputed_margin': Optional[Tensor[batch_size]]
        }
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

    def _process_example(self, ex: dict) -> tuple[torch.Tensor, torch.Tensor, dict, float | None]:
        """Process a single example into input_ids, labels, boundaries, and margin."""
        context, chosen, rejected = ex['inputs_context'], ex['inputs_chosen'], ex['inputs_rejected']

        # Sequential arrangement: [context | chosen | rejected]
        input_ids = torch.cat([context, chosen, rejected])

        # Labels: mask context with -100
        labels = input_ids.clone()
        labels[:len(context)] = DISABLE_LOSS_LABEL

        boundaries = {
            'context_end': len(context),
            'chosen_end': len(context) + len(chosen),
            'rejected_end': len(context) + len(chosen) + len(rejected)
        }

        return input_ids, labels, boundaries, ex.get('precomputed_margin')

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
        mask = torch.zeros((batch_size, 1, max_seq_len, max_seq_len), dtype=torch.float32, device=device)

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

    def _get_reward_indices(self, boundaries_tensor: torch.Tensor, max_seq_len: int) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Compute indices for extracting rewards at segment boundaries using vectorized operations.

        Args:
            boundaries_tensor: Tensor[batch_size, 3] with [context_end, chosen_end, rejected_end]
            max_seq_len: Maximum sequence length (for validation)

        Returns:
            Tuple of (chosen_indices, rejected_indices) tensors

        Raises:
            ValueError: If any index is out of bounds
        """
        # Extract indices (end position - 1)
        chosen_indices = boundaries_tensor[:, 1] - 1  # chosen_end - 1
        rejected_indices = boundaries_tensor[:, 2] - 1  # rejected_end - 1

        max_idx = max(chosen_indices.max().item(), rejected_indices.max().item())
        if max_idx >= max_seq_len:
            raise ValueError(
                f"Reward extraction index {max_idx} out of bounds for padded sequence length {max_seq_len}. "
                f"Check segment boundaries."
            )

        return chosen_indices, rejected_indices

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:  # type: ignore[override]
        """
        Collate batch of examples from PairPreferenceDataset for reward model training.

        Args:
            examples: List of example dictionaries, each containing:
                - inputs_context: torch.Tensor, dtype=torch.long, shape=[context_len]
                    Token IDs for the context/prompt
                - inputs_chosen: torch.Tensor, dtype=torch.long, shape=[chosen_len]
                    Token IDs for the chosen response
                - inputs_rejected: torch.Tensor, dtype=torch.long, shape=[rejected_len]
                    Token IDs for the rejected response
                - precomputed_margin: Optional[float]
                    Precomputed margin between chosen and rejected scores

        Returns:
            dict[str, Any]: Batched tensors with the following structure:
                - input_ids: torch.Tensor, dtype=torch.long, shape=[2*batch_size, max_seq_len]
                    Concatenated sequences (chosen examples first, then rejected)
                - attention_mask: torch.Tensor, dtype=torch.long, shape=[2*batch_size, max_seq_len]
                    Attention mask (1 for valid tokens, 0 for padding)
                - labels: torch.Tensor, dtype=torch.long, shape=[2*batch_size, max_seq_len]
                    Labels for training (-100 for context tokens, actual token IDs for answer tokens)
                - chosen_idxs: int
                    Split index for chosen/rejected (equals batch_size, used by trainer)
                - precomputed_margin: Optional[torch.Tensor], dtype=torch.float32, shape=[batch_size]
                    Precomputed margins if present in input examples

        Process:
            1. For each example, concatenate context with chosen answer and context with rejected answer
            2. Use parent DataCollatorForSeq2Seq to pad all features to uniform length
            3. Add metadata (chosen_idxs, precomputed_margin) for trainer to split and process
        """
        chosen, rejected = [], []
        precomputed_margins = []

        for ex in examples:
            chosen.append({'input_ids': torch.cat([ex['inputs_context'], ex['inputs_chosen']])})
            rejected.append({'input_ids': torch.cat([ex['inputs_context'], ex['inputs_rejected']])})
            # all_features.append(ex['input_ids'])
            if 'precomputed_margin' in ex and ex['precomputed_margin'] is not None:
                precomputed_margins.append(ex['precomputed_margin'])

        # Use parent class to handle padding - it will pad all features to max length
        batch = super().__call__(chosen + rejected)
        batch['chosen_idxs'] = len(chosen)

        if precomputed_margins:
            batch['precomputed_margin'] = torch.tensor(precomputed_margins)

        return batch
