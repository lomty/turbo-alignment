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
        chosen, rejected  = [], []
        precomputed_margins = []

        for ex in examples:
            chosen.append({'input_ids': torch.cat([ex['inputs_context'], ex['inputs_chosen']])})
            rejected.append({'input_ids': torch.cat([ex['inputs_context'], ex['inputs_rejected']])})
            # all_features.append(ex['input_ids'])
            if 'precomputed_margin' in ex and ex['precomputed_margin'] is not None:
                precomputed_margins.append(ex['precomputed_margin'])

        # Use parent class to handle padding - it will pad all features to max length
        batch = super().__call__(chosen+rejected)
        batch['chosen_idxs'] = len(chosen)

        #TODO:remove
        batch['inputs_w'], batch['inputs_l'] = dict(), dict()
        for name, key in batch.items():
            if hasattr(batch[name], 'shape'):
                batch['inputs_w'][name] = batch[name][:len(chosen)]
                batch['inputs_l'][name] = batch[name][len(chosen):]

        if precomputed_margins:
            batch['precomputed_margin'] = torch.tensor(precomputed_margins)

        return batch
