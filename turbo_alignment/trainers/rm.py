import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging

from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer
from turbo_alignment.sequence_parallel.collator import pad_for_sequence_parallel
from turbo_alignment.modeling import parallel_states

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
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        position_ids = inputs['position_ids'].to(device)

        if parallel_states.sequence_parallel_is_initialized():
            input_ids = pad_for_sequence_parallel(
                input_ids,
                parallel_states.get_sequence_parallel_world_size(),
                self.tokenizer.pad_token_id,  # type: ignore[union-attr]
                padding_side=self.tokenizer.padding_side,  # type: ignore[union-attr]
            )
            attention_mask = pad_for_sequence_parallel(
                attention_mask,
                parallel_states.get_sequence_parallel_world_size(),
                0,
                padding_side=self.tokenizer.padding_side,  # type: ignore[union-attr]
            )
            position_ids = pad_for_sequence_parallel(
                position_ids,
                parallel_states.get_sequence_parallel_world_size(),
                position_ids.shape[1],
                padding_side=self.tokenizer.padding_side,  # type: ignore[union-attr]
            )

            chunk_size = input_ids.size(-1) // parallel_states.get_sequence_parallel_world_size()
            start = chunk_size * parallel_states.get_sequence_parallel_rank()
            end = chunk_size * (parallel_states.get_sequence_parallel_rank() + 1)
            input_ids = input_ids[:, start:end].clone()
            position_ids = position_ids[:, start:end].clone()

        attention_mask = torch.finfo(model.dtype).min * (attention_mask == 0).to(model.dtype)

        hidden_states = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        logits = model.score(hidden_states)  # [batch_size, seq_len, 1]

        batch_size = input_ids.shape[0]
        # Extract rewards at segment boundaries from the sequentially packed sequence
        chosen_indices = inputs['chosen_indices'].to(device)
        rejected_indices = inputs['rejected_indices'].to(device)

        rewards_w = logits[torch.arange(batch_size, device=logits.device), chosen_indices]
        rewards_l = logits[torch.arange(batch_size, device=logits.device), rejected_indices]

        loss = -torch.nn.functional.logsigmoid(rewards_w - rewards_l).mean()

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
        logits = torch.stack(logits).mean(dim=2).T

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
