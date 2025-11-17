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

logger = logging.get_logger(__name__)


class RMTrainer(MultiGPUCherryPicksTrainer):
    """
    Reward Model Trainer for preference learning.

    Trains a reward model by comparing scores for chosen vs rejected responses.
    Expects batches from PairPreferenceDataCollator with structure:
        {
            'input_ids': Tensor[2*batch_size, seq_len],  # chosen first, then rejected
            'attention_mask': Tensor[2*batch_size, seq_len],
            'labels': Tensor[2*batch_size, seq_len],
            'chosen_idxs': int,  # batch_size, used to split chosen/rejected
            'precomputed_margin': Optional[Tensor[batch_size]]
        }

    The collator concatenates chosen and rejected inputs along batch dimension for efficient
    forward pass. The trainer splits rewards at chosen_idxs for loss computation.
    """

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,  # pylint: disable=unused-argument
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        Compute ranking loss for reward model.

        Args:
            model: Reward model
            inputs: Batch dictionary with concatenated chosen/rejected inputs
            return_outputs: If True, return (loss, outputs_dict)
            num_items_in_batch: Unused, kept for API compatibility

        Returns:
            If return_outputs=False: loss scalar
            If return_outputs=True: (loss, {'rewards_w': Tensor[batch_size],
                                            'rewards_l': Tensor[batch_size]})

        Loss computation:
            loss = -log(sigmoid(rewards_w - rewards_l)).mean()
            Encourages chosen rewards > rejected rewards
        """
        input_ids = inputs['input_ids'].to(self.accelerator.device)
        attention_mask = inputs['attention_mask'].to(self.accelerator.device)

        all_rewards = model(input_ids, attention_mask=attention_mask, return_dict=True)[0]
        # Split rewards at chosen_idxs: first half is chosen, second half is rejected
        chosen_idxs = inputs['chosen_idxs']
        rewards_w, rewards_l = all_rewards[:chosen_idxs], all_rewards[chosen_idxs:]

        # Compute ranking loss: chosen should have higher reward than rejected
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
