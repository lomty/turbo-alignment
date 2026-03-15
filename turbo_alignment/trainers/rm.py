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
from turbo_alignment.modeling import parallel_states
from turbo_alignment.sequence_parallel.gather_logits import GatherRewardAtIndex

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

        model_dtype = next(model.parameters()).dtype
        attention_mask = torch.finfo(model_dtype).min * (attention_mask == 0).to(model_dtype)

        # Prepare kwargs with cache_position if available
        kwargs = {}
        if 'cache_position' in inputs:
            kwargs['cache_position'] = inputs['cache_position'].to(device)

        # Safely unwrap DeepSpeed and PEFT to get access to the classification head (`score`)
        core_model = model
        if hasattr(core_model, 'module'):
            core_model = core_model.module  # Unwrap DeepSpeed
        if hasattr(core_model, 'base_model') and hasattr(core_model.base_model, 'model'):
            core_model = core_model.base_model.model  # Unwrap PEFT

        # Note: We call the inner model directly to get BaseModelOutputWithPast with last_hidden_state,
        # bypassing the SequenceClassifierOutputWithPast which would pool logits to last token only.
        outputs = core_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            **kwargs
        )

        # Get the last hidden state and apply the score head to get full sequence logits
        last_hidden_state = outputs.last_hidden_state
        logits = core_model.score(last_hidden_state)  # [batch_size, seq_len, 1]

        # Extract rewards at segment boundaries from the sequentially packed sequence
        chosen_indices = inputs['chosen_indices'].to(device)
        rejected_indices = inputs['rejected_indices'].to(device)

        if parallel_states.sequence_parallel_is_initialized():
            sp_group = parallel_states.get_sequence_parallel_group()
            rewards_w = GatherRewardAtIndex.apply(logits, chosen_indices, sp_group)
            rewards_l = GatherRewardAtIndex.apply(logits, rejected_indices, sp_group)
        else:
            batch_size = input_ids.shape[0]
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
        logits = torch.stack(logits)
        if logits.dim() == 3:
            logits = logits.mean(dim=2)
        logits = logits.T

        labels = logits[:, 0] > logits[:, 1]

        labels = labels.long()

        return loss, logits, labels

    def _save(self, output_dir=None, state_dict=None):
        logger.info('RMTrainer._save called, output_dir=%s', output_dir)

        core_model = self.model
        if hasattr(core_model, 'module'):
            core_model = core_model.module  # Unwrap DeepSpeed
            logger.info('Unwrapped DeepSpeed module')
        if hasattr(core_model, 'base_model') and hasattr(core_model.base_model, 'model'):
            core_model = core_model.base_model.model  # Unwrap PEFT
            logger.info('Unwrapped PEFT base_model')

        if state_dict is None:
            logger.info('state_dict not provided, calling self.model.state_dict()')
            state_dict = self.model.state_dict()
        logger.info('state_dict has %d keys', len(state_dict))

        # Handle PEFT ModulesToSaveWrapper around score
        score_module = core_model.score
        if hasattr(score_module, 'modules_to_save'):
            active_adapter = next(iter(score_module.modules_to_save))
            logger.info('score is a ModulesToSaveWrapper; using active adapter=%s', active_adapter)
            score_module = score_module.modules_to_save[active_adapter]
        score_weight_param = score_module.weight
        logger.info('score_weight_param shape (may be shard under ZeRO-3): %s', score_weight_param.data.shape)

        if is_deepspeed_zero3_enabled():
            import deepspeed
            world_size = deepspeed.comm.get_world_size()
            rank = deepspeed.comm.get_rank()
            logger.info('ZeRO-3 enabled; gathering score.weight shards from %d ranks (this rank=%d)', world_size, rank)

            local_shard = score_weight_param.data.contiguous()
            gathered = [torch.zeros_like(local_shard) for _ in range(world_size)]
            deepspeed.comm.all_gather(gathered, local_shard)
            full_flat = torch.cat(gathered, dim=0)
            logger.info('all_gather done; full_flat.shape=%s, ds_numel=%d, ds_shape=%s',
                        full_flat.shape, score_weight_param.ds_numel, score_weight_param.ds_shape)

            # Trim ZeRO-3 tail-padding and restore original shape
            full_weight = full_flat[:score_weight_param.ds_numel].reshape(score_weight_param.ds_shape)
            logger.info('score.weight reconstructed: shape=%s', full_weight.shape)
        else:
            logger.info('ZeRO-3 not enabled; using score.weight.data directly, shape=%s', score_weight_param.data.shape)
            full_weight = score_weight_param.data

        # Inject into state_dict under the key that is actually present
        score_key = next((k for k in state_dict if 'score' in k and 'weight' in k), None)
        if score_key is not None:
            logger.info('Injecting score.weight into state_dict under key=%r, weight.shape=%s', score_key, full_weight.shape)
            state_dict[score_key] = full_weight.cpu()
        else:
            logger.warning('score.weight key not found in state_dict; skipping injection. '
                           'Available keys: %s', [k for k in state_dict if 'score' in k])

        logger.info('Calling super()._save(output_dir=%s)', output_dir)
        super()._save(output_dir, state_dict)  # ← _save, not save_model
