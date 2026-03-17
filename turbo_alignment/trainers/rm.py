import os
from typing import Any

import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging

from turbo_alignment.modeling import parallel_states
from turbo_alignment.sequence_parallel.gather_logits import GatherRewardAtIndex
from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer

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

        # In ZeRO-3, state_dict (if provided by self.accelerator.get_state_dict) already contains
        # the fully gathered weights.
        score_key = next((k for k in state_dict if 'score' in k and 'weight' in k), None)

        score_weight = None
        if score_key is not None:
            logger.info('Found score.weight in state_dict under key=%r, shape=%s', score_key,
                        state_dict[score_key].shape)
            score_weight = state_dict[score_key].cpu()
        else:
            logger.warning('score.weight key not found in state_dict. Available keys: %s',
                           [k for k in state_dict if 'score' in k])

        logger.info('Calling super()._save(output_dir=%s)', output_dir)
        super()._save(output_dir, state_dict)  # ← _save, not save_model

        # PEFT's save_pretrained often drops modules_to_save keys like score.weight during its internal filtering.
        # If we have the gathered score.weight, we patch it directly into the saved safetensors file.
        if score_weight is not None and output_dir is not None:
            safetensors_path = os.path.join(output_dir, 'adapter_model.safetensors')
            if os.path.exists(safetensors_path):
                tensors = {}
                with safe_open(safetensors_path, framework='pt', device='cpu') as f:
                    for k in f.keys():
                        tensors[k] = f.get_tensor(k)

                if score_key not in tensors:
                    logger.info('Patching %s: injecting dropped score.weight under key=%r', safetensors_path, score_key)
                    tensors[score_key] = score_weight
                    save_file(tensors, safetensors_path)
                    logger.info('Patch complete')

    def _reload_weights_from_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Reload model weights from a checkpoint directory into the live training model.

        This is called after each checkpoint save to ensure the in-memory model state
        matches what was written to disk (including any patches applied in _save).

        For DeepSpeed ZeRO-3: uses deepspeed_load_checkpoint which handles sharded parameters.
        For non-DeepSpeed PEFT models: loads adapter weights including modules_to_save.
        For non-PEFT models: loads the full model state_dict.
        """
        logger.info('Reloading weights from checkpoint: %s', checkpoint_dir)

        # Unwrap model to detect PEFT (needed for both DeepSpeed and non-DeepSpeed paths)
        core_model = self.model
        if hasattr(core_model, 'module'):
            core_model = core_model.module  # Unwrap DeepSpeed or any wrapper

        is_peft_model = isinstance(core_model, PeftModel)

        # DeepSpeed path: use DeepSpeed's GatheredParameters context
        # This is required for ZeRO-3 where parameters are sharded across ranks
        if self.is_deepspeed_enabled:
            import deepspeed
            logger.info('Using DeepSpeed GatheredParameters for ZeRO-3 compatibility')
            if is_peft_model:
                adapter_path = os.path.join(checkpoint_dir, 'adapter_model.safetensors')
                if not os.path.exists(adapter_path):
                    adapter_path = os.path.join(checkpoint_dir, 'adapter_model.bin')

                if os.path.exists(adapter_path):
                    logger.info('Loading PEFT adapter from: %s', adapter_path)
                    if adapter_path.endswith('.safetensors'):
                        state_dict = load_file(adapter_path)
                    else:
                        state_dict = torch.load(adapter_path, map_location='cpu', weights_only=False)

                    # Temporarily gather sharded parameters to full shape, load on rank 0, then re-shard
                    all_params = list(core_model.parameters())
                    with deepspeed.zero.GatheredParameters(all_params, modifier_rank=0):
                        incompatible = set_peft_model_state_dict(core_model, state_dict)
                    logger.info('ZeRO-3 PEFT adapter loaded. Missing: %s, Unexpected: %s',
                                incompatible.missing_keys, incompatible.unexpected_keys)
                else:
                    logger.warning('No adapter file found in checkpoint: %s', checkpoint_dir)
            else:
                model_path = os.path.join(checkpoint_dir, 'model.safetensors')
                if not os.path.exists(model_path):
                    model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')

                if os.path.exists(model_path):
                    logger.info('Loading model weights from: %s', model_path)
                    if model_path.endswith('.safetensors'):
                        state_dict = load_file(model_path)
                    else:
                        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

                    all_params = list(core_model.parameters())
                    with deepspeed.zero.GatheredParameters(all_params, modifier_rank=0):
                        result = core_model.load_state_dict(state_dict, strict=False)
                    logger.info('ZeRO-3 Model weights loaded. Missing: %s, Unexpected: %s',
                               result.missing_keys, result.unexpected_keys)
                else:
                    logger.warning('No model file found in checkpoint: %s', checkpoint_dir)

            logger.info('Weight reload complete')
            return

        # Non-DeepSpeed path: direct state_dict loading
        if is_peft_model:
            # Load PEFT adapter weights
            adapter_path = os.path.join(checkpoint_dir, 'adapter_model.safetensors')
            if not os.path.exists(adapter_path):
                adapter_path = os.path.join(checkpoint_dir, 'adapter_model.bin')

            if os.path.exists(adapter_path):
                logger.info('Loading PEFT adapter from: %s', adapter_path)
                if adapter_path.endswith('.safetensors'):
                    state_dict = load_file(adapter_path)
                else:
                    state_dict = torch.load(adapter_path, map_location='cpu', weights_only=False)

                # Use PEFT's set_peft_model_state_dict which handles modules_to_save properly
                incompatible = set_peft_model_state_dict(core_model, state_dict)
                logger.info('PEFT adapter loaded. Missing: %s, Unexpected: %s',
                            incompatible.missing_keys, incompatible.unexpected_keys)
            else:
                logger.warning('No adapter file found in checkpoint: %s', checkpoint_dir)
        else:
            # Load full model weights for non-PEFT models
            model_path = os.path.join(checkpoint_dir, 'model.safetensors')
            if not os.path.exists(model_path):
                model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')

            if os.path.exists(model_path):
                logger.info('Loading model weights from: %s', model_path)
                if model_path.endswith('.safetensors'):
                    state_dict = load_file(model_path)
                else:
                    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

                # Load with strict=False to allow partial loads
                result = core_model.load_state_dict(state_dict, strict=False)
                logger.info('Model weights loaded. Missing: %s, Unexpected: %s',
                           result.missing_keys, result.unexpected_keys)
            else:
                logger.warning('No model file found in checkpoint: %s', checkpoint_dir)

        logger.info('Weight reload complete')

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model=model, trial=trial)  # pylint: disable=no-member

        # After saving, reload weights to ensure in-memory state matches disk
        # This is important for ZeRO-3 + PEFT where the save process may have
        # gathered and patched weights (e.g., score.weight)
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        checkpoint_dir = os.path.join(run_dir, checkpoint_folder)

        if os.path.isdir(checkpoint_dir):
            self._reload_weights_from_checkpoint(checkpoint_dir)
