from typing import Callable

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    BaseImageProcessor,
    DefaultFlowCallback,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PrinterCallback,
    ProcessorMixin,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
)
from transformers.integrations import get_reporting_integration_callbacks

from turbo_alignment.common.tf.callbacks.common import MetricsCallbackHandler
from turbo_alignment.sequence_parallel.trainer import TrainerWithSeqP
from transformers.utils import logging

import os
from pathlib import Path
from peft import PeftModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


logger = logging.get_logger(__name__)

class MultiGPUCherryPicksTrainer(TrainerWithSeqP):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        data_collator: Callable,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ):
        ref_model = kwargs.pop('ref_model', None)
        metrics_kwargs = kwargs.pop('metrics_kwargs', None)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            **kwargs,
        )

        default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = MetricsCallbackHandler(
            callbacks,
            model,
            processing_class,
            None,
            None,
            ref_model=ref_model,
            accelerator=self.accelerator,
            metrics_kwargs=metrics_kwargs,
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else ProgressCallback)
        self.control: TrainerControl = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def train(self, *args, **kwargs):
        if self.is_world_process_zero():
            total_trainable = 0
            lines = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    n = param.numel()
                    total_trainable += n
                    lines.append(
                        f"  {name:80s} {str(list(param.shape)):30s} {str(param.dtype):20s} {n:>12,d}"
                    )

            header = f"  {'Name':80s} {'Shape':30s} {'Dtype':20s} {'#Params':>12}"
            sep = "  " + "-" * (80 + 30 + 20 + 12 + 9)
            body = "\n".join(lines) if lines else "  (none)"
            logger.info(
                f"Trainable parameters weights:\n{header}\n{sep}\n{body}\n{sep}\n"
                f"  Total trainable parameters: {total_trainable:,}"
            )
        return super().train(*args, **kwargs)

    def _validate_checkpoint(self, output_dir: str, expected_keys: set[str]) -> None:
        """
        After save_model writes files, reload them from disk and verify
        every key in expected_keys is present in the saved checkpoint.
        Logs CRITICAL and raises RuntimeError on mismatch.
        """
        if not self.args.should_save:
            return

        saved_keys: set[str] = set()
        for filepath in Path(output_dir).glob('*.safetensors'):
            try:
                from safetensors import safe_open
                with safe_open(str(filepath), framework='pt', device='cpu') as f:
                    saved_keys.update(f.keys())
            except Exception as e:
                logger.warning(f'Could not read {filepath} during checkpoint validation: {e}')
        for filepath in Path(output_dir).glob('*.bin'):
            try:
                data = torch.load(str(filepath), map_location='cpu', weights_only=True)
                saved_keys.update(data.keys())
            except Exception as e:
                logger.warning(f'Could not read {filepath} during checkpoint validation: {e}')

        # PEFT might save keys without the 'base_model.model.' prefix if it unwraps it,
        # so we strip prefixes to do a robust subset check.
        def strip_prefix(k: str) -> str:
            k = k.replace('base_model.model.', '')
            k = k.replace('module.', '')
            return k

        expected_stripped = {strip_prefix(k) for k in expected_keys}
        saved_stripped = {strip_prefix(k) for k in saved_keys}

        missing = sorted(expected_stripped - saved_stripped)
        if missing:
            msg = (
                    f'CHECKPOINT VALIDATION FAILED: {len(missing)}/{len(expected_stripped)} '
                    f'keys missing from {output_dir}:\n' + '\n'.join(f'  MISSING: {k}' for k in missing)
            )
            logger.critical(msg)
            raise RuntimeError(msg)
        else:
            logger.info(
                f'Checkpoint validation passed: all {len(expected_stripped)} keys '
                f'present in {output_dir}'
            )

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        expected_keys_for_validation = None

        if self.is_deepspeed_enabled and isinstance(self.model, PeftModel) and is_deepspeed_zero3_enabled():
            import gc
            import deepspeed
            from transformers.trainer import WEIGHTS_NAME, SAFE_WEIGHTS_NAME
            from transformers.trainer_pt_utils import remove_dummy_checkpoint

            logger.info('Gathering ZeRO-3 sharded score.weight and injecting into state_dict')
            os.makedirs(output_dir, exist_ok=True)

            score_params = list(self.model.base_model.model.score.parameters())
            with deepspeed.zero.GatheredParameters(score_params, modifier_rank=0):
                try:
                    state_dict = self.accelerator.get_state_dict(self.deepspeed)
                    if self.args.should_save:
                        # Inject score parameters into state_dict
                        for name, param in self.model.named_parameters():
                            if "score" in name:
                                state_dict[name] = param.data.cpu().clone()

                        expected_keys_for_validation = set(state_dict.keys())
                        self._save(output_dir, state_dict=state_dict)

                        logger.info('Del state_dict of checkpoint')
                        del state_dict
                        gc.collect()
                except ValueError:
                    logger.warning(
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    if self.args.should_save:
                        self._save(output_dir, state_dict={})
                    remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                    self.model_wrapped.save_checkpoint(output_dir)

            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(commit_message="Model save")

        # Fallback to super() if not using DeepSpeed ZeRO-3 with PEFT
        else:
            if self.args.should_save:
                expected_keys_for_validation = {
                    k for k, p in self.model.named_parameters() if p.requires_grad
                }
            super().save_model(output_dir, _internal_call)

        # Validate checkpoint after all saving is done
        if self.args.should_save and expected_keys_for_validation is not None:
            self._validate_checkpoint(output_dir, expected_keys_for_validation)



