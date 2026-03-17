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
