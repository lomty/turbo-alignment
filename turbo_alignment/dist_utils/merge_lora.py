#!/usr/bin/env python3
"""Merge LoRA adapter into base model for reward model evaluation.

This script loads a LoRA adapter checkpoint (including modules_to_save like score.weight)
and merges it with the base model, saving a standalone model that can be loaded by
transformers AutoModelForSequenceClassification without PEFT.

Usage:
    python merge_lora.py <adapter_checkpoint_path> [output_path]
"""
import logging
import sys
import os
from safetensors import safe_open
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def checkpoint_merge_lora():
    if len(sys.argv) < 2:
        print("Usage: python merge_lora.py <adapter_checkpoint_path> [output_path]")
        sys.exit(1)

    adapter_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/merged_model"
    base_path = "/from_s3/model"

    print(f"Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print(f"Loading base model from: {base_path}")
    # Suppress expected warning about uninitialized score.weight — it will be overwritten
    # by the trained score.weight from the LoRA adapter's modules_to_save
    _logger_modeling = logging.getLogger("transformers.modeling_utils")
    _prev_level = _logger_modeling.level
    _logger_modeling.setLevel(logging.ERROR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    _logger_modeling.setLevel(_prev_level)

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    with safe_open(os.path.join(adapter_path, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        print("Patching `score.weight` in model")
        merged.score.weight.data = f.get_tensor("base_model.model.score.weight")

    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Successfully saved merged model to {output_path}")


if __name__ == "__main__":
    checkpoint_merge_lora()
