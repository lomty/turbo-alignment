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


def checkpoint_merge_lora(adapter_path: str, base_path: str, output_path: str = "/tmp/merged_model"):
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

    # If the checkpoint contains accumulated base model weights (from in-training merges), load them first
    base_weights_path = os.path.join(adapter_path, "base_model_weights.safetensors")
    if os.path.exists(base_weights_path):
        print(f"Found accumulated base model weights at {base_weights_path}, loading...")
        from safetensors.torch import load_file
        base_state_dict = load_file(base_weights_path)
        # strict=False because it doesn't contain LoRA weights
        model.load_state_dict(base_state_dict, strict=False)

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.exists(adapter_weights_path):
        with safe_open(adapter_weights_path, framework="pt", device="cpu") as f:
            if "base_model.model.score.weight" in f.keys():
                print("Patching `score.weight` in model")
                merged.score.weight.data = f.get_tensor("base_model.model.score.weight")

    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Successfully saved merged model to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_lora.py <adapter_checkpoint_path> [output_path]")
        sys.exit(1)

    adapter_path_arg = sys.argv[1]
    output_path_arg = sys.argv[2] if len(sys.argv) > 2 else "/tmp/merged_model"
    base_path_arg = "/from_s3/model"
    
    checkpoint_merge_lora(adapter_path_arg, base_path_arg, output_path_arg)
