"""MagiAttention backend for HuggingFace Transformers.

Registers a custom attention function that delegates to MagiAttention's
distributed calc_attn API. This replaces the Ulysses all_to_all approach
with MagiAttention's dispatch/undispatch paradigm.
"""
import torch
from einops import rearrange
from torch import nn
from typing import Optional

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

from magi_attention.api import calc_attn, get_most_recent_key
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType


logger = logging.get_logger(__name__)


def magi_attention_forward(
    module: nn.Module,
    query: torch.Tensor,     # [batch=1, num_heads, seq_len, head_dim]
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Drop-in attention function for HuggingFace's ALL_ATTENTION_FUNCTIONS registry.
    
    MagiAttention expects tensors in [total_seq, num_heads, head_dim] format.
    The dispatch/undispatch is handled outside the model, so here we just
    call calc_attn on the local (already dispatched) tokens.
    """
    if calc_attn is None:
        raise ImportError("MagiAttention is not installed. Please install it to use magi_attention backend.")

    magi_attn_key = get_most_recent_key()
    dtype = query.dtype
    
    # HF format: [B, heads, seq, dim] -> MagiAttn format: [seq, heads, dim]
    # Note: We assume batch size is 1 (squashed) or handled via dispatch
    q = rearrange(query, "b nh s hd -> (b s) nh hd").to(torch.bfloat16)
    k = rearrange(key, "b nh s hd -> (b s) nh hd").to(torch.bfloat16)
    v = rearrange(value, "b nh s hd -> (b s) nh hd").to(torch.bfloat16)
    
    o, meta = calc_attn(q, k, v, key=magi_attn_key)

    logger.info(f"MagiAttention forward pass executed. q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

    # Back to HF format: [seq, heads, dim] -> [B=1, seq, heads*dim]
    # We reshape to match input batch dimension, though usually it's 1 after dispatch
    batch_size = query.shape[0]
    o = rearrange(o, "(b s) nh hd -> b s (nh hd)", b=batch_size).to(dtype)
    
    return o, None

def boundaries_to_magi_ranges(ctx_end: int, chosen_end: int, rejected_end: int):
    """
    Convert [context_end, chosen_end, rejected_end] boundaries to MagiAttention ranges.

    The RM attention pattern:
      - Context [0, ctx_end): attends to context (FULL)
      - Chosen [ctx_end, chosen_end): attends to context+chosen (CAUSAL)
      - Rejected [chosen_end, rejected_end): attends to context (FULL) + self (CAUSAL)
        but NOT to chosen (rectangular exclusion)

    Returns:
        q_ranges: AttnRanges
        k_ranges: AttnRanges
        attn_mask_types: list[AttnMaskType]
    """
    if AttnRanges is None:
        raise ImportError("MagiAttention is not installed.")

    q_ranges_list = []
    k_ranges_list = []
    attn_types = []

    # Slice 1: Context queries -> Context keys (FULL bidirectional)
    q_ranges_list.append([0, ctx_end])
    k_ranges_list.append([0, ctx_end])
    attn_types.append(AttnMaskType.FULL)

    # Slice 2: Chosen queries -> Context+Chosen keys (CAUSAL)
    q_ranges_list.append([ctx_end, chosen_end])
    k_ranges_list.append([0, chosen_end])
    attn_types.append(AttnMaskType.CAUSAL)

    # Slice 3: Rejected queries -> Context keys (FULL - rejected can see all context)
    q_ranges_list.append([chosen_end, rejected_end])
    k_ranges_list.append([0, ctx_end])
    attn_types.append(AttnMaskType.FULL)

    # Slice 4: Rejected queries -> Rejected keys (CAUSAL - rejected attends to self)
    q_ranges_list.append([chosen_end, rejected_end])
    k_ranges_list.append([chosen_end, rejected_end])
    attn_types.append(AttnMaskType.CAUSAL)

    q_ranges = AttnRanges.from_ranges(q_ranges_list)
    k_ranges = AttnRanges.from_ranges(k_ranges_list)

    return q_ranges, k_ranges, attn_types

ALL_ATTENTION_FUNCTIONS.register("magi_attention", magi_attention_forward)