"""MagiAttention backend for HuggingFace Transformers.

Registers a custom attention function that delegates to MagiAttention's
distributed calc_attn API. This replaces the Ulysses all_to_all approach
with MagiAttention's dispatch/undispatch paradigm.
"""
import torch
from einops import rearrange
from torch import nn
from typing import Optional

try:
    from magi_attention.api import calc_attn, get_most_recent_key
except ImportError:
    calc_attn = None
    get_most_recent_key = None

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

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
    
    if torch.distributed.get_rank() == 0 and not hasattr(magi_attention_forward, "_logged"):
        logger.info(f"MagiAttention forward pass executed. q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        magi_attention_forward._logged = True

    # Back to HF format: [seq, heads, dim] -> [B=1, seq, heads*dim]
    # We reshape to match input batch dimension, though usually it's 1 after dispatch
    batch_size = query.shape[0]
    o = rearrange(o, "(b s) nh hd -> b s (nh hd)", b=batch_size).to(dtype)
    
    return o, None


ALL_ATTENTION_FUNCTIONS.register("magi_attention", magi_attention_forward)
