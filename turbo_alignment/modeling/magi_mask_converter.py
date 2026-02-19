"""Convert PairPreference boundaries to MagiAttention q_ranges/k_ranges.

The RM training uses [context | chosen | rejected] sequential packing with
a rectangular exclusion mask (rejected cannot attend to chosen).
This module translates those boundaries into MagiAttention's compact
q_ranges/k_ranges/attn_type_map representation.
"""
#TODO: merge with magi_attn.py
import torch

try:
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType
except ImportError:
    AttnRanges = None
    AttnMaskType = None


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

