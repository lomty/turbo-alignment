# pylint: skip-file

import torch
import torch.distributed as dist


class GatherAllLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, sp_group: dist.ProcessGroup) -> torch.Tensor:
        """
        Gather all logits accross rank in sequence parallel group
        args:
            logits: tensor with size batch_size x (seq_len // sequence parall size) x vocab_size
            sp_group: sequence parallel group
        return:
            tensor with size batch_size x seq_len x vocab_size
        """

        sp_world_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        ctx.sp_world_size = sp_world_size
        ctx.sp_rank = sp_rank
        ctx.seqlen = logits.size(1) * sp_world_size

        bs = logits.size(0)
        transposed = logits.transpose(0, 1).contiguous()
        all_logits = torch.zeros(
            (ctx.seqlen, bs) + logits.shape[2:],
            dtype=logits.dtype,
            device=logits.device,
        )

        dist.all_gather_into_tensor(all_logits, transposed, group=sp_group)
        del transposed
        return all_logits.transpose(0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        step_seqlen = ctx.seqlen // ctx.sp_world_size
        sp_rank = ctx.sp_rank
        grad_output_part = grad_output[:, step_seqlen * sp_rank : step_seqlen * (sp_rank + 1), :]
        return grad_output_part, None


class GatherRewardAtIndex(torch.autograd.Function):
    """Extract reward at a specific index from logits with proper sequence-parallel gradient handling.

    This avoids the autograd warning from in-place all_reduce inside the backward graph
    by performing the all_reduce in forward() where it's opaque to autograd.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, indices: torch.Tensor, sp_group: dist.ProcessGroup) -> torch.Tensor:
        """
        Extract reward values at specified indices from logits, handling sequence parallelism.

        Args:
            logits: Local logits chunk with shape [batch_size, seq_len_chunk, 1]
            indices: Global indices to extract rewards from, shape [batch_size]
            sp_group: Sequence parallel process group

        Returns:
            rewards: Gathered rewards with shape [batch_size], same on all ranks
        """
        sp_world_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)

        seq_len_chunk = logits.size(1)
        offset = sp_rank * seq_len_chunk

        # Determine which indices belong to this rank's chunk
        is_local = (indices >= offset) & (indices < offset + seq_len_chunk)

        # Clamp indices to valid local range for safe indexing
        local_indices = torch.where(is_local, indices - offset, torch.zeros_like(indices))

        batch_size = indices.size(0)
        batch_idx = torch.arange(batch_size, device=logits.device)

        # Extract rewards at local positions (zeros for non-local indices)
        rewards = logits[batch_idx, local_indices].squeeze(-1)
        rewards = rewards * is_local.to(rewards.dtype)

        # All-reduce to combine rewards from all ranks (inside forward, opaque to autograd)
        dist.all_reduce(rewards, op=dist.ReduceOp.SUM, group=sp_group)

        # Save for backward
        ctx.is_local = is_local
        ctx.local_indices = local_indices
        ctx.logits_shape = logits.shape

        return rewards

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: scatter gradients back to the rank that owns each index.

        Args:
            grad_output: Gradient w.r.t. rewards, shape [batch_size]

        Returns:
            Tuple of (grad_logits, None, None) for (logits, indices, sp_group)
        """
        is_local = ctx.is_local
        local_indices = ctx.local_indices
        logits_shape = ctx.logits_shape

        # Initialize gradient tensor for local logits chunk
        grad_logits = torch.zeros(logits_shape, dtype=grad_output.dtype, device=grad_output.device)

        batch_size = grad_output.size(0)
        batch_idx = torch.arange(batch_size, device=grad_output.device)

        # Scatter gradients: only place grad where this rank owns the index
        # grad_output is already the same on all ranks (from the all_reduce in forward)
        # Each rank contributes only to its local positions
        grad_logits[batch_idx, local_indices, 0] = grad_output * is_local.to(grad_output.dtype)
        return grad_logits, None, None
