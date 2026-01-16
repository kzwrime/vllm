# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
all2allv_prepare_finalize.py

This module implements MoE Expert Parallel (EP) prepare and finalize operations
using torch.distributed.all_to_all_single as the communication primitive.

Based on torch.distributed.all_to_all_single:
- Each rank splits its input tensor and sends splits to all other ranks
- Each rank receives splits from all other ranks and concatenates them
- Supports variable split sizes per rank

Reference: docs/serving/expert_parallel_deployment.md
"""

from collections.abc import Callable

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    count_expert_num_tokens,
    moe_kernel_quantize_input,
)

logger = init_logger(__name__)


class All2AllSinglePrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using torch.distributed.all_to_all_single for EP communication.

    This implementation uses the standard PyTorch all_to_all_single collective
    for expert parallel token dispatching and combining.

    Key features:
    - Uses torch.distributed.all_to_all_single for communication
    - Supports variable split sizes (each rank can send/receive different amounts)
    - Compatible with standard PyTorch process groups
    - Suitable for CPU and GPU platforms
    """

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        num_local_experts: int,
        num_dispatchers: int,
        rank_expert_offset: int,
    ):
        """
        Initialize the All2AllSingle prepare/finalize handler.

        Args:
            ep_group: The expert parallel process group
            num_local_experts: Number of local experts on this rank
            num_dispatchers: Number of dispatchers (EP group size)
            rank_expert_offset: Offset of this rank's experts in global space
        """
        super().__init__()
        self.ep_group = ep_group
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers
        self.rank_expert_offset = rank_expert_offset

        # Get rank and world size
        self.ep_rank = dist.get_rank(self.ep_group)
        self.ep_size = dist.get_world_size(self.ep_group)

        # For async operations with DBO
        self.handles: list[object | None] = [None, None]

        # Store communication metadata for finalize phase
        # These are set during prepare_async and used during finalize_async
        self._send_split_sizes: list[int] | None = None
        self._recv_split_sizes: list[int] | None = None
        self._original_indices: torch.Tensor | None = (
            None  # Maps scattered tokens back to original positions
        )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def supports_async(self) -> bool:
        return False

    def _compute_send_recv_counts(
        self,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> tuple[list[int], list[int]]:
        """
        Compute the number of tokens to send to each rank and receive from each rank.

        Args:
            topk_ids: Tensor of shape (num_tokens, topk) with expert assignments
            num_experts: Total number of global experts

        Returns:
            send_counts: List of length ep_size, tokens to send to each rank
            recv_counts: List of length ep_size, tokens to receive from each rank
        """
        num_tokens = topk_ids.size(0)
        topk = topk_ids.size(1)

        # Compute which rank each token should go to
        # Each rank handles experts: [rank * num_local_experts, (rank+1) * num_local_experts)
        experts_per_rank = num_experts // self.ep_size

        # Count tokens per rank
        send_counts = [0] * self.ep_size
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_ids[t, k].item()
                if expert_id >= 0:  # Valid expert
                    rank = expert_id // experts_per_rank
                    rank = min(rank, self.ep_size - 1)
                    send_counts[rank] += 1

        # For receive counts, we need to communicate
        # This will be done via all_to_all_single with appropriate splits

        return send_counts, [0] * self.ep_size

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        """
        Prepare input activations for expert parallel computation.

        This method:
        1. Applies router weights to input if needed
        2. Quantizes activations
        3. Dispatches tokens to appropriate EP ranks using all_to_all_single

        Args:
            a1: Input activations of shape (num_tokens, hidden_dim)
            topk_weights: Router weights for selected experts
            topk_ids: Selected expert IDs for each token
            num_experts: Total number of global experts
            expert_map: Mapping from global to local expert indices
            apply_router_weight_on_input: Whether to apply weights in dispatch
            quant_config: Quantization configuration

        Returns:
            Tuple of (expert_x, expert_x_scale, expert_tokens_meta, topk_ids, topk_weights)
        """
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
        )
        return receiver()

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> Callable:
        """
        Asynchronous prepare using all_to_all_single for token dispatch.
        (Actually this is still a synchronous op)

        Returns a callable that when invoked completes the dispatch operation.
        """

        import traceback

        traceback.print_stack()
        logger.info(f"a1.shape = {a1.shape}, topk_ids.shape = {topk_ids.shape}")

        num_tokens = a1.size(0)
        hidden_dim = a1.size(1)
        topk = topk_ids.size(1)
        device = a1.device
        dtype = a1.dtype

        # Apply router weights on input if requested
        if apply_router_weight_on_input:
            assert topk == 1, "apply_router_weight_on_input only supports topk=1"
            a1 = a1 * topk_weights.to(dtype).unsqueeze(-1)

        # Quantize activations
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            quant_config.a1_scale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )

        # Permute tokens according to expert assignment
        # Create a flat list of (token_idx, expert_id) pairs
        token_expert_pairs = []
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_ids[t, k].item()
                if expert_id >= 0:
                    token_expert_pairs.append((t, k, expert_id))

        # Sort by expert_id to group tokens destined for same expert
        experts_per_rank = num_experts // self.ep_size
        rank_assignments = [[] for _ in range(self.ep_size)]

        for token_idx, k, expert_id in token_expert_pairs:
            rank = min(expert_id // experts_per_rank, self.ep_size - 1)
            rank_assignments[rank].append((token_idx, k, expert_id))

        rank_assignments_sum = [len(x) for x in rank_assignments]
        logger.info(f"rank_assignments_sum = {rank_assignments_sum}")

        # Build send buffer: concatenate tokens for each rank in order
        send_indices = []
        send_topk_ids = []  # Flattened topk_ids for each sent token
        send_topk_weights = []  # Flattened topk_weights for each sent token
        send_split_sizes = []

        for rank_tokens in rank_assignments:
            send_split_sizes.append(len(rank_tokens))
            for token_idx, k, expert_id in rank_tokens:
                send_indices.append(token_idx)
                send_topk_ids.append(expert_id)
                if not apply_router_weight_on_input:
                    send_topk_weights.append(topk_weights[token_idx, k].item())
                else:
                    send_topk_weights.append(1.0)  # Weight already applied

        # Exchange split sizes for metadata
        send_split_sizes_tensor = torch.tensor(
            send_split_sizes, device=device, dtype=torch.long
        )
        recv_split_sizes = torch.zeros_like(send_split_sizes_tensor)

        # Exchange split sizes
        dist.all_to_all_single(
            recv_split_sizes,
            send_split_sizes_tensor,
            group=self.ep_group,
        )

        # Store split sizes for finalize phase (reverse operation will use swapped sizes)
        self._send_split_sizes = send_split_sizes
        self._recv_split_sizes = recv_split_sizes.tolist()

        # Total tokens to receive
        total_recv = recv_split_sizes.sum().item()

        # Allocate receive buffer
        recv_buffer = torch.empty(
            (total_recv, hidden_dim), device=device, dtype=a1q.dtype
        )

        # Allocate flattened metadata buffers
        recv_topk_ids = torch.empty(total_recv, device=device, dtype=topk_ids.dtype)
        recv_topk_weights = torch.empty(
            total_recv, device=device, dtype=topk_weights.dtype
        )

        # Create send buffers
        if len(send_indices) > 0:
            send_indices_tensor = torch.tensor(
                send_indices, device=device, dtype=torch.long
            )
            send_buffer = a1q[send_indices_tensor]
            send_topk_ids_tensor = torch.tensor(
                send_topk_ids, device=device, dtype=topk_ids.dtype
            )
            send_topk_weights_tensor = torch.tensor(
                send_topk_weights, device=device, dtype=topk_weights.dtype
            )
        else:
            send_indices_tensor = torch.empty((0,), device=device, dtype=torch.long)
            send_buffer = torch.empty((0, hidden_dim), device=device, dtype=a1q.dtype)
            send_topk_ids_tensor = torch.empty(
                (0,), device=device, dtype=topk_ids.dtype
            )
            send_topk_weights_tensor = torch.empty(
                (0,), device=device, dtype=topk_weights.dtype
            )

        # Perform all_to_all_single for activations
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=recv_split_sizes.tolist(),
            input_split_sizes=send_split_sizes,
            group=self.ep_group,
        )

        # Exchange flattened topk_ids and topk_weights
        dist.all_to_all_single(
            recv_topk_ids,
            send_topk_ids_tensor,
            output_split_sizes=recv_split_sizes.tolist(),
            input_split_sizes=send_split_sizes,
            group=self.ep_group,
        )

        dist.all_to_all_single(
            recv_topk_weights,
            send_topk_weights_tensor,
            output_split_sizes=recv_split_sizes.tolist(),
            input_split_sizes=send_split_sizes,
            group=self.ep_group,
        )

        # Exchange original token indices for finalize phase
        recv_original_indices = torch.empty(total_recv, device=device, dtype=torch.long)
        dist.all_to_all_single(
            recv_original_indices,
            send_indices_tensor,
            output_split_sizes=recv_split_sizes.tolist(),
            input_split_sizes=send_split_sizes,
            group=self.ep_group,
        )

        # Store original indices for finalize phase
        self._original_indices = recv_original_indices

        # Count tokens per local expert (using received topk_ids)
        expert_num_tokens = count_expert_num_tokens(
            recv_topk_ids.unsqueeze(-1), self.num_local_experts, expert_map
        )

        # Create metadata
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=expert_num_tokens.cpu().clone(),
        )

        # For scales, we need to handle them similarly
        recv_scale = None
        if a1q_scale is not None:
            if len(send_indices) > 0:
                if a1q_scale.dim() == 2:
                    send_scale_buffer = a1q_scale[send_indices]
                else:
                    send_scale_buffer = a1q_scale.expand(len(send_indices), -1)
            else:
                send_scale_buffer = torch.empty(
                    (0, a1q_scale.size(-1)), device=device, dtype=a1q_scale.dtype
                )

            recv_scale = torch.empty(
                (total_recv, *a1q_scale.shape[1:]), device=device, dtype=a1q_scale.dtype
            )

            dist.all_to_all_single(
                recv_scale,
                send_scale_buffer,
                output_split_sizes=recv_split_sizes.tolist(),
                input_split_sizes=send_split_sizes,
                group=self.ep_group,
            )

        def _receiver() -> mk.PrepareResultType:
            # Return topk_ids and topk_weights with shape (num_tokens, 1) to satisfy dim == 2 assertion
            return (
                recv_buffer,
                recv_scale,
                expert_tokens_meta,
                recv_topk_ids.unsqueeze(-1),
                recv_topk_weights.unsqueeze(-1),
            )

        return _receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        """
        Finalize expert outputs by combining results from all EP ranks.

        This method:
        1. Applies weights and reduces if needed
        2. Combines outputs from all ranks using all_to_all_single
        3. Writes result to output tensor

        Args:
            output: Output tensor to write results to
            fused_expert_output: Expert computation results
            topk_weights: Router weights
            topk_ids: Expert assignments
            apply_router_weight_on_input: Whether weights were applied in dispatch
            weight_and_reduce_impl: Weight application and reduction implementation
        """
        receiver = self.finalize_async(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        receiver()

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        """
        Asynchronous finalize using all_to_all_single for result combination.

        The finalize phase reverses the all_to_all_single operation from prepare:
        - Prepare: Local -> Expert ranks (scatter)
        - Finalize: Expert ranks -> Local (reverse scatter)

        Returns a callable that when invoked completes the combine operation.
        """
        # topk_ids and topk_weights have shape (num_tokens, 1) from prepare phase
        num_scattered_tokens = topk_ids.size(0)  # First dimension is num_tokens
        hidden_dim = output.size(1)
        device = output.device
        dtype = output.dtype

        # Get the original output size (number of original tokens)
        num_output_tokens = output.size(0)

        # Apply weight and reduce if not done in dispatch
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        if fused_expert_output.numel() > 0:
            # Apply weights and reduce
            # Note: topk_weights here are the flattened ones matching the scattered tokens
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights
                if not apply_router_weight_on_input
                else torch.ones_like(topk_weights),
                topk_ids=topk_ids.unsqueeze(-1) if topk_ids.dim() == 1 else topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # For the reverse all_to_all_single:
        # In prepare: we sent send_split_sizes, received recv_split_sizes
        # In finalize: we send recv_split_sizes, receive send_split_sizes
        # This reverses the scatter operation

        assert self._send_split_sizes is not None, (
            "send_split_sizes not set - was prepare called?"
        )
        assert self._recv_split_sizes is not None, (
            "recv_split_sizes not set - was prepare called?"
        )

        # Split sizes are swapped for reverse operation
        # In prepare: send_split_sizes[i] = tokens sent to rank i
        # In finalize: recv_split_sizes[i] = tokens to receive from rank i
        finalize_send_sizes = (
            self._recv_split_sizes
        )  # Tokens we received in prepare, now send back
        finalize_recv_sizes = (
            self._send_split_sizes
        )  # Tokens we sent in prepare, now receive back

        # Prepare send buffer
        if fused_expert_output.numel() > 0:
            send_buffer = fused_expert_output.view(num_scattered_tokens, hidden_dim)
        else:
            send_buffer = torch.empty((0, hidden_dim), device=device, dtype=dtype)

        # Allocate receive buffer
        total_recv_tokens = sum(finalize_recv_sizes)
        recv_buffer = torch.empty(
            (total_recv_tokens, hidden_dim), device=device, dtype=dtype
        )

        # Perform reverse all_to_all_single
        dist.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=finalize_recv_sizes,
            input_split_sizes=finalize_send_sizes,
            group=self.ep_group,
        )

        # The recv_buffer now contains the results in the original scattered order
        # We need to accumulate them to get the final output
        # Since each token may have been routed to multiple experts (topk > 1),
        # we need to sum the contributions at the original token positions

        # Use the original indices from prepare phase to scatter results back
        assert self._original_indices is not None, (
            "original_indices not set - was prepare called?"
        )

        output.zero_()
        if recv_buffer.numel() > 0 and total_recv_tokens > 0:
            # Scatter add: accumulate results at original token positions
            # self._original_indices maps each received result back to its original token
            for i in range(total_recv_tokens):
                original_idx = self._original_indices[i].item()
                if 0 <= original_idx < num_output_tokens:
                    output[original_idx].add_(recv_buffer[i])

        def _receiver():
            # No additional work needed for synchronous version
            pass

        return _receiver


def create_all2all_single_prepare_finalize(
    ep_group: dist.ProcessGroup,
    num_local_experts: int,
    num_dispatchers: int,
    rank_expert_offset: int,
) -> All2AllSinglePrepareAndFinalize:
    """
    Factory function to create an All2AllSinglePrepareAndFinalize instance.

    Args:
        ep_group: The expert parallel process group
        num_local_experts: Number of local experts on this rank
        num_dispatchers: Number of dispatchers (EP group size)
        rank_expert_offset: Offset of this rank's experts in global space

    Returns:
        An instance of All2AllSinglePrepareAndFinalize
    """
    return All2AllSinglePrepareAndFinalize(
        ep_group=ep_group,
        num_local_experts=num_local_experts,
        num_dispatchers=num_dispatchers,
        rank_expert_offset=rank_expert_offset,
    )
