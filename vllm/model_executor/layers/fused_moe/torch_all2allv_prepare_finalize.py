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


def permute_tokens_for_ranks(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts_per_rank: int,
    ep_size: int,
    apply_router_weight_on_input: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[list[int]],
]:
    """
    Permute tokens for rank-based all_to_all communication using vectorized operations.

    This function prepares tokens for expert parallel communication by:
    1. Computing which rank each expert belongs to
    2. Determining which ranks each token needs to be sent to
    3. Duplicating tokens as needed and organizing by target rank

    Args:
        hidden_states: Quantized activations [num_tokens, hidden_dim]
        topk_ids: Expert assignments [num_tokens, topk]
        topk_weights: Router weights [num_tokens, topk]
        experts_per_rank: Number of experts per rank
        ep_size: Number of ranks (expert parallel size)
        apply_router_weight_on_input: Whether weights were applied to input

    Returns:
        send_token_indices: Indices of tokens to send [total_send]
        send_topk_ids: Expert IDs for each sent token [total_send, topk]
        send_topk_weights: Weights for each sent token [total_send, topk]
        send_hidden_states: Activations to send [total_send, hidden_dim]
        rank_assignments: Tuple of token indices for each rank
    """
    num_tokens, topk = topk_ids.shape

    assert topk_ids.min() >= 0

    # expert_ranks shape: [num_tokens, topk]
    expert_ranks = torch.div(topk_ids, experts_per_rank, rounding_mode="floor")

    assert expert_ranks.max() < ep_size

    # TODO: Create a c++ kernel for this
    rank_assignments: list[list[int]] = [[] for _ in range(ep_size)]
    for token_idx in range(num_tokens):
        rank_ids = expert_ranks[token_idx].unique()
        for rank_id in rank_ids:
            rank_assignments[rank_id].append(token_idx)
    all_send_indices = [idx for sublist in rank_assignments for idx in sublist]
    send_token_indices = torch.tensor(all_send_indices)

    send_topk_ids_tensor = topk_ids[send_token_indices]
    if apply_router_weight_on_input:
        send_topk_weights_tensor = torch.ones(
            (send_token_indices.shape[0], topk),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
    else:
        send_topk_weights_tensor = topk_weights[send_token_indices]
    send_hidden_states = hidden_states[send_token_indices]

    send_split_sizes = [len(rank_assignments[i]) for i in range(ep_size)]
    send_split_sizes_tensor = torch.tensor(send_split_sizes, dtype=torch.long)

    return (
        send_token_indices,
        send_topk_ids_tensor,
        send_topk_weights_tensor,
        send_hidden_states,
        send_split_sizes_tensor,
        rank_assignments,
    )


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
        self._send_sizes_sum: int | None = None
        self._recv_sizes_sum: int | None = None

        # Store which tokens were sent to each rank
        # (for finalize phase reverse mapping)
        # _send_rank_assignments[rank] = [token_idx_0, ...]
        self._send_rank_assignments: list[list[int]] | None = (
            None  # Which tokens were sent to each rank
        )
        self._recv_rank_assignments: dict[int, tuple[int, ...]] | None = (
            None  # Which tokens were received from each rank
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
            Tuple of
            (expert_x, expert_x_scale, expert_tokens_meta, topk_ids, topk_weights)
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

        # Assign tokens to ranks based on their expert assignments
        # Each token is sent to each rank that has at least one of its assigned experts
        # The token, topk_ids, and topk_weights are all duplicated per rank

        assert num_experts % self.ep_size == 0
        experts_per_rank = num_experts // self.ep_size

        # Use vectorized permute function to prepare send data
        (
            send_token_indices_tensor,
            send_topk_ids_tensor,
            send_topk_weights_tensor,
            send_hidden_states,
            send_split_sizes_tensor,
            self._send_rank_assignments,
        ) = permute_tokens_for_ranks(
            hidden_states=a1q,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            experts_per_rank=experts_per_rank,
            ep_size=self.ep_size,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        recv_split_sizes_tensor = torch.zeros_like(send_split_sizes_tensor)

        # Exchange split sizes
        dist.all_to_all_single(
            recv_split_sizes_tensor,
            send_split_sizes_tensor,
            group=self.ep_group,
        )

        # Store split sizes for finalize phase
        self._send_split_sizes = send_split_sizes_tensor.tolist()
        self._recv_split_sizes = recv_split_sizes_tensor.tolist()

        # Total tokens to receive
        self._send_sizes_sum = int(send_split_sizes_tensor.sum().item())
        self._recv_sizes_sum = int(recv_split_sizes_tensor.sum().item())

        # Allocate receive tensors
        recv_hidden_states = torch.empty(
            (self._recv_sizes_sum, hidden_dim), device=device, dtype=a1q.dtype
        )

        # Allocate metadata buffers with shape [self._recv_sizes_sum, topk]
        recv_topk_ids = torch.empty(
            (self._recv_sizes_sum, topk), device=device, dtype=topk_ids.dtype
        )
        recv_topk_weights = torch.empty(
            (self._recv_sizes_sum, topk), device=device, dtype=topk_weights.dtype
        )

        # Perform all_to_all_single for hidden states
        dist.all_to_all_single(
            recv_hidden_states,
            send_hidden_states,
            output_split_sizes=self._recv_split_sizes,
            input_split_sizes=self._send_split_sizes,
            group=self.ep_group,
        )

        # Exchange topk_ids and topk_weights
        dist.all_to_all_single(
            recv_topk_ids,
            send_topk_ids_tensor,
            output_split_sizes=self._recv_split_sizes,
            input_split_sizes=self._send_split_sizes,
            group=self.ep_group,
        )

        dist.all_to_all_single(
            recv_topk_weights,
            send_topk_weights_tensor,
            output_split_sizes=self._recv_split_sizes,
            input_split_sizes=self._send_split_sizes,
            group=self.ep_group,
        )

        assert (
            recv_hidden_states.shape[0]
            == recv_topk_ids.shape[0]
            == recv_topk_weights.shape[0]
        ), "Metadata shapes do not match hidden_states"

        # Count tokens per local expert (using received topk_ids)
        expert_num_tokens = count_expert_num_tokens(
            recv_topk_ids.unsqueeze(-1), self.num_local_experts, expert_map
        )

        # Create metadata
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=expert_num_tokens.cpu().clone(),
        )

        # TODO: Not Tested
        assert a1q_scale is None, "a1q_scale is not tested now"
        # For scales, we need to handle them similarly
        recv_scale = None
        if a1q_scale is not None:
            total_send = send_token_indices_tensor.size(0)
            if total_send > 0:
                if a1q_scale.dim() == 2:
                    send_scale = a1q_scale[send_token_indices_tensor]
                else:
                    send_scale = a1q_scale.expand(total_send, -1)
            else:
                send_scale = torch.empty(
                    (0, a1q_scale.size(-1)), device=device, dtype=a1q_scale.dtype
                )

            recv_scale = torch.empty(
                (self._recv_sizes_sum, *a1q_scale.shape[1:]),
                device=device,
                dtype=a1q_scale.dtype,
            )

            dist.all_to_all_single(
                recv_scale,
                send_scale,
                output_split_sizes=self._recv_split_sizes,
                input_split_sizes=self._send_split_sizes,
                group=self.ep_group,
            )

        def _receiver() -> mk.PrepareResultType:
            # Return topk_ids and topk_weights with shape [num_tokens, topk]
            return (
                recv_hidden_states,
                recv_scale,
                expert_tokens_meta,
                recv_topk_ids,  # Already has shape [num_tokens, topk]
                recv_topk_weights,  # Already has shape [num_tokens, topk]
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
        # topk_ids and topk_weights have shape [num_tokens, topk] from prepare phase
        hidden_dim = output.size(1)
        device = output.device
        dtype = output.dtype

        assert output.dtype == fused_expert_output.dtype
        assert fused_expert_output.dim() == 2
        assert fused_expert_output.size(0) == topk_ids.size(0) == topk_weights.size(0)
        assert output.size(1) == fused_expert_output.size(1)

        # Apply weight and reduce if not done in dispatch
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        # TODO: Not Tested.
        # Apply weights and reduce if needed,
        # usually weight_and_reduce_impl is TopKWeightAndReduceNoOP
        if fused_expert_output.numel() > 0:
            # Note: topk_weights here have shape [num_recv_tokens, topk]
            # We need to apply them to the expert outputs
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights
                if not apply_router_weight_on_input
                else torch.ones_like(topk_weights),
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        assert self._send_split_sizes is not None
        assert self._recv_split_sizes is not None
        assert self._send_sizes_sum is not None
        assert self._recv_sizes_sum is not None
        assert self._send_rank_assignments is not None

        # Split sizes are swapped for reverse operation
        finalize_send_sizes = (
            self._recv_split_sizes
        )  # Tokens we received, now send back
        finalize_recv_sizes = self._send_split_sizes  # Tokens we sent, now receive back

        if fused_expert_output.numel() > 0:
            send_hidden_states = fused_expert_output
        else:
            send_hidden_states = torch.empty(
                (0, hidden_dim), device=device, dtype=dtype
            )

        # Allocate receive hidden states
        total_recv_tokens = self._send_sizes_sum
        recv_hidden_states = torch.empty(
            (total_recv_tokens, hidden_dim), device=device, dtype=dtype
        )

        assert self._recv_sizes_sum == send_hidden_states.size(0)

        # Perform reverse all_to_all_single
        dist.all_to_all_single(
            recv_hidden_states,
            send_hidden_states,
            output_split_sizes=finalize_recv_sizes,
            input_split_sizes=finalize_send_sizes,
            group=self.ep_group,
        )

        # Get the original output size (number of original tokens)
        num_output_tokens = output.size(0)

        # Now recv_hidden_states contains results from other ranks
        # We need to scatter them back to original token positions
        # Using send_rank_assignments which tells us which tokens we sent to each rank
        output.zero_()
        if recv_hidden_states.numel() > 0 and total_recv_tokens > 0:
            recv_offset = 0
            for rank in range(self.ep_size):
                num_from_rank = finalize_recv_sizes[rank]
                if num_from_rank == 0:
                    continue

                tokens_sent_to_rank = self._send_rank_assignments[rank]
                for i, original_token_idx in enumerate(tokens_sent_to_rank):
                    assert recv_offset + i < recv_hidden_states.shape[0]
                    assert 0 <= original_token_idx < num_output_tokens
                    output[original_token_idx].add_(recv_hidden_states[recv_offset + i])

                recv_offset += num_from_rank

        def _receiver():
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
