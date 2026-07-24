# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
)


class _ZeroWorkspaceExperts:
    @staticmethod
    def workspace_dtype(dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def workspace_shapes(
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        del (
            N,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )
        return (0,), (0,), (M, K)


def test_zero_workspace_experts_use_graph_local_output(monkeypatch):
    impl = object.__new__(FusedMoEKernelModularImpl)
    impl.fused_experts = _ZeroWorkspaceExperts()

    def fail_if_workspace_manager_is_used():
        raise AssertionError("zero-workspace experts must not use WorkspaceManager")

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.modular_kernel.current_workspace_manager",
        fail_if_workspace_manager_is_used,
    )

    workspace13, workspace2, output = impl._allocate_buffers(
        out_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        M_chunk=4,
        M_full=8,
        N=16,
        K=32,
        top_k=2,
        global_num_experts=8,
        local_num_experts=4,
        expert_tokens_meta=None,
        activation=MoEActivation.SILU,
    )

    assert workspace13.shape == (0,)
    assert workspace2.shape == (0,)
    assert output.shape == (8, 32)
    assert output.dtype == torch.bfloat16
