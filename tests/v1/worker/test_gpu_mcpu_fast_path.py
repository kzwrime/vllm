# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm_xcpu_plugin.fake_triton.vllm_kernels import register_vllm_kernels

from vllm.platforms import current_platform
from vllm.v1.worker.gpu import input_batch as input_batch_module
from vllm.v1.worker.gpu import mcpu_ops
from vllm.v1.worker.gpu.input_batch import (
    get_num_sampled_and_rejected,
    post_update,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

register_vllm_kernels()

DEVICE = torch.device(current_platform.device_type)


pytestmark = pytest.mark.skipif(not mcpu_ops.is_mcpu(), reason="MCPU-only fast path")


def _sync() -> None:
    torch.mcpu.synchronize()


def _num_rejected_case() -> list[torch.Tensor]:
    return [
        torch.tensor([1, 2, 3], dtype=torch.int32, device=DEVICE),
        torch.tensor([5, 2, 8], dtype=torch.int32, device=DEVICE),
        torch.tensor([0, 4, 7, 12], dtype=torch.int32, device=DEVICE),
        torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE),
        torch.tensor([4, 6, 8], dtype=torch.int32, device=DEVICE),
    ]


def test_get_num_sampled_direct_matches_fake_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_inputs = _num_rejected_case()
    _, direct_rejected = get_num_sampled_and_rejected(*direct_inputs)
    _sync()
    direct = (direct_inputs[0].cpu(), direct_rejected.cpu())

    fallback_inputs = _num_rejected_case()
    monkeypatch.setattr(mcpu_ops, "try_get_num_sampled_and_rejected", lambda *a: False)
    _, fallback_rejected = get_num_sampled_and_rejected(*fallback_inputs)
    _sync()

    assert torch.equal(fallback_inputs[0].cpu(), direct[0])
    assert torch.equal(fallback_rejected.cpu(), direct[1])


def _post_update_case(
    *, optional: bool = False
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    idx_mapping = torch.tensor([-1, 1, 0], dtype=torch.int32, device=DEVICE)
    num_computed = torch.tensor([10, 20], dtype=torch.int32, device=DEVICE)
    last_sampled = torch.full((2, 1), -1, dtype=torch.int64, device=DEVICE)
    output_counts = None
    if not optional:
        output_counts = torch.zeros((2, 10), dtype=torch.int32, device=DEVICE)
    sampled_base = torch.tensor(
        [[99, 98, -1], [3, 4, -1], [5, 6, -1]],
        dtype=torch.int64,
        device=DEVICE,
    )
    sampled = sampled_base[:, :2]
    num_sampled = torch.tensor([1, 2, 1], dtype=torch.int32, device=DEVICE)
    num_rejected = torch.tensor([9, 1, 0], dtype=torch.int32, device=DEVICE)
    query_start = None
    if not optional:
        query_start = torch.tensor([0, 2, 5, 6], dtype=torch.int32, device=DEVICE)
    all_token_ids_base = torch.full((2, 10), -1, dtype=torch.int32, device=DEVICE)
    all_token_ids = all_token_ids_base[:, :8]
    total_len = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    return output_counts, [
        idx_mapping,
        num_computed,
        last_sampled,
        sampled,
        num_sampled,
        num_rejected,
        query_start,
        all_token_ids,
        total_len,
    ]


def _run_post_update(output_counts, tensors) -> list[torch.Tensor | None]:
    post_update(
        tensors[0],
        tensors[1],
        tensors[2],
        output_counts,
        tensors[3],
        tensors[4],
        tensors[5],
        tensors[6],
        tensors[7],
        tensors[8],
    )
    _sync()
    return [output_counts, *tensors]


@pytest.mark.parametrize("optional", [False, True])
def test_post_update_direct_matches_fake_triton(
    monkeypatch: pytest.MonkeyPatch, optional: bool
) -> None:
    direct = _run_post_update(*_post_update_case(optional=optional))

    monkeypatch.setattr(mcpu_ops, "try_post_update", lambda *a: False)
    fallback = _run_post_update(*_post_update_case(optional=optional))

    for direct_tensor, fallback_tensor in zip(direct, fallback):
        if direct_tensor is None:
            assert fallback_tensor is None
        else:
            assert fallback_tensor is not None
            assert torch.equal(fallback_tensor.cpu(), direct_tensor.cpu())


def _run_scatter(monkeypatch: pytest.MonkeyPatch | None = None) -> torch.Tensor:
    if monkeypatch is not None:
        monkeypatch.setattr(mcpu_ops, "try_scatter_num_accepted", lambda *a: False)
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 77, dtype=torch.int32, device=DEVICE
    )
    state._align_mode = False
    state._mamba_ctx = None
    state.postprocess_state(
        torch.tensor([2, -1, 0, 3], dtype=torch.int32, device=DEVICE),
        torch.tensor([0, 5, -3, 6], dtype=torch.int32, device=DEVICE),
    )
    _sync()
    return state.num_accepted_tokens_gpu.cpu()


def test_scatter_num_accepted_direct_matches_fake_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = _run_scatter()
    fallback = _run_scatter(monkeypatch)
    assert torch.equal(fallback, direct)
    assert direct.tolist() == [1, 77, 1, 6]


@pytest.mark.parametrize(
    ("adapter_name", "kernel_name", "call"),
    [
        (
            "try_get_num_sampled_and_rejected",
            "_get_num_sampled_and_rejected_kernel",
            lambda: get_num_sampled_and_rejected(*_num_rejected_case()),
        ),
        (
            "try_post_update",
            "_post_update_kernel",
            lambda: _run_post_update(*_post_update_case()),
        ),
    ],
)
def test_registered_direct_op_skips_fake_launcher(
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
    kernel_name: str,
    call,
) -> None:
    module = input_batch_module
    monkeypatch.setattr(
        module,
        kernel_name,
        pytest.fail,
    )
    assert getattr(mcpu_ops, adapter_name)
    call()
    _sync()


def test_unavailable_operator_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "vllm_get_num_sampled_and_rejected"
    monkeypatch.setitem(mcpu_ops._OPS, name, None)
    assert mcpu_ops._resolve_op(name) is None
    assert mcpu_ops._resolve_op(name) is None


def test_non_mcpu_platform_does_not_resolve_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "vllm_get_num_sampled_and_rejected"
    monkeypatch.setattr(mcpu_ops, "_IS_MCPU", False)
    monkeypatch.setitem(mcpu_ops._OPS, name, mcpu_ops._UNRESOLVED)
    assert mcpu_ops._resolve_op(name) is None
    assert mcpu_ops._OPS[name] is mcpu_ops._UNRESOLVED
