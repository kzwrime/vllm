# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu import mcpu_ops
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample import sampler as sampler_module
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rejection_module
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device(current_platform.device_type)
VOCAB_SIZE = 16


def _make_sampler(max_num_reqs: int = 4) -> Sampler:
    req_states = RequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=32,
        max_num_batched_tokens=32,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(max_num_reqs, VOCAB_SIZE, DEVICE, req_states)


@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        pytest.param(SamplingParams(), True, id="defaults"),
        pytest.param(SamplingParams(temperature=0.0), True, id="greedy"),
        pytest.param(SamplingParams(top_k=-1), True, id="top-k-minus-one"),
        pytest.param(SamplingParams(top_k=VOCAB_SIZE), True, id="top-k-vocab"),
        pytest.param(SamplingParams(top_k=VOCAB_SIZE + 1), True, id="top-k-large"),
        pytest.param(
            SamplingParams(allowed_token_ids=[1]), False, id="allowed-token-ids"
        ),
        pytest.param(SamplingParams(logit_bias={1: 0.5}), False, id="logit-bias"),
        pytest.param(SamplingParams(min_tokens=1), False, id="min-tokens"),
        pytest.param(SamplingParams(_bad_words_token_ids=[[1]]), False, id="bad-words"),
        pytest.param(
            SamplingParams(repetition_penalty=1.1), False, id="repetition-penalty"
        ),
        pytest.param(
            SamplingParams(frequency_penalty=0.1), False, id="frequency-penalty"
        ),
        pytest.param(
            SamplingParams(presence_penalty=0.1), False, id="presence-penalty"
        ),
        pytest.param(SamplingParams(temperature=0.5), False, id="temperature"),
        pytest.param(SamplingParams(min_p=0.1), False, id="min-p"),
        pytest.param(SamplingParams(top_k=4), False, id="top-k"),
        pytest.param(SamplingParams(top_p=0.9), False, id="top-p"),
    ],
)
def test_default_sampling_params_eligibility(
    sampling_params: SamplingParams, expected: bool
) -> None:
    sampler = _make_sampler()
    sampler.add_request(2, prompt_len=3, sampling_params=sampling_params)
    assert sampler._default_sampling_params[2] == expected


def test_default_sampling_params_overwritten_when_slot_is_reused() -> None:
    sampler = _make_sampler()
    sampler.add_request(1, 3, SamplingParams(top_p=0.9))
    assert not sampler._default_sampling_params[1]

    sampler.add_request(1, 3, SamplingParams())
    assert sampler._default_sampling_params[1]


def _sampling_inputs(num_rows: int) -> tuple[torch.Tensor, ...]:
    idx = torch.arange(num_rows, dtype=torch.int32, device=DEVICE)
    return (
        idx,
        torch.arange(num_rows, dtype=torch.int64, device=DEVICE),
        torch.zeros(num_rows, dtype=torch.int32, device=DEVICE),
        torch.zeros(num_rows, dtype=torch.int32, device=DEVICE),
    )


def test_default_batch_fast_path_and_mixed_batch_fallback() -> None:
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams())
    sampler.add_request(1, 1, SamplingParams(top_p=0.9))
    expanded_idx, pos, input_ids, local_pos = _sampling_inputs(2)
    logits = torch.randn(2, VOCAB_SIZE, dtype=torch.bfloat16, device=DEVICE)

    fast = sampler.apply_sampling_params(
        logits[:1],
        expanded_idx[:1],
        np.array([0], dtype=np.int32),
        pos[:1],
        input_ids[:1],
        local_pos[:1],
    ).clone()
    fallback = sampler.apply_sampling_params(
        logits,
        expanded_idx,
        np.array([0, 1], dtype=np.int32),
        pos,
        input_ids,
        local_pos,
        skip_top_k_top_p=True,
    ).clone()
    torch.mcpu.synchronize()

    torch.testing.assert_close(fast.cpu(), logits[:1].float().cpu())
    torch.testing.assert_close(fallback.cpu(), logits.float().cpu())
    assert sampler.default_sampling_fast_path_hits == 1
    assert sampler.default_sampling_fast_path_fallbacks == 1


def test_fast_path_matches_forced_generic_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(seed=7))
    expanded_idx, pos, input_ids, local_pos = _sampling_inputs(1)
    idx_mapping_np = np.array([0], dtype=np.int32)
    logits = torch.randn(1, VOCAB_SIZE, dtype=torch.bfloat16, device=DEVICE)
    monkeypatch.setattr(
        sampler_module,
        "gumbel_sample",
        lambda processed_logits, *args, **kwargs: processed_logits.argmax(dim=-1),
    )

    sampler._default_sampling_params[0] = False
    expected_tokens, expected_logits = sampler.sample(
        logits,
        expanded_idx,
        idx_mapping_np,
        pos,
        input_ids,
        local_pos,
    )
    expected_tokens = expected_tokens.clone()
    expected_logits = expected_logits.clone()

    sampler._default_sampling_params[0] = True
    actual_tokens, actual_logits = sampler.sample(
        logits,
        expanded_idx,
        idx_mapping_np,
        pos,
        input_ids,
        local_pos,
    )
    torch.mcpu.synchronize()

    assert torch.equal(actual_tokens.cpu(), expected_tokens.cpu())
    torch.testing.assert_close(actual_logits.cpu(), expected_logits.cpu())


def test_default_sample_skips_redundant_top_k_top_p_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    expanded_idx, pos, input_ids, local_pos = _sampling_inputs(1)
    monkeypatch.setattr(
        sampler.sampling_states,
        "get_top_k_top_p",
        lambda *args: pytest.fail("default sampling must not recheck top-k/top-p"),
    )
    monkeypatch.setattr(
        sampler_module,
        "gumbel_sample",
        lambda processed_logits, *args, **kwargs: processed_logits.argmax(dim=-1),
    )

    sampler.sample(
        torch.randn(1, VOCAB_SIZE, dtype=torch.bfloat16, device=DEVICE),
        expanded_idx,
        np.array([0], dtype=np.int32),
        pos,
        input_ids,
        local_pos,
    )


@pytest.mark.skipif(not mcpu_ops.is_mcpu(), reason="MCPU-only buffer reuse")
def test_processed_logits_buffer_reuse_and_reallocation() -> None:
    sampler = _make_sampler()

    first = sampler._copy_logits_to_fp32(
        torch.randn(3, 8, dtype=torch.bfloat16, device=DEVICE)
    )
    first_ptr = first.data_ptr()
    smaller = sampler._copy_logits_to_fp32(
        torch.randn(1, 8, dtype=torch.bfloat16, device=DEVICE)
    )
    assert smaller.shape == (1, 8)
    assert smaller.data_ptr() == first_ptr

    larger = sampler._copy_logits_to_fp32(
        torch.randn(5, 8, dtype=torch.bfloat16, device=DEVICE)
    )
    assert larger.shape == (5, 8)
    assert larger.data_ptr() != first_ptr
    larger_ptr = larger.data_ptr()

    new_dtype = sampler._copy_logits_to_fp32(
        torch.randn(5, 8, dtype=torch.float32, device=DEVICE)
    )
    assert new_dtype.data_ptr() != larger_ptr
    dtype_ptr = new_dtype.data_ptr()

    new_tail = sampler._copy_logits_to_fp32(
        torch.randn(2, 4, dtype=torch.float32, device=DEVICE)
    )
    assert new_tail.shape == (2, 4)
    assert new_tail.data_ptr() != dtype_ptr

    cpu_result = sampler._copy_logits_to_fp32(torch.randn(2, 4, device="cpu"))
    assert cpu_result.device.type == "cpu"
    assert cpu_result.dtype == torch.float32


def _num_rejected_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([1, 2], dtype=torch.int32, device=DEVICE),
        torch.tensor([5, 2], dtype=torch.int32, device=DEVICE),
        torch.tensor([0, 4, 7], dtype=torch.int32, device=DEVICE),
        torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
        torch.tensor([4, 6], dtype=torch.int32, device=DEVICE),
    )


def test_num_rejected_out_reuses_prefix_and_none_preserves_api() -> None:
    inputs = _num_rejected_inputs()
    out = torch.empty(4, dtype=torch.int32, device=DEVICE)
    num_sampled, num_rejected = get_num_sampled_and_rejected(*inputs, out=out)
    torch.mcpu.synchronize()
    assert num_rejected.shape == (2,)
    assert num_rejected.data_ptr() == out.data_ptr()
    assert num_sampled.cpu().tolist() == [1, 0]
    assert num_rejected.cpu().tolist() == [3, 0]

    inputs = _num_rejected_inputs()
    _, allocated = get_num_sampled_and_rejected(*inputs)
    torch.mcpu.synchronize()
    assert allocated.shape == (2,)
    assert allocated.data_ptr() != inputs[0].data_ptr()


@pytest.mark.parametrize("kind", ["device", "dtype", "ndim", "stride", "capacity"])
def test_num_rejected_out_validation(kind: str) -> None:
    inputs = _num_rejected_inputs()
    if kind == "device":
        out = torch.empty(2, dtype=torch.int32, device="cpu")
    elif kind == "dtype":
        out = torch.empty(2, dtype=torch.int64, device=DEVICE)
    elif kind == "ndim":
        out = torch.empty((1, 2), dtype=torch.int32, device=DEVICE)
    elif kind == "stride":
        out = torch.empty(4, dtype=torch.int32, device=DEVICE)[::2]
    else:
        out = torch.empty(1, dtype=torch.int32, device=DEVICE)

    with pytest.raises(ValueError):
        get_num_sampled_and_rejected(*inputs, out=out)


def test_sampler_and_rejection_sampler_pass_owned_num_rejected_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams())
    expected_ptr = sampler._num_rejected_buffer.data_ptr()
    seen_ptrs: list[int] = []
    original = get_num_sampled_and_rejected

    def capture_out(*args, out=None, **kwargs):
        assert out is not None
        seen_ptrs.append(out.data_ptr())
        return original(*args, out=out, **kwargs)

    sampler_module_obj = __import__(
        "vllm.v1.worker.gpu.sample.sampler", fromlist=["get_num_sampled_and_rejected"]
    )
    monkeypatch.setattr(sampler_module_obj, "get_num_sampled_and_rejected", capture_out)
    monkeypatch.setattr(rejection_module, "get_num_sampled_and_rejected", capture_out)

    sampled = torch.tensor([3], dtype=torch.int64, device=DEVICE)
    monkeypatch.setattr(sampler, "sample", lambda *args, **kwargs: (sampled, args[0]))
    input_batch = SimpleNamespace(
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        idx_mapping_np=np.array([0], dtype=np.int32),
        cu_num_logits_np=np.array([0, 1], dtype=np.int32),
        expanded_local_pos=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        positions=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        input_ids=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        logits_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        cu_num_logits=torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
        idx_mapping=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        num_reqs=1,
    )
    sampler(torch.randn(1, VOCAB_SIZE, device=DEVICE), input_batch)

    rejection = object.__new__(RejectionSampler)
    rejection.sampler = sampler
    rejection.num_speculative_steps = 1
    rejection.use_block_verification = False
    rejection.synthetic_conditional_rates = None
    num_sampled = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    monkeypatch.setattr(
        rejection_module,
        "rejection_sample",
        lambda *args, **kwargs: (sampled.view(1, 1), num_sampled),
    )
    rejection(
        torch.randn(1, VOCAB_SIZE, device=DEVICE),
        input_batch,
        draft_logits=None,
    )

    assert seen_ptrs == [expected_ptr, expected_ptr]
