# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import VllmConfig


def _make_config(*, max_num_scheduled_tokens: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=32,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            long_prefill_token_threshold=0,
            disable_chunked_mm_input=False,
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


def test_mamba_align_rejects_scheduled_budget_smaller_than_block() -> None:
    config = _make_config(max_num_scheduled_tokens=8)

    with pytest.raises(
        ValueError,
        match=r"block_size \(16\) must be <= max_num_scheduled_tokens \(8\)",
    ):
        VllmConfig.validate_block_size(config)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_num_scheduled_tokens", [None, 16, 32])
def test_mamba_align_accepts_budget_covering_block(
    max_num_scheduled_tokens: int | None,
) -> None:
    config = _make_config(max_num_scheduled_tokens=max_num_scheduled_tokens)

    VllmConfig.validate_block_size(config)  # type: ignore[arg-type]
