# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any, cast

import torch

from vllm.platforms import current_platform

_IS_MCPU = current_platform.device_name == "mcpu"
_UNRESOLVED = object()
_OPS: dict[str, object | None] = {
    "vllm_get_num_sampled_and_rejected": _UNRESOLVED,
    "vllm_post_update": _UNRESOLVED,
    "vllm_scatter_num_accepted": _UNRESOLVED,
}


def is_mcpu() -> bool:
    return _IS_MCPU


def _resolve_op(name: str) -> Callable[..., Any] | None:
    if not _IS_MCPU:
        return None

    op = _OPS[name]
    if op is _UNRESOLVED:
        try:
            op = getattr(torch.ops.mcpu, name)
        except AttributeError:
            op = None
        _OPS[name] = op
    return cast(Callable[..., Any] | None, op)


def try_get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> bool:
    op = _resolve_op("vllm_get_num_sampled_and_rejected")
    if op is None:
        return False
    op(
        num_sampled,
        num_rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )
    return True


def try_post_update(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
) -> bool:
    op = _resolve_op("vllm_post_update")
    if op is None:
        return False
    op(
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
    )
    return True


def try_scatter_num_accepted(
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_accepted: torch.Tensor,
) -> bool:
    op = _resolve_op("vllm_scatter_num_accepted")
    if op is None:
        return False
    op(idx_mapping, num_sampled, num_accepted)
    return True
