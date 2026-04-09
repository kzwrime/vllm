# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
mcpu C++ implementations of Triton kernels used in the GPU model runner.

This module is loaded by the mcpu backend (torch_mcpu) to override the
pure-PyTorch fallbacks in torch_triton_utils.  Each function delegates to
a C++ operator registered under torch.ops.mcpu.vllm_* (implemented in
torch_mcpu/csrc/aten/vllm_kernels/).

Injection: torch_mcpu/__init__.py calls patch_torch_triton_utils() so that
every importer of vllm.utils.torch_triton_utils gets the mcpu implementations
without any changes to the vllm source tree.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import torch


class _FuncWrapper:
    """Wraps a function to support kernel[(grid)](*args) syntax."""

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __getitem__(self, *args, **kwargs) -> Callable:
        return self.func


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


# =============================================================================
# Structured Outputs - Grammar Bitmask
# =============================================================================


def _apply_grammar_bitmask_kernel_impl(
    logits: torch.Tensor,
    logits_stride: int,
    logits_indices: torch.Tensor,
    bitmask: torch.Tensor,
    bitmask_stride: int,
    vocab_size: int,
    BLOCK_SIZE: int = 8192,
) -> None:
    torch.ops.mcpu.vllm_apply_grammar_bitmask(
        logits, logits_indices, bitmask, vocab_size
    )


apply_grammar_bitmask_kernel = _FuncWrapper(_apply_grammar_bitmask_kernel_impl)


# =============================================================================
# Logit Bias
# =============================================================================


def _bias_kernel_impl(
    logits: torch.Tensor,
    logits_stride: int,
    vocab_size: int,
    expanded_idx_mapping: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    allowed_token_ids_stride: int,
    num_logit_bias: torch.Tensor,
    bias_token_ids_ptr: torch.Tensor,
    bias_token_ids_stride: int,
    bias_ptr: torch.Tensor,
    bias_stride: int,
    pos_ptr: torch.Tensor,
    min_lens_ptr: torch.Tensor,
    num_stop_token_ids_ptr: torch.Tensor,
    stop_token_ids_ptr: torch.Tensor,
    stop_token_ids_stride: int,
    BLOCK_SIZE: int = 1024,
    LOGITS_BLOCK_SIZE: int = 8192,
) -> None:
    torch.ops.mcpu.vllm_bias_kernel(
        logits,
        vocab_size,
        expanded_idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        num_logit_bias,
        bias_token_ids_ptr,
        bias_ptr,
        pos_ptr,
        min_lens_ptr,
        num_stop_token_ids_ptr,
        stop_token_ids_ptr,
    )


_bias_kernel = _FuncWrapper(_bias_kernel_impl)


# =============================================================================
# Bad Words
# =============================================================================


def _bad_words_kernel_impl(
    logits: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping: torch.Tensor,
    bad_word_token_ids: torch.Tensor,
    bad_word_token_ids_stride: int,
    bad_word_offsets: torch.Tensor,
    bad_word_offsets_stride: int,
    num_bad_words: torch.Tensor,
    all_token_ids: torch.Tensor,
    all_token_ids_stride: int,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
) -> None:
    torch.ops.mcpu.vllm_bad_words_kernel(
        logits,
        expanded_idx_mapping,
        bad_word_token_ids,
        bad_word_offsets,
        num_bad_words,
        all_token_ids,
        prompt_len,
        total_len,
        input_ids,
        expanded_local_pos,
    )


_bad_words_kernel = _FuncWrapper(_bad_words_kernel_impl)


# =============================================================================
# Min-P Sampling
# =============================================================================


def _min_p_kernel_impl(
    logits: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping: torch.Tensor,
    min_p: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int = 1024,
) -> None:
    torch.ops.mcpu.vllm_min_p_kernel(logits, expanded_idx_mapping, min_p, vocab_size)


_min_p_kernel = _FuncWrapper(_min_p_kernel_impl)


# =============================================================================
# Gumbel Sampling - Temperature & Random Generation
# =============================================================================


def _temperature_kernel_impl(
    logits: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int = 1024,
) -> None:
    torch.ops.mcpu.vllm_temperature_kernel(
        logits, expanded_idx_mapping, temperature, vocab_size
    )


_temperature_kernel = _FuncWrapper(_temperature_kernel_impl)


def _gumbel_sample_kernel_impl(
    logits: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int = 1024,
    processed_logits_out: torch.Tensor | None = None,
    apply_temperature: bool = False,
) -> torch.Tensor:
    return torch.ops.mcpu.vllm_gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        vocab_size,
        processed_logits_out,
        apply_temperature,
    )


_gumbel_sample_kernel = _FuncWrapper(_gumbel_sample_kernel_impl)


# =============================================================================
# Logprobs
# =============================================================================


def _topk_log_softmax_kernel_impl(
    output: torch.Tensor,
    logits: torch.Tensor,
    logits_stride: int,
    topk_ids: torch.Tensor,
    topk: int,
    vocab_size: int,
    BLOCK_SIZE: int = 1024,
    PADDED_TOPK: int = 1,
) -> None:
    torch.ops.mcpu.vllm_topk_log_softmax_kernel(
        output, logits, topk_ids, topk, vocab_size
    )


_topk_log_softmax_kernel = _FuncWrapper(_topk_log_softmax_kernel_impl)


def _ranks_kernel_impl(
    output: torch.Tensor,
    logits: torch.Tensor,
    logits_stride: int,
    token_ids: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int = 8192,
) -> None:
    torch.ops.mcpu.vllm_ranks_kernel(output, logits, token_ids, vocab_size)


_ranks_kernel = _FuncWrapper(_ranks_kernel_impl)


# =============================================================================
# Prompt Logprobs
# =============================================================================


def _prompt_logprobs_token_ids_kernel_impl(
    prompt_logprobs_token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
    all_token_ids_stride: int,
    BLOCK_SIZE: int = 1024,
) -> None:
    torch.ops.mcpu.vllm_prompt_logprobs_token_ids(
        prompt_logprobs_token_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
    )


_prompt_logprobs_token_ids_kernel = _FuncWrapper(_prompt_logprobs_token_ids_kernel_impl)


# =============================================================================
# Penalties - Repetition / Frequency / Presence
# =============================================================================


def _penalties_kernel_impl(
    logits: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    prompt_bin_mask_stride: int,
    output_bin_counts: torch.Tensor,
    output_bin_counts_stride: int,
    vocab_size: int,
    BLOCK_SIZE: int = 8192,
    MAX_SPEC_LEN: int = 0,
) -> None:
    torch.ops.mcpu.vllm_penalties_kernel(
        logits,
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
        vocab_size,
        MAX_SPEC_LEN,
    )


_penalties_kernel = _FuncWrapper(_penalties_kernel_impl)


# =============================================================================
# Penalties - Bincount (prompt token histogram)
# =============================================================================


def _bincount_kernel_impl(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    all_token_ids_stride: int,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    prompt_bin_mask_stride: int,
    output_bin_counts: torch.Tensor,
    output_bin_counts_stride: int,
    BLOCK_SIZE: int = 1024,
) -> None:
    torch.ops.mcpu.vllm_bincount_kernel(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
    )


_bincount_kernel = _FuncWrapper(_bincount_kernel_impl)


# =============================================================================
# Input-Batch Preparation
# =============================================================================


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    torch.ops.mcpu.vllm_prepare_prefill_inputs(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    )


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    torch.ops.mcpu.vllm_prepare_pos_seq_lens(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        pos,
        seq_lens,
    )


def post_update(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
) -> None:
    torch.ops.mcpu.vllm_post_update(
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )


def post_update_pool(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    torch.ops.mcpu.vllm_post_update_pool(
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
) -> torch.Tensor:
    return torch.ops.mcpu.vllm_combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits,
    )


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.mcpu.vllm_get_num_sampled_and_rejected(
        num_sampled,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.mcpu.vllm_expand_idx_mapping(
        idx_mapping,
        total_num_logits,
        cu_num_logits,
        max_expand_len,
    )


# =============================================================================
# Module injection
# =============================================================================


def patch_torch_triton_utils() -> None:
    """Replace vllm.utils.torch_triton_utils with this module.

    Called once by torch_mcpu during backend initialisation so that all
    vllm code that imports from torch_triton_utils gets mcpu implementations
    without touching any vllm source files.
    """
    sys.modules["vllm.utils.torch_triton_utils"] = sys.modules[__name__]
