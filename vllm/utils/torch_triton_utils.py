# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PyTorch fallback implementations for Triton kernels used in GPU model runner.

This module provides pure-PyTorch equivalents for Triton kernels, following
the same pattern as vllm/utils/cpu_triton_utils.py. Each kernel is wrapped
with _FuncWrapper to maintain the kernel[(grid)](*args) call syntax.
"""

from collections.abc import Callable

import torch


class _FuncWrapper:
    """Wraps a function to support kernel[(grid)](*args) syntax."""

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __getitem__(self, *args, **kwargs) -> Callable:
        return self.func


def _cdiv(a: int, b: int) -> int:
    """Ceiling division."""
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
    """Apply grammar bitmask to logits using PyTorch operations."""
    num_masks = logits_indices.shape[0]
    if num_masks == 0:
        return

    # Unpack bitmask: each int32 covers 32 vocab items
    bit_positions = torch.arange(32, device=bitmask.device)
    bitmask_expanded = bitmask.unsqueeze(-1)  # (num_masks, num_blocks, 1)
    bits = (bitmask_expanded >> bit_positions) & 1  # (num_masks, num_blocks, 32)
    unpacked_bitmask = bits.reshape(num_masks, -1)[:, :vocab_size]
    mask_bool = unpacked_bitmask.bool()

    # Apply mask to corresponding logits rows
    for i in range(num_masks):
        logits_idx = logits_indices[i].item()
        logits[logits_idx, :vocab_size][mask_bool[i]] = float("-inf")


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
    """Apply logit bias using PyTorch operations."""
    num_tokens = logits.shape[0]

    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()

        # 1. Allowed token IDs - set all to -inf except allowed ones
        num_allowed = num_allowed_token_ids[req_state_idx].item()
        if num_allowed > 0:
            allowed_ids = allowed_token_ids[req_state_idx, :num_allowed].clamp(
                max=vocab_size - 1
            )
            original_logits = logits[token_idx, allowed_ids].clone()
            logits[token_idx, :] = float("-inf")
            logits[token_idx, allowed_ids] = original_logits

        # 2. Logit bias - add bias values
        num_bias = num_logit_bias[req_state_idx].item()
        if num_bias > 0:
            bias_ids = bias_token_ids_ptr[req_state_idx, :num_bias].clamp(
                max=vocab_size - 1
            )
            bias_values = bias_ptr[req_state_idx, :num_bias]
            logits[token_idx, bias_ids] += bias_values

        # 3. Min tokens - mask stop tokens if pos < min_len
        num_stop = num_stop_token_ids_ptr[req_state_idx].item()
        if num_stop > 0:
            current_pos = pos_ptr[token_idx].item()
            min_len = min_lens_ptr[req_state_idx].item()
            if current_pos < min_len:
                stop_ids = stop_token_ids_ptr[req_state_idx, :num_stop].clamp(
                    max=vocab_size - 1
                )
                logits[token_idx, stop_ids] = float("-inf")


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
    """Apply bad words masking using PyTorch operations."""
    num_tokens = logits.shape[0]

    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        num_bad = num_bad_words[req_state_idx].item()
        if num_bad == 0:
            continue

        pos = expanded_local_pos[token_idx].item()
        cur_req_first_pos = token_idx - pos
        prompt_len_val = prompt_len[req_state_idx].item()
        total_len_val = total_len[req_state_idx].item()
        output_len = total_len_val - prompt_len_val
        effective_len = output_len + pos

        offsets = bad_word_offsets[req_state_idx]
        output_base = all_token_ids[req_state_idx, prompt_len_val:total_len_val]

        for bw_idx in range(num_bad):
            start = offsets[bw_idx].item()
            end = offsets[bw_idx + 1].item()
            prefix_len = end - start - 1

            if prefix_len > effective_len:
                continue

            last_token = bad_word_token_ids[req_state_idx, end - 1].item()

            match = True
            for i in range(prefix_len):
                expected = bad_word_token_ids[req_state_idx, start + i].item()
                actual_pos = effective_len - prefix_len + i

                if actual_pos >= output_len:
                    spec_offset = actual_pos - output_len
                    actual = input_ids[cur_req_first_pos + spec_offset].item()
                else:
                    actual = output_base[actual_pos].item()

                if expected != actual:
                    match = False
                    break

            if match and last_token < logits.shape[1]:
                logits[token_idx, last_token] = float("-inf")


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
    """Apply min-p sampling using PyTorch operations."""
    num_tokens = logits.shape[0]

    for token_idx in range(num_tokens):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        min_p_val = min_p[req_state_idx].item()

        if min_p_val == 0.0:
            continue

        max_val = logits[token_idx].max().item()
        threshold = (
            max_val
            + torch.log(
                torch.tensor(min_p_val, dtype=logits.dtype, device=logits.device)
            ).item()
        )

        mask = logits[token_idx] < threshold
        logits[token_idx][mask] = float("-inf")


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
    """Apply temperature scaling using PyTorch operations."""
    temp = temperature[expanded_idx_mapping].to(torch.float32)

    needs_scale = (temp != 0.0) & (temp != 1.0)
    if not needs_scale.any():
        return

    scale = torch.where(needs_scale, temp, torch.ones_like(temp))
    logits[needs_scale] = (
        logits[needs_scale].to(torch.float32) / scale[needs_scale].unsqueeze(-1)
    ).to(logits.dtype)


_temperature_kernel = _FuncWrapper(_temperature_kernel_impl)


def _tl_rand64(seed: int, pos: int, includes_zero: bool = False) -> float:
    """Generate a single fp64 uniform random value (compatibility wrapper)."""
    combined = seed ^ pos
    gen = torch.Generator(device="cpu")
    gen.manual_seed(combined & 0x7FFFFFFFFFFFFFFF)
    u = torch.empty(1, dtype=torch.float64, device="cpu")
    u.uniform_(generator=gen)
    if not includes_zero:
        tiny = torch.finfo(torch.float64).tiny
        u.clamp_(min=tiny)
    return u.item()


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
    """Gumbel-max sampling using PyTorch operations."""
    num_tokens, vocab_size = logits.shape
    logits_f32 = logits.to(torch.float32)
    temp = temperature[expanded_idx_mapping].to(torch.float32)

    if apply_temperature:
        needs_scale = (temp != 0.0) & (temp != 1.0)
        if needs_scale.any():
            scale = torch.where(needs_scale, temp, torch.ones_like(temp))
            logits_f32[needs_scale] = logits_f32[needs_scale] / scale[
                needs_scale
            ].unsqueeze(-1)

    if processed_logits_out is not None:
        req_indices = expanded_idx_mapping.unsqueeze(-1).expand(-1, vocab_size)
        processed_logits_out.scatter_(0, req_indices, logits_f32)

    logits_f64 = logits_f32.to(torch.float64)
    stochastic = temp != 0.0

    if stochastic.any():
        req_seeds = seed[expanded_idx_mapping]
        token_pos = pos.to(req_seeds.dtype)
        combined_seeds = req_seeds ^ token_pos

        # Generate random values on CPU and move to device
        u = torch.empty(num_tokens, vocab_size, dtype=torch.float64, device="cpu")
        u.uniform_()
        for i in range(num_tokens):
            if stochastic[i]:
                gen = torch.Generator(device="cpu")
                raw = int(combined_seeds[i].item())
                gen.manual_seed(raw & 0x7FFFFFFFFFFFFFFF)
                u[i].uniform_(generator=gen)

        tiny = torch.finfo(torch.float64).tiny
        u.clamp_(min=tiny)
        u = u.to(logits.device)

        gumbel = -torch.log(-torch.log(u))
        stochastic_2d = stochastic.unsqueeze(-1)
        logits_f64 = torch.where(stochastic_2d, logits_f64 + gumbel, logits_f64)

    return logits_f64.argmax(dim=-1)


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
    """Compute log softmax for specific token IDs using PyTorch operations."""
    batch_size = logits.shape[0]
    max_val = logits.max(dim=1, keepdim=True)[0]
    shifted_logits = logits - max_val
    exp_logits = torch.exp(shifted_logits)
    lse = torch.log(exp_logits.sum(dim=1, keepdim=True)) + max_val

    token_ids_clamped = topk_ids.clamp(max=vocab_size - 1)
    batch_indices = torch.arange(batch_size, device=logits.device).unsqueeze(1)
    selected_logits = logits[batch_indices, token_ids_clamped]
    logprobs = selected_logits - lse

    output.copy_(logprobs.to(torch.float32))


_topk_log_softmax_kernel = _FuncWrapper(_topk_log_softmax_kernel_impl)


def _ranks_kernel_impl(
    output: torch.Tensor,
    logits: torch.Tensor,
    logits_stride: int,
    token_ids: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int = 8192,
) -> None:
    """Compute ranks of sampled tokens using PyTorch operations."""
    batch_size = logits.shape[0]
    for req_idx in range(batch_size):
        token_id = token_ids[req_idx].item()
        x = logits[req_idx, token_id]
        rank = (logits[req_idx] >= x).sum().item()
        output[req_idx] = rank


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
    """Get token IDs for prompt logprobs using PyTorch operations."""
    num_reqs = idx_mapping.shape[0]

    for batch_idx in range(num_reqs):
        req_state_idx = idx_mapping[batch_idx].item()
        query_start = query_start_loc[batch_idx].item()
        query_end = query_start_loc[batch_idx + 1].item()
        query_len = query_end - query_start
        num_computed = num_computed_tokens[req_state_idx].item()

        for i in range(query_len):
            target_pos = num_computed + 1 + i
            token_id = all_token_ids[req_state_idx, target_pos].item()
            prompt_logprobs_token_ids[query_start + i] = token_id


_prompt_logprobs_token_ids_kernel = _FuncWrapper(_prompt_logprobs_token_ids_kernel_impl)


# =============================================================================
# Penalties - Repetition / Frequency / Presence
# =============================================================================


def _penalties_kernel_impl(
    logits: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,  # flat [total_tokens] 1-D, speculative draft tokens
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,  # [max_num_reqs, cdiv(vocab_size, 32)] packed int32
    prompt_bin_mask_stride: int,
    output_bin_counts: torch.Tensor,  # [max_num_reqs, vocab_size] int32
    output_bin_counts_stride: int,
    vocab_size: int,
    BLOCK_SIZE: int = 8192,
    MAX_SPEC_LEN: int = 0,
) -> None:
    """Apply repetition, frequency, presence penalties.

    Mirrors vllm/v1/worker/gpu/sample/penalties.py::_penalties_kernel.
    """
    num_tokens = logits.shape[0]

    for token_idx in range(num_tokens):
        req_idx = int(expanded_idx_mapping[token_idx].item())
        rep_pen = float(repetition_penalty[req_idx].item())
        freq_pen = float(frequency_penalty[req_idx].item())
        pres_pen = float(presence_penalty[req_idx].item())

        if rep_pen == 1.0 and freq_pen == 0.0 and pres_pen == 0.0:
            continue

        # Base output counts (pre-computed by bincount()).
        base_counts = output_bin_counts[req_idx, :vocab_size].to(torch.int32)

        # Add draft-token counts for speculative decoding (positions 1..pos).
        pos = int(expanded_local_pos[token_idx].item())
        if pos > 0 and MAX_SPEC_LEN > 0:
            start_idx = token_idx - pos
            for prev_pos in range(pos):
                draft_tok = int(token_ids[start_idx + prev_pos + 1].item())
                if 0 <= draft_tok < vocab_size:
                    base_counts = base_counts.clone()
                    base_counts[draft_tok] += 1

        output_mask = base_counts > 0
        logits_row = logits[token_idx].to(torch.float32)

        if rep_pen != 1.0:
            # Unpack prompt_bin_mask (packed int32 → bool per token).
            packed = prompt_bin_mask[req_idx]  # [cdiv(vocab_size, 32)]
            bit_pos = torch.arange(32, device=packed.device)
            bits = (packed.unsqueeze(-1) >> bit_pos) & 1
            prompt_mask = bits.reshape(-1)[:vocab_size].bool()
            gen_mask = prompt_mask | output_mask
            # Positive logits: divide by penalty; negative: multiply.
            scale = torch.where(gen_mask, rep_pen, 1.0)
            logits_row = torch.where(
                logits_row > 0, logits_row / scale, logits_row * scale
            )

        logits_row = logits_row - freq_pen * base_counts.to(logits_row.dtype)
        logits_row = logits_row - pres_pen * output_mask.to(logits_row.dtype)
        logits[token_idx] = logits_row.to(logits.dtype)


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
    prompt_bin_mask: torch.Tensor,  # [max_num_reqs, cdiv(vocab_size, 32)] int32
    prompt_bin_mask_stride: int,
    output_bin_counts: torch.Tensor,  # [max_num_reqs, vocab_size] int32
    output_bin_counts_stride: int,
    BLOCK_SIZE: int = 1024,
) -> None:
    """Populate prompt_bin_mask (packed bits) and output_bin_counts.

    Mirrors vllm/v1/worker/gpu/sample/penalties.py::_bincount_kernel.
    """
    num_reqs = expanded_idx_mapping.shape[0]
    packed_cols = prompt_bin_mask.shape[1]

    for token_idx in range(num_reqs):
        req_idx = int(expanded_idx_mapping[token_idx].item())
        plen = int(prompt_len[req_idx].item())
        flen = int(prefill_len[req_idx].item())

        # Pack prompt token IDs into prompt_bin_mask as bit flags.
        for pos in range(plen):
            tid = int(all_token_ids[req_idx, pos].item())
            word = tid // 32
            bit = tid % 32
            if 0 <= word < packed_cols:
                prompt_bin_mask[req_idx, word] |= 1 << bit

        # Count output (prefill-beyond-prompt) token occurrences.
        for pos in range(plen, flen):
            tid = int(all_token_ids[req_idx, pos].item())
            if 0 <= tid < output_bin_counts.shape[1]:
                output_bin_counts[req_idx, tid] += 1


_bincount_kernel = _FuncWrapper(_bincount_kernel_impl)


# =============================================================================
# Input-Batch Preparation (equivalents of input_batch.py Triton kernels)
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
    """Fill input_ids and next_prefill_tokens for prefill requests.

    Mirrors vllm/v1/worker/gpu/input_batch.py::_prepare_prefill_inputs_kernel.
    """
    num_reqs = idx_mapping.shape[0]
    for batch_idx in range(num_reqs):
        req = int(idx_mapping[batch_idx].item())
        plen = int(prefill_len[req].item())
        ncomp = int(num_computed_tokens[req].item())
        if ncomp >= plen:
            continue
        qstart = int(query_start_loc[batch_idx].item())
        qend = int(query_start_loc[batch_idx + 1].item())
        qlen = qend - qstart
        input_ids[qstart:qend] = all_token_ids[req, ncomp : ncomp + qlen]
        next_pos = ncomp + qlen
        if next_pos < plen:
            next_prefill_tokens[req] = all_token_ids[req, next_pos]


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Fill position IDs and sequence lengths for the current batch.

    Mirrors vllm/v1/worker/gpu/input_batch.py::_prepare_pos_seq_lens_kernel.
    """
    num_reqs = idx_mapping.shape[0]
    max_num_reqs = seq_lens.shape[0]
    for req_id in range(num_reqs):
        req = int(idx_mapping[req_id].item())
        ncomp = int(num_computed_tokens[req].item())
        start = int(query_start_loc[req_id].item())
        end = int(query_start_loc[req_id + 1].item())
        qlen = end - start
        seq_lens[req_id] = ncomp + qlen
        pos[start:end] = torch.arange(qlen, dtype=pos.dtype, device=pos.device) + ncomp
    # Pad unused rows to 0 (CUDA graph compatibility).
    if num_reqs < max_num_reqs:
        seq_lens[num_reqs:max_num_reqs] = 0


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
    """Update state after each sampling step.

    Mirrors vllm/v1/worker/gpu/input_batch.py::_post_update_kernel.
    """
    num_reqs = idx_mapping.shape[0]
    for req_id in range(num_reqs):
        req = int(idx_mapping[req_id].item())
        n = int(num_sampled[req_id].item())
        tlen = int(total_len[req].item())
        if n > 0:
            last_sampled_tokens[req] = sampled_tokens[req_id, n - 1]
            total_len[req] = tlen + n
            all_token_ids[req, tlen : tlen + n] = sampled_tokens[req_id, :n]
            if output_bin_counts is not None:
                for tok in sampled_tokens[req_id, :n].tolist():
                    output_bin_counts[req, tok] += 1
        qstart = int(query_start_loc[req_id].item())
        qend = int(query_start_loc[req_id + 1].item())
        qlen = qend - qstart
        nr = int(num_rejected[req_id].item())
        num_computed_tokens[req] = num_computed_tokens[req] + qlen - nr


def post_update_pool(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Update num_computed_tokens for pooling (no sampling).

    Mirrors vllm/v1/worker/gpu/input_batch.py::_post_update_pool_kernel.
    """
    num_reqs = idx_mapping.shape[0]
    for batch_id in range(num_reqs):
        req = int(idx_mapping[batch_id].item())
        qstart = int(query_start_loc[batch_id].item())
        qend = int(query_start_loc[batch_id + 1].item())
        num_computed_tokens[req] = num_computed_tokens[req] + (qend - qstart)


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
    """Combine sampled and draft tokens into input_ids; return logits_indices.

    Mirrors vllm/v1/worker/gpu/input_batch.py::_combine_sampled_and_draft_tokens_kernel.
    """
    num_reqs = idx_mapping.shape[0]
    logits_indices = torch.empty(num_logits, dtype=torch.int64, device=input_ids.device)
    for batch_idx in range(num_reqs):
        req = int(idx_mapping[batch_idx].item())
        logits_start = int(cu_num_logits[batch_idx].item())
        logits_end = int(cu_num_logits[batch_idx + 1].item())
        n_logits = logits_end - logits_start
        n_draft = n_logits - 1
        query_end = int(query_start_loc[batch_idx + 1].item())
        pos_start = query_end - n_logits
        logits_indices[logits_start:logits_end] = torch.arange(
            pos_start, pos_start + n_logits, dtype=torch.int64, device=input_ids.device
        )
        seq_len = int(seq_lens[batch_idx].item())
        plen = int(prefill_len[req].item())
        if seq_len <= plen:
            continue  # chunked prefill: no sampled/draft tokens
        input_ids[query_end - n_logits] = last_sampled_tokens[req]
        if n_draft > 0:
            input_ids[query_end - n_draft : query_end] = draft_tokens[req, :n_draft]
    return logits_indices


def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute num_sampled and num_rejected for each request.

    Mirrors vllm/v1/worker/gpu/input_batch.py::_get_num_sampled_and_rejected_kernel.
    """
    num_reqs = idx_mapping.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    for batch_idx in range(num_reqs):
        req = int(idx_mapping[batch_idx].item())
        seq_len = int(seq_lens[batch_idx].item())
        plen = int(prefill_len[req].item())
        is_chunked = seq_len < plen
        logits_start = int(cu_num_logits[batch_idx].item())
        logits_end = int(cu_num_logits[batch_idx + 1].item())
        n_logits = logits_end - logits_start
        if is_chunked:
            num_sampled[batch_idx] = 0
            num_rejected[batch_idx] = 0
        else:
            num_rejected[batch_idx] = n_logits - num_sampled[batch_idx]
    return num_sampled, num_rejected


def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand request index mapping and produce local position indices.

    Mirrors vllm/v1/worker/gpu/input_batch.py::_expand_idx_mapping_kernel.
    """
    num_reqs = idx_mapping.shape[0]
    device = idx_mapping.device
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(total_num_logits, dtype=torch.int32, device=device)
    for req_idx in range(num_reqs):
        start = int(cu_num_logits[req_idx].item())
        end = int(cu_num_logits[req_idx + 1].item())
        n = end - start
        expanded_idx_mapping[start:end] = idx_mapping[req_idx]
        expanded_local_pos[start:end] = torch.arange(
            n, dtype=torch.int32, device=device
        )
    return expanded_idx_mapping, expanded_local_pos
