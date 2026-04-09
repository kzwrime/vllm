# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure PyTorch paged attention backend (no C++ kernels)."""

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    is_quantized_kv_cache,
)
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

logger = init_logger(__name__)


class TorchAttentionBackend(AttentionBackend):
    """Pure PyTorch attention backend — no C++ / custom ops."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # PyTorch SDPA works with any head size; return empty to allow all.
        return []

    @staticmethod
    def get_name() -> str:
        return "TORCH_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["TorchAttentionImpl"]:
        return TorchAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TorchAttentionMetadataBuilder"]:
        return TorchAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # [K/V, num_blocks, num_kv_heads, block_size, head_size]
        return 2, num_blocks, num_kv_heads, block_size, head_size

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class TorchAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor  # (num_reqs + 1,) int32
    max_seq_len: int
    seq_lens: torch.Tensor  # (num_reqs,) int32
    block_table: torch.Tensor  # (num_reqs, max_blocks_per_seq) int32
    slot_mapping: torch.Tensor  # (num_tokens,) int64
    causal: bool = True


class TorchAttentionMetadataBuilder(AttentionMetadataBuilder[TorchAttentionMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(None, False)
        self.is_cross_attention = isinstance(kv_cache_spec, CrossAttentionSpec)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TorchAttentionMetadata:
        causal = False if self.is_cross_attention else common_attn_metadata.causal
        return TorchAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=causal,
        )


class TorchAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if alibi_slopes is not None:
            self.alibi_slopes: torch.Tensor | None = torch.tensor(
                alibi_slopes, dtype=torch.float32
            )
        else:
            self.alibi_slopes = None

        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap if logits_soft_cap else 0.0
        self.attn_type = attn_type
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.sinks = sinks

        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError(
                "FP8 KV cache is unsupported in TorchAttentionBackend"
            )

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_and_cache(
        key: torch.Tensor,  # (num_tokens, num_kv_heads, head_size)
        value: torch.Tensor,
        key_cache: torch.Tensor,  # (num_blocks, num_kv_heads, block_size, head_size)
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,  # (num_tokens,) int64
    ) -> None:
        """Scatter key/value tokens into the paged KV caches.

        key_cache layout: (num_blocks, num_kv_heads, block_size, head_size)
        slot = block_idx * block_size + block_offset
        """
        num_blocks, num_kv_heads, block_size, head_size = key_cache.shape
        block_idx = slot_mapping // block_size  # (num_tokens,)
        block_off = slot_mapping % block_size  # (num_tokens,)

        # Expand indices to (num_tokens, num_kv_heads, head_size)
        n = slot_mapping.shape[0]
        bi = block_idx[:, None, None].expand(n, num_kv_heads, head_size)
        hi = torch.arange(num_kv_heads, device=key.device)[None, :, None].expand(
            n, num_kv_heads, head_size
        )
        bo = block_off[:, None, None].expand(n, num_kv_heads, head_size)
        di = torch.arange(head_size, device=key.device)[None, None, :].expand(
            n, num_kv_heads, head_size
        )
        key_cache[bi, hi, bo, di] = key
        value_cache[bi, hi, bo, di] = value

    # ------------------------------------------------------------------
    # Core paged attention (pure PyTorch, inspired by ref_paged_attn)
    # ------------------------------------------------------------------

    def _paged_attention(
        self,
        query: torch.Tensor,  # (total_q, num_heads, head_size)
        key_cache: torch.Tensor,  # (num_blocks, num_kv_heads, block_size, head_size)
        value_cache: torch.Tensor,  # same shape
        output: torch.Tensor,  # (total_q, num_heads, head_size)
        query_start_loc: torch.Tensor,  # (num_seqs+1,) int32
        seq_lens: torch.Tensor,  # (num_seqs,) int32
        block_table: torch.Tensor,  # (num_seqs, max_blocks) int32
        causal: bool,
    ) -> None:
        num_seqs = seq_lens.shape[0]
        block_size = key_cache.shape[2]
        dtype = query.dtype

        # Work in float32 for numerical stability, convert back at the end.
        alibi_slopes = self.alibi_slopes
        if alibi_slopes is not None:
            alibi_slopes = alibi_slopes.float()

        query_start_loc_cpu = query_start_loc.cpu().numpy()
        seq_lens_cpu = seq_lens.cpu().numpy()
        block_table_cpu = block_table.cpu()

        for i in range(num_seqs):
            q_start = int(query_start_loc_cpu[i])
            q_end = int(query_start_loc_cpu[i + 1])
            query_len = q_end - q_start
            kv_len = int(seq_lens_cpu[i])

            q = query[q_start:q_end].float()  # (q_len, num_heads, head_size)
            q = q * self.scale

            # Gather K and V from paged cache
            num_kv_blocks = (kv_len + block_size - 1) // block_size
            blk_indices = block_table_cpu[i, :num_kv_blocks]  # (num_kv_blocks,)

            # key_cache layout: (num_blocks, num_kv_heads, block_size, head_size)
            # After gather: (num_kv_blocks, num_kv_heads, block_size, head_size)
            # Permute to (num_kv_blocks, block_size, num_kv_heads, head_size)
            # so that contiguous tokens are adjacent before reshape.
            k = key_cache[blk_indices]
            k = k.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_size)
            k = k[:kv_len].float()
            v = value_cache[blk_indices]
            v = v.permute(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_size)
            v = v[:kv_len].float()

            # Expand KV heads for GQA
            if self.num_queries_per_kv > 1:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

            # attn shape: (num_heads, query_len, kv_len)
            attn = torch.einsum("qhd,khd->hqk", q, k)

            # Causal mask
            if causal:
                empty_mask = torch.ones(query_len, kv_len, device=query.device)
                causal_mask = torch.triu(
                    empty_mask, diagonal=kv_len - query_len + 1
                ).bool()

                if self.sliding_window is not None:
                    sw_mask = (
                        torch.triu(
                            empty_mask,
                            diagonal=kv_len - (query_len + self.sliding_window) + 1,
                        )
                        .bool()
                        .logical_not()
                    )
                    causal_mask = causal_mask | sw_mask

                attn.masked_fill_(causal_mask, float("-inf"))

            # Softcap
            if self.logits_soft_cap:
                attn = self.logits_soft_cap * torch.tanh(attn / self.logits_soft_cap)

            # ALiBi bias
            if alibi_slopes is not None:
                q_start_pos = kv_len - query_len
                q_pos = (
                    q_start_pos
                    + torch.arange(0, query_len, device=query.device)[None, :, None]
                )
                kv_pos = torch.arange(0, kv_len, device=query.device)[None, None, :]
                dist = q_pos - kv_pos
                alibi_bias = -alibi_slopes.to(query.device)[:, None, None] * dist
                attn = attn + alibi_bias

            attn = torch.softmax(attn, dim=-1)

            out = torch.einsum("hqk,khd->qhd", attn, v).to(dtype=dtype)
            output[q_start:q_end] = out

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TorchAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported in TorchAttentionBackend"
            )

        # Warm-up path
        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Encoder / encoder-only path: no KV cache
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._run_sdpa_forward(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
            )

        # Decoder / encoder-decoder: use paged KV cache.
        # kv_cache shape: [2, num_blocks, num_kv_heads, block_size, head_size]
        key_cache, value_cache = kv_cache.unbind(0)

        # Update KV cache (skip for cross-attention sharing)
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            self._reshape_and_cache(
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                key_cache,
                value_cache,
                attn_metadata.slot_mapping[:num_actual_tokens],
            )

        if num_actual_tokens > 0:
            self._paged_attention(
                query=query[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                output=output[:num_actual_tokens],
                query_start_loc=attn_metadata.query_start_loc,
                seq_lens=attn_metadata.seq_lens,
                block_table=attn_metadata.block_table,
                causal=attn_metadata.causal,
            )

        return output

    # ------------------------------------------------------------------
    # SDPA path for encoder attention (no paged KV cache)
    # ------------------------------------------------------------------

    def _run_sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TorchAttentionMetadata,
    ) -> torch.Tensor:
        query_start_loc = attn_metadata.query_start_loc.cpu().numpy()
        num_seqs = attn_metadata.seq_lens.shape[0]
        causal_attn = self.attn_type == AttentionType.DECODER

        # (total_tokens, heads, head_size) -> (heads, total_tokens, head_size)
        q = query.movedim(0, query.dim() - 2)
        k = key.movedim(0, key.dim() - 2)
        v = value.movedim(0, value.dim() - 2)

        for i in range(num_seqs):
            s = int(query_start_loc[i])
            e = int(query_start_loc[i + 1])
            sub_out = (
                F.scaled_dot_product_attention(
                    q[None, :, s:e, :],
                    k[None, :, s:e, :],
                    v[None, :, s:e, :],
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=causal_attn,
                    scale=self.scale,
                    enable_gqa=self.num_heads > self.num_kv_heads,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[s:e] = sub_out
        return output
