# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache


class _FakeAttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype: str,
    ) -> tuple[int, int, int, int, int]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)


def test_reshape_kv_cache_supports_hybrid_attention_mamba() -> None:
    num_blocks = 3
    attn_spec = AttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=1,
        shapes=((3,), (2, 2)),
        dtypes=(torch.float16, torch.float16),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=attn_spec.page_size_bytes * num_blocks,
                shared_by=["layer.0.attn"],
            ),
            KVCacheTensor(
                size=mamba_spec.page_size_bytes * num_blocks,
                shared_by=["layer.1.mixer"],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer.0.attn"],
                kv_cache_spec=attn_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["layer.1.mixer"],
                kv_cache_spec=mamba_spec,
            ),
        ],
    )
    raw_tensors = {
        "layer.0.attn": torch.zeros(
            attn_spec.page_size_bytes * num_blocks, dtype=torch.int8
        ),
        "layer.1.mixer": torch.zeros(
            mamba_spec.page_size_bytes * num_blocks, dtype=torch.int8
        ),
    }

    kv_caches = _reshape_kv_cache(
        kv_cache_config,
        raw_tensors,
        {"layer.0.attn": _FakeAttentionBackend},
        cache_dtype="auto",
    )

    attn_cache = kv_caches["layer.0.attn"]
    assert isinstance(attn_cache, torch.Tensor)
    assert attn_cache.shape == (2, num_blocks, 4, 1, 2)
    assert attn_cache.stride() == (8, 16, 2, 2, 1)

    mamba_cache = kv_caches["layer.1.mixer"]
    assert isinstance(mamba_cache, list)
    assert len(mamba_cache) == 2
    assert mamba_cache[0].shape == (num_blocks, 3)
    assert mamba_cache[0].stride() == (7, 1)
    assert mamba_cache[0].storage_offset() == 0
    assert mamba_cache[1].shape == (num_blocks, 2, 2)
    assert mamba_cache[1].stride() == (7, 2, 1)
    assert mamba_cache[1].storage_offset() == 3
