# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVQuantMode,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache, build_attn_metadata
from vllm.v1.worker.utils import AttentionGroup


class _FakeMetadataBuilder:
    supports_update_block_table = True

    def __init__(self):
        self.build_count = 0
        self.update_count = 0
        self.capture_count = 0
        self.built_metadata = []
        self.updated_from = []
        self.updated_block_tables = []
        self.updated_slot_mappings = []

    def build(self, **kwargs):
        self.build_count += 1
        metadata = object()
        self.built_metadata.append(metadata)
        return metadata

    def update_block_table(self, metadata, blk_table, slot_mapping):
        self.update_count += 1
        self.updated_from.append(metadata)
        self.updated_block_tables.append(blk_table)
        self.updated_slot_mappings.append(slot_mapping)
        return object()

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self.capture_count += 1
        return object()


class _OtherFakeMetadataBuilder(_FakeMetadataBuilder):
    pass


def _run_fake_metadata_build(
    builders: list[_FakeMetadataBuilder],
    specs: list[FullAttentionSpec],
    for_cudagraph_capture: bool = False,
):
    groups = []
    kv_cache_groups = []
    block_tables = []
    slot_mappings = []
    for i, (builder, spec) in enumerate(zip(builders, specs)):
        layer_name = f"layer{i}"
        groups.append(
            [
                AttentionGroup(
                    backend=object,
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                    kv_cache_group_id=i,
                    metadata_builders=[builder],
                )
            ]
        )
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)
        )
        block_tables.append(torch.zeros((1, 1), dtype=torch.int32))
        slot_mappings.append(torch.zeros(1, dtype=torch.int64))

    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=kv_cache_groups,
    )
    return build_attn_metadata(
        attn_groups=groups,
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        for_cudagraph_capture=for_cudagraph_capture,
    )


def _fake_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )


def test_build_attn_metadata_reuses_same_spec_and_builder_type():
    builder_a = _FakeMetadataBuilder()
    builder_b = _FakeMetadataBuilder()
    spec = _fake_spec()

    _run_fake_metadata_build([builder_a, builder_b], [spec, spec])
    assert builder_a.build_count == 1
    assert builder_b.build_count == 0
    assert builder_b.update_count == 1
    assert builder_b.updated_from == builder_a.built_metadata

    # The cache is local to one build_attn_metadata call.
    _run_fake_metadata_build([builder_a, builder_b], [spec, spec])
    assert builder_a.build_count == 2
    assert builder_b.update_count == 2


def test_v1_runner_reuses_same_spec_and_builder_type():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    builder_a = _FakeMetadataBuilder()
    builder_b = _FakeMetadataBuilder()
    spec = _fake_spec()
    groups = []
    kv_cache_groups = []
    block_tables = []
    slot_mappings = {}

    for i, builder in enumerate((builder_a, builder_b)):
        layer_name = f"layer{i}"
        groups.append(
            [
                AttentionGroup(
                    backend=object,
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                    kv_cache_group_id=i,
                    metadata_builders=[builder],
                )
            ]
        )
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)
        )
        block_table = torch.full((1, 1), i, dtype=torch.int32)
        block_table_manager = MagicMock()
        block_table_manager.get_device_tensor.return_value = block_table
        block_tables.append(block_table_manager)
        slot_mappings[i] = torch.full((1,), i, dtype=torch.int64)

    runner = MagicMock()
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=kv_cache_groups,
    )
    runner.attn_groups = groups
    runner.input_batch.block_table = block_tables
    runner.input_batch.num_computed_tokens_cpu_tensor = torch.zeros(
        1, dtype=torch.int32
    )
    runner.input_batch.num_prompt_tokens_cpu_tensor = torch.zeros(1, dtype=torch.int32)
    runner.optimistic_seq_lens_cpu = torch.ones(1, dtype=torch.int32)
    runner.query_start_loc.gpu = torch.tensor([0, 1], dtype=torch.int32)
    runner.query_start_loc.cpu = torch.tensor([0, 1], dtype=torch.int32)
    runner.seq_lens = torch.ones(1, dtype=torch.int32)
    runner.positions = torch.zeros(1, dtype=torch.int64)
    runner.routed_experts_initialized = False
    runner.use_async_spec_decode = False
    runner.dcp_world_size = 1
    runner.speculative_config = None
    runner.is_mm_prefix_lm = False
    runner._get_encoder_seq_lens.return_value = (None, None)

    GPUModelRunner._build_attention_metadata(
        runner,
        num_tokens=1,
        num_reqs=1,
        max_query_len=1,
        slot_mappings=slot_mappings,
    )

    assert builder_a.build_count == 1
    assert builder_b.build_count == 0
    assert builder_b.update_count == 1
    assert builder_b.updated_from == builder_a.built_metadata
    assert (
        builder_b.updated_block_tables[0]
        is block_tables[1].get_device_tensor.return_value
    )
    assert builder_b.updated_slot_mappings[0] is slot_mappings[1]


@pytest.mark.parametrize("different_key", ["spec", "builder"])
def test_build_attn_metadata_does_not_reuse_different_cache_key(different_key):
    builder_a = _FakeMetadataBuilder()
    builder_b = (
        _OtherFakeMetadataBuilder()
        if different_key == "builder"
        else _FakeMetadataBuilder()
    )
    specs = (
        [_fake_spec(), _fake_spec(32)]
        if different_key == "spec"
        else [_fake_spec()] * 2
    )

    _run_fake_metadata_build([builder_a, builder_b], specs)
    assert builder_a.build_count == 1
    assert builder_b.build_count == 1
    assert builder_a.update_count == 0
    assert builder_b.update_count == 0


def test_build_attn_metadata_does_not_cache_unsupported_builder():
    builder_a = _FakeMetadataBuilder()
    builder_b = _FakeMetadataBuilder()
    builder_a.supports_update_block_table = False
    builder_b.supports_update_block_table = False

    _run_fake_metadata_build([builder_a, builder_b], [_fake_spec()] * 2)
    assert builder_a.build_count == 1
    assert builder_b.build_count == 1
    assert builder_a.update_count == 0
    assert builder_b.update_count == 0


def test_build_attn_metadata_capture_bypasses_runtime_cache():
    builder_a = _FakeMetadataBuilder()
    builder_b = _FakeMetadataBuilder()

    _run_fake_metadata_build(
        [builder_a, builder_b], [_fake_spec()] * 2, for_cudagraph_capture=True
    )
    assert builder_a.build_count == 0
    assert builder_b.build_count == 0
    assert builder_a.update_count == 0
    assert builder_b.update_count == 0
    assert builder_a.capture_count == 1
    assert builder_b.capture_count == 1


class FakeFlashAttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


class FakeHNDFlashAttentionBackend(FakeFlashAttentionBackend):
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 3, 2, 4)


def test_reshape_padded_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )


def test_reshape_padded_hnd_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=3,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=1024,
    )
    assert spec.real_page_size_bytes == 768

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeHNDFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 3, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache.stride(2) == 2
    assert kv_cache.stride(3) == spec.block_size * spec.head_size
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )
    assert (
        kv_cache[1, 1, 3, 2].storage_offset()
        == (
            spec.page_size_bytes
            + spec.real_page_size_bytes // 2
            + 3 * spec.head_size * 4
            + 2 * spec.block_size * spec.head_size * 4
        )
        // 4
    )


class FakeDiffKVBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size * 2)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3)


def test_reshape_padded_diff_kv_cache_does_not_infer_kv_dim():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeDiffKVBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 16, 1, 4)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == 4


class FakePerTokenScaleBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size + 4)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


def test_reshape_padded_quantized_kv_cache_preserves_scale_stride():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 128
    assert spec.page_size_bytes == 384

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakePerTokenScaleBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "int8_per_token_head",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 8)
    assert kv_cache.stride(0) == spec.page_size_bytes
    assert kv_cache.stride(1) == 16 * 1 * 8
    assert kv_cache[1, 1].storage_offset() == spec.page_size_bytes + 16 * 1 * 8
