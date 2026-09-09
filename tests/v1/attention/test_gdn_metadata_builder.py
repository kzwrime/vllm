# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass, fields

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
    mamba_cache_mode: str = "none",
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
    block_table: torch.Tensor | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    if block_table is not None:
        common.block_table_tensor = block_table
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


def _make_block_table(batch_spec: BatchSpec, first_block: int) -> torch.Tensor:
    max_blocks = (max(batch_spec.seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch_spec.batch_size * max_blocks
    return (
        torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).view(
            batch_spec.batch_size, max_blocks
        )
        + first_block
    )


def _assert_metadata_values_equal(
    actual: GDNAttentionMetadata, expected: GDNAttentionMetadata
) -> None:
    """Compare every observable metadata field, including nested tensors."""

    def assert_value_equal(actual_value, expected_value) -> None:
        if isinstance(actual_value, torch.Tensor):
            assert isinstance(expected_value, torch.Tensor)
            torch.testing.assert_close(actual_value, expected_value)
        elif isinstance(actual_value, dict):
            assert isinstance(expected_value, dict)
            assert actual_value.keys() == expected_value.keys()
            for key in actual_value:
                assert_value_equal(actual_value[key], expected_value[key])
        else:
            assert actual_value == expected_value

    for metadata_field in fields(GDNAttentionMetadata):
        assert_value_equal(
            getattr(actual, metadata_field.name),
            getattr(expected, metadata_field.name),
        )


GDN_UPDATE_CASES = {
    "pure_non_spec_decode": (
        BatchSpec(seq_lens=[40, 24], query_lens=[1, 1]),
        None,
        0,
    ),
    "pure_spec_decode": (
        BatchSpec(seq_lens=[80, 96], query_lens=[4, 4]),
        [3, 3],
        3,
    ),
    "mixed_prefill_decode_spec": (
        BatchSpec(seq_lens=[100, 65, 20], query_lens=[50, 1, 3]),
        [-1, -1, 2],
        2,
    ),
    "zero_length_padding": (
        BatchSpec(seq_lens=[16, 65, 20, 0], query_lens=[0, 1, 3, 0]),
        [-1, -1, 2, -1],
        2,
    ),
}


@pytest.mark.parametrize("mamba_cache_mode", ["all", "none", "align"])
@pytest.mark.parametrize("test_case", GDN_UPDATE_CASES.items())
def test_gdn_update_block_table_matches_fresh_build(
    mamba_cache_mode: str,
    test_case: tuple[str, tuple[BatchSpec, list[int] | None, int]],
):
    """Updating a cached plan must only change block-table-derived fields."""
    _, (batch, draft_tokens, num_speculative_tokens) = test_case
    builder_a = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        mamba_cache_mode=mamba_cache_mode,
    )
    builder_b = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        mamba_cache_mode=mamba_cache_mode,
    )
    block_table_a = _make_block_table(batch, first_block=0)
    block_table_b = _make_block_table(batch, first_block=1000)

    metadata_a = _build(builder_a, batch, draft_tokens, block_table_a)
    metadata_a_state = (
        None
        if metadata_a.spec_state_indices_tensor is None
        else metadata_a.spec_state_indices_tensor.clone()
    )
    metadata_a_non_spec_state = (
        None
        if metadata_a.non_spec_state_indices_tensor is None
        else metadata_a.non_spec_state_indices_tensor.clone()
    )
    metadata_b_updated = builder_b.update_block_table(
        metadata_a,
        block_table_b,
        torch.empty(0, dtype=torch.int64, device=DEVICE),
    )
    metadata_b_fresh = _build(builder_b, batch, draft_tokens, block_table_b)

    _assert_metadata_values_equal(metadata_b_updated, metadata_b_fresh)
    if metadata_a_state is not None:
        assert not torch.equal(
            metadata_a_state, metadata_b_updated.spec_state_indices_tensor
        )
    if metadata_a_non_spec_state is not None:
        assert not torch.equal(
            metadata_a_non_spec_state,
            metadata_b_updated.non_spec_state_indices_tensor,
        )

    # replace() must leave the cached metadata and its tensors untouched.
    if metadata_a_state is not None:
        torch.testing.assert_close(
            metadata_a.spec_state_indices_tensor, metadata_a_state
        )
    if metadata_a_non_spec_state is not None:
        torch.testing.assert_close(
            metadata_a.non_spec_state_indices_tensor, metadata_a_non_spec_state
        )


@pytest.mark.parametrize("use_spec_decode", [False, True])
def test_gdn_update_uses_current_builder_buffers_and_runtime_handle(
    monkeypatch: pytest.MonkeyPatch, use_spec_decode: bool
):
    """Updates must use builder B's full-cudagraph buffers and handle."""
    monkeypatch.setenv("VLLM_XCPU_GDN_COMPILE", "1")
    runtime_calls: list[tuple[GDNAttentionMetadataBuilder, GDNAttentionMetadata]] = []

    def record_runtime_metadata(
        builder: GDNAttentionMetadataBuilder,
        metadata: GDNAttentionMetadata,
    ) -> None:
        runtime_calls.append((builder, metadata))

    monkeypatch.setattr(
        GDNAttentionMetadataBuilder,
        "_set_runtime_metadata",
        record_runtime_metadata,
    )

    if use_spec_decode:
        batch = BatchSpec(
            seq_lens=[80, 96, 0, 0],
            query_lens=[4, 4, 0, 0],
        )
        draft_tokens = [3, 3, -1, -1]
        num_speculative_tokens = 3
    else:
        batch = BatchSpec(
            seq_lens=[40, 24, 0, 0],
            query_lens=[1, 1, 0, 0],
        )
        draft_tokens = None
        num_speculative_tokens = 0

    builder_a = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
        mamba_cache_mode="all",
    )
    builder_b = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
        mamba_cache_mode="all",
    )
    assert builder_a._xcpu_runtime_metadata_handle is not None
    assert builder_b._xcpu_runtime_metadata_handle is not None
    assert builder_a._xcpu_runtime_metadata_handle != (
        builder_b._xcpu_runtime_metadata_handle
    )

    block_table_a = _make_block_table(batch, first_block=0)
    block_table_b = _make_block_table(batch, first_block=1000)
    metadata_a = _build(builder_a, batch, draft_tokens, block_table_a)
    metadata_a_state = {
        name: None if value is None else value.clone()
        for name, value in (
            ("spec_state_indices_tensor", metadata_a.spec_state_indices_tensor),
            ("non_spec_state_indices_tensor", metadata_a.non_spec_state_indices_tensor),
        )
    }

    metadata_b_updated = builder_b.update_block_table(
        metadata_a,
        block_table_b,
        torch.empty(0, dtype=torch.int64, device=DEVICE),
    )
    metadata_b_fresh = _build(builder_b, batch, draft_tokens, block_table_b)
    _assert_metadata_values_equal(metadata_b_updated, metadata_b_fresh)

    for name, value in metadata_a_state.items():
        assert value is None or torch.equal(value, getattr(metadata_a, name))
    assert metadata_b_updated.xcpu_runtime_metadata_handle == (
        builder_b._xcpu_runtime_metadata_handle
    )
    assert runtime_calls[0][0] is builder_a
    assert runtime_calls[1][0] is builder_b
    assert runtime_calls[1][1].xcpu_runtime_metadata_handle == (
        builder_b._xcpu_runtime_metadata_handle
    )

    owned_buffers = {
        "spec_state_indices_tensor": builder_b.spec_state_indices_tensor,
        "spec_sequence_masks": builder_b.spec_sequence_masks,
        "spec_token_indx": builder_b.spec_token_indx,
        "non_spec_token_indx": builder_b.non_spec_token_indx,
        "spec_query_start_loc": builder_b.spec_query_start_loc,
        "num_accepted_tokens": builder_b.num_accepted_tokens,
        "non_spec_state_indices_tensor": builder_b.non_spec_state_indices_tensor,
        "non_spec_query_start_loc": builder_b.non_spec_query_start_loc,
    }
    for name, buffer in owned_buffers.items():
        value = getattr(metadata_b_updated, name)
        if value is not None:
            assert value.untyped_storage().data_ptr() == (
                buffer.untyped_storage().data_ptr()
            )


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)
