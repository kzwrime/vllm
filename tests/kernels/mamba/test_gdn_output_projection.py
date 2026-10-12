# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape contract for ``QwenGatedDeltaNetAttention._output_projection``.

Every ``forward_*`` already allocates ``core_attn_out`` and ``z`` as
``(N, H, D)``. The helper now norms those tensors as-is and flattens once
for ``out_proj``; it used to be rank-agnostic via ``reshape(z_shape_og)``.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)


@pytest.mark.parametrize("dtype", [torch.float32])
@torch.inference_mode()
def test_output_projection_norms_per_head_and_flattens(
    default_vllm_config,
    dtype: torch.dtype,
) -> None:
    num_tokens, num_heads, head_dim = 3, 4, 8
    hidden = num_heads * head_dim

    layer = types.SimpleNamespace()
    layer.norm = RMSNormGated(
        head_dim,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device="cpu",
        dtype=dtype,
    )
    layer.out_proj = lambda x: (x, None)
    layer._output_projection = types.MethodType(
        QwenGatedDeltaNetAttention._output_projection, layer
    )

    core_attn_out = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    z = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    out = layer._output_projection(core_attn_out, z)

    assert core_attn_out.shape == (num_tokens, num_heads, head_dim)
    assert z.shape == (num_tokens, num_heads, head_dim)
    assert out.shape == (num_tokens, hidden)
    assert out.dtype == dtype


@pytest.mark.parametrize("metadata", [None, {}, {"other_layer": None}])
def test_mcpu_dummy_core_initializes_output(monkeypatch, metadata):
    """Dummy runs must not pass uninitialized output to the gated norm."""
    monkeypatch.setattr(
        gdn, "current_platform", types.SimpleNamespace(device_name="mcpu")
    )
    monkeypatch.setattr(
        gdn,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata=metadata),
    )
    layer = types.SimpleNamespace(
        prefix="layer", _warmup_prefill_kernels=lambda *args: None
    )
    output = torch.full((5, 2, 4), float("nan"))
    qkv, gates = torch.empty(5, 16), torch.empty(5, 2)
    QwenGatedDeltaNetAttention._forward_core(layer, qkv, gates, gates, output)
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize("padding", [0, 3])
def test_mcpu_packed_decode_initializes_only_padding(monkeypatch, padding):
    """The packed early return must preserve written tokens and zero padding."""
    monkeypatch.setattr(
        gdn, "current_platform", types.SimpleNamespace(device_name="mcpu")
    )
    metadata = gdn.GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
    )
    monkeypatch.setattr(
        gdn,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={"layer": metadata}),
    )
    expected = torch.arange(16.0).reshape(2, 2, 4)

    def packed_decode(**kwargs):
        output = kwargs["core_attn_out"]
        # Real rows need no initialization: the kernel overwrites every value.
        assert torch.isnan(output[:2]).all()
        output[:2].copy_(expected)

    layer = types.SimpleNamespace(
        prefix="layer",
        enable_packed_recurrent_decode=True,
        _forward_core_decode_non_spec=packed_decode,
    )
    output = torch.full((2 + padding, 2, 4), float("nan"))
    qkv, gates = torch.empty(2 + padding, 16), torch.empty(2 + padding, 2)
    QwenGatedDeltaNetAttention._forward_core(layer, qkv, gates, gates, output)
    torch.testing.assert_close(output[:2], expected)
    torch.testing.assert_close(output[2:], torch.zeros_like(output[2:]))


@pytest.mark.parametrize("platform", ["mcpu", "cuda"])
@pytest.mark.parametrize("tokens", [1, 5])
def test_rearrange_qkv_preserves_values_and_backend_layout(
    monkeypatch, platform, tokens
):
    """XCPU consumes packed views; other backends retain contiguous Q/K/V."""
    monkeypatch.setattr(
        gdn, "current_platform", types.SimpleNamespace(device_name=platform)
    )
    layer = types.SimpleNamespace(
        tp_size=2, key_dim=16, value_dim=32, head_k_dim=4, head_v_dim=4
    )
    storage = torch.arange(tokens * 48.0).reshape(tokens, 48)
    packed = storage[:, :32]
    outputs = QwenGatedDeltaNetAttention.rearrange_mixed_qkv(layer, packed)
    for actual, expected in zip(outputs, packed.split([8, 8, 16], dim=-1)):
        torch.testing.assert_close(actual, expected.view(1, tokens, -1, 4))
        if platform == "mcpu":
            assert actual.untyped_storage().data_ptr() == storage.data_ptr()
            if tokens > 1:
                assert actual.stride(-3) == storage.stride(0)
        else:
            assert actual.is_contiguous()
