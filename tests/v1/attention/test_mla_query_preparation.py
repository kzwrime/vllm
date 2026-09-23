# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch_xcpu

from vllm.model_executor.layers.attention.mla_attention import MLAAttention


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize(
    "shape_tokens,compiled", [(False, False), (True, False), (False, True)]
)
def test_mla_crops_padding_and_uses_explicit_projection_state(
    monkeypatch, projected, shape_tokens, compiled
):
    # Same raw/final width must not select a different computation. Exercise
    # the actual forwarding method with a CPU reference projection/backend.
    q = torch.ones(4, 1, 6)
    q[3].fill_(100)
    seen = []

    def mqa(query, cache, metadata, layer):
        nope = query[0] if isinstance(query, tuple) else query[..., :4]
        seen.append(nope.clone())
        return nope, None

    def project(q, w, out):
        out.copy_(torch.einsum("thk,hkn->thn", q, w))

    monkeypatch.setattr(torch_xcpu.ops, "einsum_mhk_hkn_to_mhn", project)
    layer = SimpleNamespace(
        num_heads=1,
        v_head_dim=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        kv_cache_dtype="auto",
        use_pcp=False,
        prefill_backend=None,
        _use_shape_token_count=shape_tokens,
        q_pad_num_heads=None,
        is_aiter_triton_fp4_bmm_enabled=False,
        is_aiter_triton_fp8_bmm_enabled=False,
        W_UK_T=2 * torch.eye(4).unsqueeze(0),
        impl=SimpleNamespace(dcp_world_size=1, is_sparse=True, forward_mqa=mqa),
        _v_up_proj=lambda values, out: out.copy_(values.flatten(1)),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=3, num_decodes=3, num_prefills=0, num_decode_tokens=3
    )
    output = torch.full((4, 4), -99.0)

    def forward(q, output):
        return MLAAttention.forward_impl(
            layer,
            q,
            torch.zeros(4, 4),
            torch.zeros(4, 1, 2),
            torch.empty(0),
            metadata,
            output,
            q_is_projected=projected,
        )

    if compiled:
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    result = forward(q, output)
    expected = 1 if projected else 2
    torch.testing.assert_close(seen[0], torch.full((3, 1, 4), float(expected)))
    torch.testing.assert_close(result[:3], torch.full((3, 4), float(expected)))
    assert torch.equal(result[3], torch.full((4,), -99.0))
