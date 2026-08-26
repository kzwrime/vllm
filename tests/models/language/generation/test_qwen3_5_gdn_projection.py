# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model


@pytest.mark.parametrize(
    "checkpoint_name,expected_name,expected_shard",
    [
        ("layer.in_proj_qkv.weight", "layer.in_proj_qkvzba.weight", (0, 1, 2)),
        ("layer.in_proj_z.weight", "layer.in_proj_qkvzba.weight", 3),
        ("layer.in_proj_b.weight", "layer.in_proj_qkvzba.weight", 4),
        ("layer.in_proj_a.weight", "layer.in_proj_qkvzba.weight", 5),
    ],
)
def test_fused_gdn_mapper_preserves_projection_order(
    checkpoint_name: str,
    expected_name: str,
    expected_shard: tuple[int, ...] | int,
) -> None:
    weight = torch.empty(0)
    [(mapped_name, mapped_weight)] = Qwen3_5Model.fused_gdn_hf_to_vllm_mapper.apply(
        [(checkpoint_name, weight)]
    )

    assert mapped_name == expected_name
    assert mapped_weight is weight
    assert mapped_weight.shard_id == expected_shard


def test_fused_gdn_projection_splits_qkvz_and_ba() -> None:
    projected = torch.arange(32).reshape(2, 16)
    layer = SimpleNamespace(
        use_fused_in_proj_qkvzba=True,
        in_proj_qkvzba=lambda hidden_states: (projected, None),
        key_dim=2,
        value_dim=4,
        num_v_heads=2,
        tp_size=1,
    )

    qkvz, ba = QwenGatedDeltaNetAttention._project_qkvz_ba(layer, torch.empty(2, 1))

    torch.testing.assert_close(qkvz, projected[:, :12])
    torch.testing.assert_close(ba, projected[:, 12:])
    assert not qkvz.is_contiguous()
    assert not ba.is_contiguous()
    assert qkvz.stride() == (16, 1)
    assert ba.stride() == (16, 1)
    assert qkvz.untyped_storage().data_ptr() == projected.untyped_storage().data_ptr()
    assert ba.untyped_storage().data_ptr() == projected.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    "override",
    [
        {"gqa_interleaved_layout": True},
        {"quant_config": object()},
        {"lora_config": object()},
        {"dtype": torch.float16},
        {"model_type": "qwen3_next"},
    ],
)
def test_fused_gdn_projection_rejects_unsupported_configs(
    monkeypatch: pytest.MonkeyPatch, override: dict[str, object]
) -> None:
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "current_platform",
        SimpleNamespace(device_name="mcpu"),
    )
    monkeypatch.setattr(
        qwen_gdn_linear_attn.envs,
        "VLLM_XCPU_FUSE_GDN_IN_PROJ_QKVZBA",
        True,
    )
    layer = SimpleNamespace(gqa_interleaved_layout=False, quant_config=None)
    config = SimpleNamespace(model_type="qwen3_5_text")
    vllm_config = SimpleNamespace(
        lora_config=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )

    for name, value in override.items():
        if name == "model_type":
            setattr(config, name, value)
        elif name in ("lora_config", "dtype"):
            target = vllm_config if name == "lora_config" else vllm_config.model_config
            setattr(target, name, value)
        else:
            setattr(layer, name, value)

    assert not QwenGatedDeltaNetAttention._should_fuse_in_proj_qkvzba(
        layer, config, vllm_config
    )


def test_fused_gdn_projection_accepts_supported_xcpu_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "current_platform",
        SimpleNamespace(device_name="mcpu"),
    )
    monkeypatch.setattr(
        qwen_gdn_linear_attn.envs,
        "VLLM_XCPU_FUSE_GDN_IN_PROJ_QKVZBA",
        True,
    )
    layer = SimpleNamespace(gqa_interleaved_layout=False, quant_config=None)
    config = SimpleNamespace(model_type="qwen3_5_text")
    vllm_config = SimpleNamespace(
        lora_config=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )

    assert QwenGatedDeltaNetAttention._should_fuse_in_proj_qkvzba(
        layer, config, vllm_config
    )
