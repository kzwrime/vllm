# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""直接调用 MLA 时，注意力必须使用 connector 恢复的历史缓存。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.attention import (
    attention,
    kv_transfer_utils,
    mla_attention,
)


@pytest.mark.parametrize("with_connector", [False, True])
def test_direct_mla_uses_transferred_cache(monkeypatch, with_connector):
    cache = torch.zeros(2)
    events = []

    def update(*args):
        cache[1] = 3

    def restore(name):
        events.append("load")
        cache[0] = 9

    def forward(*args, output, **kwargs):
        events.append("attention")
        output.fill_(cache.sum())

    def save(name, value, metadata):
        events.append("save")
        assert torch.equal(value, torch.tensor([9.0, 3.0]))

    layer = SimpleNamespace(
        calculate_kv_scales=False,
        use_direct_call=True,
        use_pcp=False,
        layer_name="layer",
        kv_cache=cache,
        kv_cache_dtype="auto",
        _k_scale=torch.tensor(1.0),
        impl=SimpleNamespace(do_kv_cache_update=update),
        forward_impl=forward,
    )
    context = SimpleNamespace(
        attn_metadata={"layer": SimpleNamespace(num_decode_tokens=1)},
        no_compile_layers={"layer": layer},
        slot_mapping={"layer": torch.tensor([1])},
    )
    monkeypatch.setattr(mla_attention, "get_forward_context", lambda: context)
    monkeypatch.setattr(attention, "get_forward_context", lambda: context)
    monkeypatch.setattr(mla_attention, "_encode_layer_name", lambda name: name)
    monkeypatch.setattr(
        kv_transfer_utils, "has_kv_transfer_group", lambda: with_connector
    )
    monkeypatch.setattr(kv_transfer_utils, "is_v1_kv_transfer_group", lambda: True)
    connector = Mock()
    connector.has_connector_metadata.return_value = True
    connector.wait_for_layer_load.side_effect = restore
    connector.save_kv_layer.side_effect = save
    monkeypatch.setattr(kv_transfer_utils, "get_kv_transfer_group", lambda: connector)
    q = torch.zeros(1)
    result = mla_attention.MLAAttention.forward(layer, q, q, q, output_shape=q.shape)
    assert result.item() == (12 if with_connector else 3)
    assert events == (
        ["load", "attention", "save"] if with_connector else ["attention"]
    )
