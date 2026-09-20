# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models import deepseek_eagle as eagle_mod
from vllm.model_executor.models import deepseek_v2 as deepseek_mod


class _PPGroup:
    is_first_rank = True
    is_last_rank = True


class _Embedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], self.hidden_size), dtype=torch.float32)


class _Norm(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _Projection(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, : self.hidden_size]


class _DecoderLayer(nn.Module):
    def __init__(self, use_sequence_parallel_moe: bool):
        super().__init__()
        self.use_sequence_parallel_moe = use_sequence_parallel_moe
        self.input_layouts: list[bool] = []

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
        input_is_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.input_layouts.append(input_is_sequence_parallel)
        if residual is None:
            residual = hidden_states
        return hidden_states, residual


def _patch_collectives(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    gathered: list[torch.Tensor] = []

    def all_gather(hidden_states: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0
        gathered.append(hidden_states)
        return torch.cat([hidden_states, hidden_states], dim=0)

    monkeypatch.setattr(deepseek_mod, "tensor_model_parallel_all_gather", all_gather)
    monkeypatch.setattr(deepseek_mod, "get_pp_group", lambda: _PPGroup())
    return gathered


def _make_target_model(
    layers: list[_DecoderLayer], aux_layers: tuple[int, ...] = ()
) -> deepseek_mod.DeepseekV2Model:
    model = deepseek_mod.DeepseekV2Model.__new__(deepseek_mod.DeepseekV2Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace()
    model.hidden_size = 4
    model.embed_tokens = _Embedding(model.hidden_size)
    model.layers = nn.ModuleList(layers)
    model.start_layer = 0
    model.end_layer = len(layers)
    model.aux_hidden_state_layers = aux_layers
    model.norm = _Norm()
    return model


@pytest.mark.cpu_test
def test_target_tracks_ambiguous_single_token_sp_layout(monkeypatch):
    gathered = _patch_collectives(monkeypatch)
    layers = [_DecoderLayer(False), _DecoderLayer(True), _DecoderLayer(True)]
    model = _make_target_model(layers, aux_layers=(0, 2))

    output, aux_hidden_states = model.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        intermediate_tensors=None,
    )

    # The full tensor and a TP2 padded shard both have one row.  The second
    # MoE layer must still be told that its input is sequence parallel.
    assert [layer.input_layouts for layer in layers] == [[False], [False], [True]]
    # One gather restores the requested target aux state and one restores the
    # final target hidden state consumed by DFlash2/DSpark.
    assert len(gathered) == 2
    assert output.shape == (1, 4)
    assert len(aux_hidden_states) == 2
    assert all(hidden_states.shape == (1, 4) for hidden_states in aux_hidden_states)


@pytest.mark.cpu_test
def test_target_restores_sp_before_dense_layer(monkeypatch):
    gathered = _patch_collectives(monkeypatch)
    layers = [_DecoderLayer(True), _DecoderLayer(False)]
    model = _make_target_model(layers)

    output = model.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        intermediate_tensors=None,
    )

    assert [layer.input_layouts for layer in layers] == [[False], [False]]
    assert len(gathered) == 1
    assert output.shape == (1, 4)


@pytest.mark.cpu_test
def test_deepseek_eagle_restores_single_token_sp_output(monkeypatch):
    gathered = _patch_collectives(monkeypatch)
    layers = [_DecoderLayer(True), _DecoderLayer(True)]
    model = eagle_mod.DeepseekV2Model.__new__(eagle_mod.DeepseekV2Model)
    nn.Module.__init__(model)
    model.embed_tokens = _Embedding(4)
    model.enorm = _Norm()
    model.hnorm = _Norm()
    model.fc = _Projection(4)
    model.layers = nn.ModuleList(layers)
    model.norm = _Norm()

    logits_hidden, recycled_hidden = model.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        hidden_states=torch.zeros((1, 4)),
    )

    assert [layer.input_layouts for layer in layers] == [[False], [True]]
    assert len(gathered) == 1
    assert logits_hidden.shape == (1, 4)
    assert recycled_hidden.shape == (1, 4)
