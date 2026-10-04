# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.example_mla_connector import (
    ExampleMLAConnector,
)


def make_connector(tmp_path, monkeypatch, rank=0):
    monkeypatch.setattr(
        "vllm.distributed.get_pcp_group", lambda: NS(rank_in_group=rank)
    )
    config = NS(
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleMLAConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
        ),
        cache_config=NS(block_size=4, cache_dtype="auto"),
        scheduler_config=NS(max_num_seqs=1),
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
        ),
    )
    caches = NS(kv_cache_groups=[NS(layer_names=["mla", "indexer"])])
    return ExampleMLAConnector(config, KVConnectorRole.WORKER, caches)


def metadata(store, block):
    result = ExampleConnectorMetadata()
    result.add_request(list(range(6)), [block, block + 1], 4, store, [])
    return result


def cache(dtype):
    result = torch.empty_strided((4, 4, 3), (16, 3, 1), dtype=dtype)
    result.copy_(torch.arange(48).reshape(4, 4, 3))
    return result


def test_complete_cache_restored_to_other_slots(tmp_path, monkeypatch):
    writer = make_connector(tmp_path, monkeypatch)
    source = {"mla": cache(torch.bfloat16), "indexer": cache(torch.uint8)}
    writer.register_kv_caches(source)
    writer.bind_connector_metadata(metadata(True, 1))
    writer.save_kv_layer("mla", source["mla"], None)
    assert not writer._found_match_for_prompt(list(range(7)), [])
    writer.save_kv_layer("indexer", source["indexer"], None)
    assert writer._found_match_for_prompt(list(range(7)), [])

    reader = make_connector(tmp_path, monkeypatch)
    target = {k: cache(v.dtype).zero_() for k, v in source.items()}
    reader.register_kv_caches(target)
    reader.bind_connector_metadata(metadata(False, 2))
    reader.start_load_kv(NS(attn_metadata={}))
    for name in target:
        assert torch.equal(target[name][2], source[name][1])
        assert not target[name][:2].any()
        assert not target[name][3].any()
    next(tmp_path.glob("*/indexer.safetensors")).unlink()
    assert not reader._found_match_for_prompt(list(range(7)), [])


def test_secondary_pcp_rank_does_not_write(tmp_path, monkeypatch):
    connector = make_connector(tmp_path, monkeypatch, rank=1)
    connector.bind_connector_metadata(metadata(True, 1))
    connector.save_kv_layer("mla", cache(torch.bfloat16), None)
    assert not list(tmp_path.iterdir())


def test_rejects_partial_prefill_before_publishing(tmp_path, monkeypatch):
    connector = make_connector(tmp_path, monkeypatch)
    schedule = NS(
        scheduled_new_reqs=[
            NS(req_id="r", num_computed_tokens=0, prompt_token_ids=list(range(6)))
        ],
        num_scheduled_tokens={"r": 3},
    )
    with pytest.raises(ValueError, match="chunked prefill"):
        connector.build_connector_meta(schedule)
    assert not list(tmp_path.iterdir())


def test_rejects_incompatible_cache_format(tmp_path, monkeypatch):
    connector = make_connector(tmp_path, monkeypatch)
    caches = {"mla": cache(torch.bfloat16), "indexer": cache(torch.uint8)}
    connector.register_kv_caches(caches)
    connector.bind_connector_metadata(metadata(True, 1))
    for name, value in caches.items():
        connector.save_kv_layer(name, value, None)
    caches["mla"] = caches["mla"].float()
    connector.register_kv_caches(caches)
    connector.bind_connector_metadata(metadata(False, 2))
    with pytest.raises(ValueError, match="Cache format mismatch"):
        connector.start_load_kv(NS())


def test_incomplete_previous_step_cannot_publish_new_cache(tmp_path, monkeypatch):
    connector = make_connector(tmp_path, monkeypatch)
    caches = {"mla": cache(torch.bfloat16), "indexer": cache(torch.uint8)}
    connector.register_kv_caches(caches)
    connector.bind_connector_metadata(metadata(True, 1))
    connector.save_kv_layer("mla", caches["mla"], None)
    connector.bind_connector_metadata(metadata(True, 1))
    connector.save_kv_layer("indexer", caches["indexer"], None)
    assert not connector._found_match_for_prompt(list(range(7)), [])
    connector.save_kv_layer("mla", caches["mla"], None)
    assert connector._found_match_for_prompt(list(range(7)), [])
