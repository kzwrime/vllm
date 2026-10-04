# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronous file transfer example for MLA and indexer caches.

Requires replicated PCP caches and a dedicated directory for one writer,
model and cache format. Cache pages may have padding. Do not delete published
files while readers are active. Not intended for production serving.
"""

import json
import os
from pathlib import Path

import safetensors.torch
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
    align_to_block_size,
)


class ExampleMLAConnector(ExampleConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        p = vllm_config.parallel_config
        if (
            any(
                size != 1
                for size in (
                    p.tensor_parallel_size,
                    p.pipeline_parallel_size,
                    p.data_parallel_size,
                    p.decode_context_parallel_size,
                )
            )
            or p.enable_expert_parallel
            or vllm_config.scheduler_config.max_num_seqs != 1
            or vllm_config.cache_config.cache_dtype != "auto"
            or len(kv_cache_config.kv_cache_groups) != 1
        ):
            raise ValueError(
                "ExampleMLAConnector requires TP=PP=DP=DCP=1, no EP, "
                "one request, regular KV cache and one cache group"
            )
        self._layers = set(kv_cache_config.kv_cache_groups[0].layer_names)
        self._caches: dict[str, torch.Tensor] = {}
        self._saved: dict[str, set[str]] = {}
        self._writer = True
        if role == KVConnectorRole.WORKER:
            from vllm.distributed import get_pcp_group

            self._writer = get_pcp_group().rank_in_group == 0

    def register_kv_caches(self, kv_caches):
        if set(kv_caches) != self._layers:
            raise ValueError("Registered caches must match the complete layer set")
        for name, cache in kv_caches.items():
            if cache.ndim != 3 or cache.shape[1] != self._block_size:
                raise ValueError(f"Unsupported cache layout: {name}: {cache.shape}")
        self._caches = kv_caches

    def bind_connector_metadata(self, connector_metadata):
        super().bind_connector_metadata(connector_metadata)
        self._saved.clear()

    def _manifest(self, tokens):
        return {
            "version": 1,
            "block_size": self._block_size,
            "tokens": tokens,
            "layers": sorted(self._layers),
        }

    def _found_match_for_prompt(self, prompt_token_ids, mm_hashes):
        n = align_to_block_size(len(prompt_token_ids) - 1, self._block_size)
        if n <= 0:
            return False
        folder = Path(
            self._generate_foldername_debug(
                torch.tensor(prompt_token_ids[:n]), mm_hashes
            )
        )
        manifest = folder / "complete.json"
        if not manifest.is_file():
            return False
        return json.loads(manifest.read_text()) == self._manifest(n) and all(
            (folder / f"{name}.safetensors").is_file() for name in self._layers
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if self._kv_transfer_config.kv_role == "kv_producer":
            return 0, False
        count, async_load = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        return max(0, count), async_load

    def build_connector_meta(self, scheduler_output):
        for request in scheduler_output.scheduled_new_reqs:
            scheduled = scheduler_output.num_scheduled_tokens[request.req_id]
            if request.num_computed_tokens + scheduled < len(
                request.prompt_token_ids or []
            ):
                raise ValueError("ExampleMLAConnector does not support chunked prefill")
        if scheduler_output.scheduled_cached_reqs.resumed_req_ids:
            raise ValueError(
                "ExampleMLAConnector does not support resuming preempted requests"
            )
        meta = super().build_connector_meta(scheduler_output)
        if self._kv_transfer_config.kv_role == "kv_consumer":
            meta.requests = [r for r in meta.requests if not r.is_store]
        return meta

    def _blocks(self, request, cache):
        slots = request.slot_mapping.cpu().long()
        n = len(request.token_ids)
        if n == 0 or len(slots) != n or n % self._block_size:
            raise ValueError("Cache transfer requires complete, nonempty blocks")
        rows = slots.reshape(-1, self._block_size)
        starts = rows[:, 0]
        if (
            torch.any(starts < 0)
            or torch.any(starts % self._block_size)
            or not torch.equal(rows, starts[:, None] + torch.arange(self._block_size))
            or torch.any(starts // self._block_size >= cache.shape[0])
        ):
            raise ValueError("Cache slots are misaligned or out of bounds")
        return (starts // self._block_size).tolist()

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if not self._writer:
            return
        if layer_name not in self._caches or kv_layer is not self._caches[layer_name]:
            raise ValueError(
                f"Saving requires the registered cache tensor: {layer_name}"
            )
        for request in self._get_connector_metadata().requests:
            if not request.is_store or len(request.token_ids) == 0:
                continue
            filename = Path(
                self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes
                )
            )
            if (filename.parent / "complete.json").exists():
                continue
            blocks = self._blocks(request, kv_layer)
            cpu = torch.cat([kv_layer[b].detach().cpu() for b in blocks]).contiguous()
            temporary = filename.with_suffix(f".{os.getpid()}.tmp")
            safetensors.torch.save_file({"kv_cache": cpu}, str(temporary))
            temporary.replace(filename)
            saved = self._saved.setdefault(str(filename.parent), set())
            saved.add(layer_name)
            if saved == self._layers:
                temporary = filename.parent / f"complete.{os.getpid()}.tmp"
                temporary.write_text(json.dumps(self._manifest(len(request.token_ids))))
                temporary.replace(filename.parent / "complete.json")
                del self._saved[str(filename.parent)]

    def start_load_kv(self, forward_context, **kwargs):
        for request in self._get_connector_metadata().requests:
            if request.is_store:
                continue
            if set(self._caches) != self._layers:
                raise ValueError("All caches must be registered before loading")
            for name, cache in self._caches.items():
                folder = Path(
                    self._generate_foldername_debug(
                        request.token_ids, request.mm_hashes
                    )
                )
                manifest = json.loads((folder / "complete.json").read_text())
                if manifest != self._manifest(len(request.token_ids)):
                    raise ValueError("Cache manifest is incompatible with this engine")
                cpu = safetensors.torch.load_file(
                    str(folder / f"{name}.safetensors"), device="cpu"
                )["kv_cache"]
                if (
                    cpu.shape != (len(request.token_ids), cache.shape[2])
                    or cpu.dtype != cache.dtype
                ):
                    raise ValueError(f"Cache format mismatch: {name}")
                blocks = self._blocks(request, cache)
                # Copy per page: reshape may copy a padded cache.
                for i, block in enumerate(blocks):
                    cache[block].copy_(
                        cpu[i * self._block_size : (i + 1) * self._block_size].to(
                            cache.device
                        )
                    )
