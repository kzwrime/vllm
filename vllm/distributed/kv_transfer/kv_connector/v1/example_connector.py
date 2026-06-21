# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.utils.hashing import safe_hash
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool
    mm_hashes: list[str]

    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> "ReqMeta":
        # A scheduler step can allocate fewer blocks than the full prompt
        # length, for example with chunked prefill. Keep the cache key and
        # slot mapping limited to KV that exists for this step.
        valid_num_tokens = min(
            align_to_block_size(len(token_ids), block_size),
            len(block_ids) * block_size,
        )
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
        )


@dataclass
class ExampleConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store, mm_hashes)
        )


class ExampleConnector(KVConnectorBase_V1, SupportsHMA):
    # NOTE: This is Simple debug implementation of the KV connector.
    # It save / load the KV cache to / from the disk.
    # It does extra work which will overwrite the existing prefix-cache in GPU
    # - to remove the overhead, need to add some "mask" in the ReqMeta class

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, tuple[Request, int]] = {}
        self._requests_need_store: dict[
            str, tuple[list[int], list[str], list[int]]
        ] = {}
        self._matched_num_tokens: dict[str, int] = {}
        self._pending_metadata: dict[str, tuple[torch.Tensor, list[str]]] = {}
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """

        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            attn_metadata: AttentionMetadata,
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache
                    layer. In shape [2, num_pages, page_size, xxx] if not
                    using MLA, [num_pages, page_size, xxx] otherwise.
                src_kv_cache (torch.Tensor): the source KV cache. In shape
                    [2, num_tokens, xxx] if not using MLA, [num_tokens, xxx]
                    otherwise.
                slot_mapping (torch.Tensor): the slot mapping. In shape
                    [num_tokens].
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1
                )
                dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
            elif _is_triton_block_kv_layout(attn_metadata):
                block_idxs = slot_mapping // self._block_size
                offsets = slot_mapping % self._block_size
                dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1
                )
                dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ExampleConnectorMetadata)

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("In connector.start_load_kv, but the attn_metadata is None")
            return

        # Load the KV for each request each layer
        for request in metadata.requests:
            if request.is_store:
                continue
            logger.info(
                "Inject KV cache of %d tokens to the paged memory",
                len(request.slot_mapping),
            )
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache
                # attribute (attention layers) Skip non-attention
                # layers like FusedMoE/MLP etc.
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue

                filename = self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes
                )
                kv_cache = safetensors.torch.load_file(filename)["kv_cache"].to(
                    kv_cache_layer.device
                )
                if isinstance(attn_metadata, dict):
                    inject_kv_into_layer(
                        kv_cache_layer,
                        kv_cache,
                        request.slot_mapping,
                        attn_metadata[layer_name],
                    )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx)
            if MLA is not used, and (num_pages, page_size, xxx) otherwise.
            """
            if isinstance(attn_metadata, MLACommonMetadata):
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
            elif _is_triton_block_kv_layout(attn_metadata):
                block_idxs = slot_mapping // self._block_size
                offsets = slot_mapping % self._block_size
                return layer[block_idxs, :, offsets]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleConnectorMetadata)
        for request in connector_metadata.requests:
            if request.is_store:
                filename = self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes
                )
                kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
                tensors = {"kv_cache": kv_cache.detach().cpu()}
                safetensors.torch.save_file(tensors, filename)

    def wait_for_save(self):
        for foldername, (token_ids, mm_hashes) in self._pending_metadata.items():
            self._write_metadata_debug(foldername, token_ids, mm_hashes)
        self._pending_metadata.clear()
        return

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._requests_need_store.pop(request.request_id, None)
        self._requests_need_load.pop(request.request_id, None)
        self._matched_num_tokens.pop(request.request_id, None)
        return False, None

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # NOTE: in this debug implementation, we assume that the prompt is
        # cached_prompt + newly_generated_single_token
        # Therefore, we use prompt_token_ids[:-1] to determine the folder name

        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned
        # with the block granularity. And it expects the returned blocks and
        # num_computed_tokens to also be aligned with the block granularity.
        num_matched_tokens = self._get_num_matched_tokens_for_request(request)
        if num_matched_tokens <= num_computed_tokens:
            return 0, False

        logger.info("External Cache Hit!")

        # Now, first num_tokens_to_check tokens are hit, we need to prepare
        # the metadata for the worker connector to correctly load the KV
        self._matched_num_tokens[request.request_id] = num_matched_tokens

        return num_matched_tokens - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        if num_external_tokens > 0:
            num_matched_tokens = self._matched_num_tokens.pop(
                request.request_id, num_external_tokens
            )
            self._requests_need_load[request.request_id] = (
                request,
                num_matched_tokens,
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = ExampleConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            if new_req.req_id in self._requests_need_load:
                request, num_matched_tokens = self._requests_need_load[new_req.req_id]
                meta.add_request(
                    token_ids=list(request.prompt_token_ids or [])[:num_matched_tokens],
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )
                total_need_load += 1
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but a single request can have both store and load.
                # NOTE(rob): for this debug implementation, we only cache
                # the original prompt tokens.
                if not self._found_match_for_prompt(token_ids, mm_hashes):
                    num_cache_tokens = align_to_block_size(
                        len(token_ids), self._block_size
                    )
                    if num_cache_tokens > 0:
                        block_ids = list(new_req.block_ids[0])
                        num_available_tokens = (
                            new_req.num_computed_tokens
                            + scheduler_output.num_scheduled_tokens[new_req.req_id]
                        )
                        if num_available_tokens >= num_cache_tokens:
                            meta.add_request(
                                token_ids=token_ids[:num_cache_tokens],
                                block_ids=block_ids,
                                block_size=self._block_size,
                                is_store=True,
                                mm_hashes=mm_hashes,
                            )
                        else:
                            self._requests_need_store[new_req.req_id] = (
                                list(token_ids),
                                mm_hashes,
                                block_ids,
                            )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids
            new_block_ids = cached_reqs.new_block_ids[i]
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]

            if resumed_from_preemption and req_id in self._requests_need_load:
                # NOTE(rob): cached_req_data does not have the full
                # list of token ids (only new tokens). So we look it
                # up in the actual request object.
                request, num_matched_tokens = self._requests_need_load[req_id]
                total_tokens = num_computed_tokens + num_new_tokens

                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                assert new_block_ids is not None
                block_ids = new_block_ids[0]

                meta.add_request(
                    token_ids=token_ids[:num_matched_tokens],
                    block_ids=block_ids,
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=[f.identifier for f in request.mm_features],
                )
                total_need_load += 1

            if req_id not in self._requests_need_store:
                continue

            token_ids, mm_hashes, block_ids = self._requests_need_store[req_id]
            if new_block_ids is not None:
                if resumed_from_preemption:
                    block_ids = list(new_block_ids[0])
                else:
                    block_ids.extend(new_block_ids[0])
            num_cache_tokens = align_to_block_size(len(token_ids), self._block_size)
            num_available_tokens = num_computed_tokens + num_new_tokens
            if num_available_tokens >= num_cache_tokens:
                meta.add_request(
                    token_ids=token_ids[:num_cache_tokens],
                    block_ids=block_ids,
                    block_size=self._block_size,
                    is_store=True,
                    mm_hashes=mm_hashes,
                )
                self._requests_need_store.pop(req_id)
            else:
                self._requests_need_store[req_id] = (token_ids, mm_hashes, block_ids)

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_request(
        self,
        request: "Request",
    ) -> bool:
        """Check if the cache is hit for the request."""
        return self._get_num_matched_tokens_for_request(request) > 0

    def _get_num_matched_tokens_for_request(
        self,
        request: "Request",
    ) -> int:
        return self._get_num_matched_tokens_for_prompt(
            list(request.prompt_token_ids or []),
            [f.identifier for f in request.mm_features],
        )

    def _found_match_for_prompt(
        self,
        prompt_token_ids: list[int],
        mm_hashes: list[str],
    ) -> bool:
        return self._get_num_matched_tokens_for_prompt(prompt_token_ids, mm_hashes) > 0

    def _get_num_matched_tokens_for_prompt(
        self,
        prompt_token_ids: list[int],
        mm_hashes: list[str],
    ) -> int:
        max_num_tokens = align_to_block_size(len(prompt_token_ids), self._block_size)
        if max_num_tokens <= 0:
            return 0

        # Fast path for the common case where the decode prompt is exactly
        # cached_prompt + one generated token.
        num_tokens_to_check = align_to_block_size(
            len(prompt_token_ids) - 1, self._block_size
        )
        foldername = self._generate_foldername_debug(
            torch.tensor(prompt_token_ids)[:num_tokens_to_check],
            mm_hashes,
            create_folder=False,
        )
        if num_tokens_to_check > 0 and self._is_complete_folder_debug(foldername):
            return num_tokens_to_check

        # Chat templates can tokenize the prefilled text into more than one
        # token. Fall back to the longest cached prefix recorded on disk.
        best = 0
        if not os.path.isdir(self._storage_path):
            return best
        for entry in os.scandir(self._storage_path):
            if not entry.is_dir():
                continue
            metadata_file = os.path.join(entry.path, "metadata.json")
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    metadata = json.load(f)
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                continue
            if metadata.get("mm_hashes", []) != mm_hashes:
                continue
            token_ids = metadata.get("token_ids")
            if not isinstance(token_ids, list):
                continue
            num_tokens = len(token_ids)
            if (
                num_tokens > best
                and num_tokens <= max_num_tokens
                and num_tokens % self._block_size == 0
                and prompt_token_ids[:num_tokens] == token_ids
            ):
                best = num_tokens
        return best

    def _is_complete_folder_debug(self, foldername: str) -> bool:
        return os.path.exists(os.path.join(foldername, "metadata.json"))

    def _generate_foldername_debug(
        self,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
        create_folder=False,
    ) -> str:
        """Generate a folder name based on the hash of the bytes of the input
        ids.
        """
        token_bytes = token_ids.numpy().tobytes()
        # Add mm_hashes to the bytes being hashed to avoid path traversal and
        # to create a canonical key.
        if mm_hashes:
            mm_str = "-".join(mm_hashes)
            token_bytes += mm_str.encode("utf-8")
        input_ids_hash = safe_hash(token_bytes, usedforsecurity=False).hexdigest()

        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
            self._pending_metadata[foldername] = (token_ids.clone(), list(mm_hashes))
        return foldername

    def _write_metadata_debug(
        self,
        foldername: str,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
    ) -> None:
        metadata_file = os.path.join(foldername, "metadata.json")
        if os.path.exists(metadata_file):
            return
        metadata = {
            "token_ids": token_ids.tolist(),
            "mm_hashes": mm_hashes,
        }
        tmp_file = f"{metadata_file}.{os.getpid()}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        os.replace(tmp_file, metadata_file)

    def _generate_filename_debug(
        self,
        layer_name: str,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
    ) -> str:
        """Generate a file name based on the layer name and the hash
        of the bytes of the input ids.
        """
        foldername = self._generate_foldername_debug(
            token_ids, mm_hashes=mm_hashes, create_folder=True
        )
        return os.path.join(foldername, f"{layer_name}.safetensors")


def align_to_block_size(num_tokens: int, block_size) -> int:
    """Align the number of tokens to the block size."""
    return num_tokens // block_size * block_size


def _is_triton_block_kv_layout(attn_metadata: AttentionMetadata) -> bool:
    return (
        isinstance(attn_metadata, TritonAttentionMetadata)
        and getattr(attn_metadata, "kv_cache_tensor_layout", "BLOCK_KV") == "BLOCK_KV"
    )
