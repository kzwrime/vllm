# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
import time
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.parallel_state import prepare_communication_buffer_for_model
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model, get_model_loader
from vllm.model_executor.models.interfaces import is_mixture_of_experts, supports_eagle3, supports_multimodal_pruning
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class CPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device == torch.device("cpu")
        assert self.speculative_config is None, "spec decode is not supported."

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tensors()

    def _postprocess_tensors(self) -> None:
        # Note: replace device tensors with cpu tensors
        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if cpu_tensor is not None and device_tensor is not None:
                assert isinstance(cpu_tensor, torch.Tensor)
                assert isinstance(device_tensor, torch.Tensor)
                setattr(obj, device_attr_name, cpu_tensor)

        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                v.gpu = v.cpu

        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        for block_table in self.input_batch.block_table.block_tables:
            for v in vars(block_table).values():
                if isinstance(v, CpuGpuBuffer):
                    v.gpu = v.cpu

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )

        global_expert_loads, old_global_expert_indices_per_model, rank_mapping = (
            EplbState.get_eep_state(self.parallel_config)
            if eep_scale_up
            else (None, None, None)
        )

        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
            eplb_models = 0


        # with DeviceMemoryProfiler() as m:
        time_before_load = time.perf_counter()
        model_loader = get_model_loader(self.load_config)
        self.model = model_loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )
        if self.lora_config:
            self.model = self.load_lora_model(
                self.model, self.vllm_config, self.device
            )
        if hasattr(self, "drafter"):
            logger.info_once("Loading drafter model...")
            self.drafter.load_model(self.model)
            if (
                hasattr(self.drafter, "model")
                and is_mixture_of_experts(self.drafter.model)
                and self.parallel_config.enable_eplb
            ):
                spec_config = self.vllm_config.speculative_config
                assert spec_config is not None
                assert spec_config.draft_model_config is not None
                logger.info_once(
                    "EPLB is enabled for drafter model %s.",
                    spec_config.draft_model_config.model,
                )

                global_expert_load = (
                    global_expert_loads[eplb_models]
                    if global_expert_loads
                    else None
                )
                old_global_expert_indices = (
                    old_global_expert_indices_per_model[eplb_models]
                    if old_global_expert_indices_per_model
                    else None
                )
                if self.eplb_state is None:
                    self.eplb_state = EplbState(
                        self.parallel_config, self.device
                    )
                self.eplb_state.add_model(
                    self.drafter.model,
                    spec_config.draft_model_config,
                    global_expert_load,
                    old_global_expert_indices,
                    rank_mapping,
                )
                eplb_models += 1

        if self.use_aux_hidden_state_outputs:
            if not supports_eagle3(self.get_model()):
                raise RuntimeError(
                    "Model does not support EAGLE3 interface but "
                    "aux_hidden_state_outputs was requested"
                )

            # Try to get auxiliary layers from speculative config,
            # otherwise use model's default layers
            aux_layers = self._get_eagle3_aux_layers_from_config()
            if aux_layers:
                logger.info(
                    "Using auxiliary layers from speculative config: %s",
                    aux_layers,
                )
            else:
                aux_layers = self.model.get_eagle3_aux_hidden_state_layers()
    
            self.model.set_aux_hidden_state_layers(aux_layers)
        time_after_load = time.perf_counter()
        # self.model_memory_usage = m.consumed_memory
        logger.info_once(
            "Model loading took %.6f seconds",
            time_after_load - time_before_load,
            scope="local",
        )
        prepare_communication_buffer_for_model(self.model)
        if (drafter := getattr(self, "drafter", None)) and (
            drafter_model := getattr(drafter, "model", None)
        ):
            prepare_communication_buffer_for_model(drafter_model)
        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        )

        if is_mixture_of_experts(self.model) and self.parallel_config.enable_eplb:
            logger.info_once("EPLB is enabled for model %s.", self.model_config.model)
            global_expert_load = (
                global_expert_loads[eplb_models] if global_expert_loads else None
            )
            old_global_expert_indices = (
                old_global_expert_indices_per_model[eplb_models]
                if old_global_expert_indices_per_model
                else None
            )
            assert self.eplb_state is not None
            self.eplb_state.add_model(
                self.model,
                self.model_config,
                global_expert_load,
                old_global_expert_indices,
                rank_mapping,
            )
            if self.eplb_state.is_async:
                self.eplb_state.start_async_loop(rank_mapping=rank_mapping)

    def get_model(self) -> nn.Module:
        return self.model

    def warming_up_model(self) -> None:
        logger.info("Warming up model for the compilation...")
        # Only generate graph for the generic shape
        with _set_global_compilation_settings(self.vllm_config):
            self._dummy_run(
                min(
                    max(16, self.max_num_reqs),
                    self.scheduler_config.max_num_batched_tokens,
                )
            )

        logger.info("Warming up done.")

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        pass

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        # Note: For CPU backend, dp padding is not required for now.
        return 0, None


@contextmanager
def _torch_cuda_wrapper():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    cuda_event = torch.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.Event = cuda_event
        torch.cuda.Stream = cuda_stream


@contextmanager
def _set_global_compilation_settings(config: VllmConfig):
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    # Note: The MKLDNN and CPPGEMM backend requires freezing parameters.
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
