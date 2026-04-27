# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Helpers for selecting Triton fallback implementations.

The selector resolves torch-triton fallback kernels once per process import path,
and validates that the selected fallback wrapper keeps a compatible callable
signature with the local Triton kernel.
"""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from typing import Any

from vllm.triton_utils import HAS_TRITON


def _callable_for_signature(kernel: Any) -> Any:
    """Return the underlying callable for signature comparison."""

    return kernel.func if hasattr(kernel, "func") else kernel


def _signature_param_signature(
    signature: inspect.Signature,
) -> tuple[tuple[str, str], ...]:
    """Extract a stable, default-safe argument signature for alignment.

    The fallback implementations commonly use names like `output`/`logits`
    while Triton kernels use `output_ptr`/`logits_ptr`. Suffix normalization
    keeps compatibility while still detecting true signature drift.
    """

    def _normalize_param_name(name: str) -> str:
        if name.endswith("_ptr"):
            return name[: -len("_ptr")]
        return name

    return tuple(
        (_normalize_param_name(param.name), param.kind.name)
        for param in signature.parameters.values()
    )


@lru_cache(maxsize=1)
def _resolve_fallback_module_name() -> str:
    if HAS_TRITON:
        # Keep this for completeness; callers should not invoke fallback resolution
        # when Triton is available.
        return "vllm.utils.torch_triton_utils"

    import vllm.envs as envs

    if envs.VLLM_TRITON_UTILS_SELECT.upper() == "MCPU":
        return "vllm.utils.mcpu_triton_utils"
    return "vllm.utils.torch_triton_utils"


def resolve_fallback_kernel(local_kernel: Any, kernel_name: str) -> Any:
    """Resolve and validate a fallback kernel once.

    Returns a drop-in callable:
    - when Triton is enabled: returns the local kernel.
    - when Triton is disabled: returns the selected fallback wrapper kernel.
    """

    if HAS_TRITON:
        return local_kernel

    module_name = _resolve_fallback_module_name()
    module = importlib.import_module(module_name)
    fallback_kernel = getattr(module, kernel_name)

    local_signature = inspect.signature(_callable_for_signature(local_kernel))
    fallback_signature = inspect.signature(_callable_for_signature(fallback_kernel))

    if _signature_param_signature(local_signature) != _signature_param_signature(
        fallback_signature
    ):
        raise RuntimeError(
            f"Kernel signature mismatch for '{kernel_name}'. "
            f"Triton signature: {local_signature}; "
            f"fallback signature: {fallback_signature}; "
            f"module: {module_name}"
        )

    return fallback_kernel
