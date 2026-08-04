# SPDX-License-Identifier: Apache-2.0
"""Optional M=1 affine projection routing for MTP target verification.

MLX and mlx-vlm may use a different reduction order for multi-row quantized
matmuls than for independent decode rows. The hidden
``server.mtp_target_verify_exact`` setting enables a QA path that evaluates
each verify position through the same M=1 ``QuantizedLinear`` call used by
standard decode.

The default is disabled. In that state this module does not patch
``QuantizedLinear`` or mlx-vlm's target-verification helpers.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..settings import get_settings

logger = logging.getLogger(__name__)

_ROUTE_ARMED = threading.local()
_ENABLED = False
_QL_PATCHED = False


def _settings_enabled() -> bool:
    try:
        return bool(get_settings().server.mtp_target_verify_exact)
    except RuntimeError:
        return False


def target_verify_exact_enabled() -> bool:
    """Return whether the startup-only exact target-verification path is on."""
    return _ENABLED


def set_target_verify_exact_armed(flag: bool) -> None:
    """Arm exact routing only for the duration of an MTP target forward."""
    if not _ENABLED:
        return
    _ROUTE_ARMED.value = bool(flag)


def _is_armed() -> bool:
    return _ENABLED and getattr(_ROUTE_ARMED, "value", False)


def _exact_eligible(linear: Any, x: mx.array) -> bool:
    return (
        isinstance(linear, nn.QuantizedLinear)
        and x.ndim == 3
        and int(x.shape[1]) > 1
        and int(getattr(linear, "bits", 0)) in (2, 4, 5, 6, 8)
        and getattr(linear, "mode", "affine") == "affine"
    )


def exact_quantized_linear(
    linear: nn.QuantizedLinear,
    x: mx.array,
    call: Callable[[mx.array], mx.array] | None = None,
) -> mx.array:
    """Evaluate a rank-3 affine projection as independent M=1 calls."""
    invoke = call or linear
    batches = []
    for batch in range(int(x.shape[0])):
        batches.append(
            mx.concatenate(
                [
                    invoke(x[batch : batch + 1, row : row + 1])
                    for row in range(int(x.shape[1]))
                ],
                axis=1,
            )
        )
    return mx.concatenate(batches, axis=0)


def configure_target_verify_exact() -> bool:
    """Install the process-wide router when the hidden setting enables it."""
    global _ENABLED, _QL_PATCHED

    _ENABLED = _settings_enabled()
    if not _ENABLED:
        _ROUTE_ARMED.value = False
        return True
    if _QL_PATCHED:
        return True

    cls = nn.QuantizedLinear
    if getattr(cls, "_omlx_target_verify_exact_patched", False):
        _QL_PATCHED = True
        return True

    original_call = cls.__call__

    def patched_call(self, x):
        if _is_armed() and _exact_eligible(self, x):
            return exact_quantized_linear(
                self,
                x,
                call=lambda values: original_call(self, values),
            )
        return original_call(self, x)

    cls.__call__ = patched_call
    cls._omlx_target_verify_exact_patched = True
    _QL_PATCHED = True
    logger.info("Exact M=1 affine MTP target verification enabled")
    return True


def patch_qwen_target_verify_helpers(language_module: Any) -> None:
    """Route mlx-vlm's direct target-verify QMV helper through M=1 calls."""
    if not _ENABLED:
        return
    if getattr(language_module, "_omlx_target_verify_exact_patched", False):
        return

    original = getattr(language_module, "_target_verify_linear", None)
    if original is None:
        return

    def patched(linear, x: mx.array, target_verify: bool):
        if target_verify and _exact_eligible(linear, x):
            return exact_quantized_linear(linear, x)
        return original(linear, x, target_verify)

    language_module._target_verify_linear = patched
    language_module._omlx_target_verify_exact_patched = True


def _set_enabled_for_tests(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)
    if not _ENABLED:
        _ROUTE_ARMED.value = False
