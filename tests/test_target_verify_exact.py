# SPDX-License-Identifier: Apache-2.0
"""Regression tests for optional exact MTP target verification."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import target_verify_exact as exact


@pytest.fixture(autouse=True)
def _disable_exact_after_test():
    exact._set_enabled_for_tests(False)
    yield
    exact._set_enabled_for_tests(False)


def _quantized_linear(bits: int) -> nn.QuantizedLinear:
    module = nn.QuantizedLinear(
        512,
        256,
        bias=False,
        group_size=64,
        bits=bits,
    )
    module.scales = module.scales.astype(mx.bfloat16)
    module.biases = module.biases.astype(mx.bfloat16)
    return module


def _independent_m1(module, inputs: mx.array) -> mx.array:
    batches = []
    for batch in range(int(inputs.shape[0])):
        batches.append(
            mx.concatenate(
                [
                    module(inputs[batch : batch + 1, row : row + 1])
                    for row in range(int(inputs.shape[1]))
                ],
                axis=1,
            )
        )
    return mx.concatenate(batches, axis=0)


@pytest.mark.parametrize("bits", [2, 4, 5, 6, 8])
@pytest.mark.parametrize("batches", [1, 2])
def test_exact_quantized_linear_matches_independent_m1(bits, batches):
    module = _quantized_linear(bits)
    inputs = mx.random.normal((batches, 4, 512), dtype=mx.bfloat16)

    expected = _independent_m1(module, inputs)
    actual = exact.exact_quantized_linear(module, inputs)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


def test_qwen_helper_routes_only_target_verification():
    exact._set_enabled_for_tests(True)
    module = _quantized_linear(4)
    inputs = mx.random.normal((1, 4, 512), dtype=mx.bfloat16)
    calls = {"original": 0}

    def original(linear, values, _target_verify):
        calls["original"] += 1
        return linear(values)

    language = SimpleNamespace(_target_verify_linear=original)
    exact.patch_qwen_target_verify_helpers(language)

    verify_output = language._target_verify_linear(module, inputs, True)
    ordinary_output = language._target_verify_linear(module, inputs, False)
    expected_verify = _independent_m1(module, inputs)
    expected_ordinary = module(inputs)
    mx.eval(verify_output, ordinary_output, expected_verify, expected_ordinary)

    assert mx.array_equal(verify_output, expected_verify).item()
    assert mx.array_equal(ordinary_output, expected_ordinary).item()
    assert calls == {"original": 1}


def test_qwen_helper_is_untouched_when_exact_is_disabled():
    calls = {"original": 0}

    def original(linear, values, _target_verify):
        calls["original"] += 1
        return linear(values)

    language = SimpleNamespace(_target_verify_linear=original)
    exact.patch_qwen_target_verify_helpers(language)

    assert language._target_verify_linear is original
    assert not hasattr(language, "_omlx_target_verify_exact_patched")
