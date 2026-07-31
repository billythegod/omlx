# SPDX-License-Identifier: Apache-2.0
"""Inkling Lightning MTP runtime tests (single-checkpoint head)."""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture(scope="module")
def runtime():
    from omlx.patches.mlx_vlm_mtp import inkling_vlm_runtime
    from omlx.patches.mlx_lm_mtp import set_mtp_active, set_mtp_depth

    assert inkling_vlm_runtime.apply()
    set_mtp_active(True)
    set_mtp_depth(4)
    yield inkling_vlm_runtime
    set_mtp_active(False)


def _mtp_language_model():
    import importlib

    from tests.test_mlx_vlm_inkling_compat import _tiny_text_config

    language = importlib.import_module("mlx_vlm.models.inkling.language")
    config = _tiny_text_config()
    config.mtp_num_hidden_layers = 3
    config.mtp_local_layer_ids = [0, 2]
    mx.random.seed(21)
    model = language.LanguageModel(config)
    mx.eval(model.parameters())
    return model


def test_config_plumb_and_attach(runtime):
    import importlib

    inkling_pkg = importlib.import_module("mlx_vlm.models.inkling")
    config = inkling_pkg.ModelConfig.from_dict(
        {
            "model_type": "inkling_mm_model",
            "text_config": {"hidden_size": 32, "num_hidden_layers": 2},
            "mtp_config": {
                "num_nextn_predict_layers": 8,
                "local_layer_ids": [0, 2, 4, 5, 6, 7],
                "chain_hidden_post_norm": False,
            },
        }
    )
    assert config.text_config.mtp_num_hidden_layers == 8
    assert config.text_config.mtp_local_layer_ids == [0, 2, 4, 5, 6, 7]

    model = _mtp_language_model()
    assert hasattr(model, "mtp")
    assert len(model.mtp.blocks) == 3
    assert model._omlx_mtp_decode_enabled
    assert model._omlx_mtp_chain
    assert model._omlx_mtp_head_clone
    assert model._omlx_mtp_head_prenorm
    assert model._omlx_mtp_depth == 3  # clamped to the shipped block count


def test_mtp_forward_fold_and_chain_routing(runtime):
    from omlx.patches.mlx_lm_mtp.batch_generator import _clone_mtp_head_cache

    model = _mtp_language_model()
    mtp_cache = model.make_mtp_cache()
    assert len(mtp_cache) == 3

    hidden = mx.random.normal((1, 5, 32))
    tokens = mx.array([[3, 9, 4, 7, 1]])
    logits, head_hidden = model.mtp_forward(
        hidden, tokens, mtp_cache, return_hidden=True, logits_keep=1
    )
    mx.eval(logits, head_hidden)
    assert logits.shape == (1, 1, 128)
    # The fold runs block 0 on the persistent cache only.
    assert mtp_cache[0][0].state[0].shape[2] == 5
    assert mtp_cache[1][0].offset == 0
    assert mtp_cache[2][0].offset == 0

    # Chain steps run on a per-cycle clone and advance block 0 -> 1 -> 2.
    chain_cache = _clone_mtp_head_cache(mtp_cache)
    h = head_hidden[:, -1:]
    tok = mx.array([[11]])
    for expected_block, expected_len in ((0, 6), (1, 1), (2, 1)):
        logits, h = model.mtp_forward(h, tok, chain_cache, return_hidden=True)
        h = h[:, -1:]
        mx.eval(logits)
        assert chain_cache[expected_block][0].state[0].shape[2] == expected_len, (
            f"chain step should have used block {expected_block}"
        )
    # The persistent cache is untouched by chain steps.
    assert mtp_cache[0][0].state[0].shape[2] == 5
    assert mtp_cache[1][0].offset == 0

    # A new fold resets the chain counter.
    model.mtp_forward(hidden[:, -1:], tokens[:, -1:], mtp_cache, logits_keep=1)
    assert model._omlx_inkling_chain_step == 0


def test_verify_rollback_matches_sequential_decode(runtime):
    """Rolling back a rejected verify chunk must leave the backbone cache
    equivalent to having decoded only the accepted tokens one by one."""
    model = _mtp_language_model()
    prompt = mx.array([[5, 17, 42, 91, 12, 63]])
    step_tokens = [7, 33, 54, 76]  # verify chunk; accept first 3, reject last

    ref_cache = model.make_cache()
    model(prompt, cache=ref_cache)
    for tok in step_tokens[:3]:
        model(mx.array([[tok]]), cache=ref_cache)

    cache = model.make_cache()
    model(prompt, cache=cache)
    verify = mx.array([step_tokens])
    out = model(verify, cache=cache, return_hidden=True)
    assert isinstance(out.gdn_states, dict)
    assert out.gdn_states["verify_len"] == 4

    accepted = model.rollback_speculative_cache(
        cache, out.gdn_states, accepted=2, block_size=4
    )
    assert accepted == 2
    for layer_cache, ref_layer in zip(cache, ref_cache):
        assert layer_cache[0].offset == ref_layer[0].offset
        for slot in range(4):
            got = layer_cache[1][slot]
            want = ref_layer[1][slot]
            assert got is not None and want is not None
            diff = mx.max(mx.abs(got - want)).item()
            assert diff < 1e-4, f"conv slot {slot} diverged after rollback: {diff}"

    ref_out = model(mx.array([[100]]), cache=ref_cache)
    test_out = model(mx.array([[100]]), cache=cache)
    mx.eval(ref_out.logits, test_out.logits)
    diff = mx.max(mx.abs(test_out.logits - ref_out.logits)).item()
    assert diff < 1e-3, f"post-rollback logits diverged: {diff}"


def test_sanitize_hook_maps_mtp_keys(runtime):
    import importlib

    inkling_mod = importlib.import_module("mlx_vlm.models.inkling.inkling")
    model = inkling_mod.Model.__new__(inkling_mod.Model)

    hidden, inter = 8, 4
    w13 = mx.arange(2 * inter * hidden, dtype=mx.float32).reshape(2 * inter, hidden)
    weights = {
        "model.mtp.layers.0.input_proj.weight": mx.zeros((hidden, 2 * hidden)),
        "model.mtp.layers.0.embed_norm.weight": mx.ones((hidden,)),
        "model.mtp.layers.0.transformer_block.attn.wq_du.weight": mx.zeros(
            (hidden, hidden)
        ),
        "model.mtp.layers.0.transformer_block.attn.k_sconv.weight": mx.zeros(
            (hidden, 4, 1)
        ),
        "model.mtp.layers.0.transformer_block.mlp.w13_dn.weight": w13,
        "model.llm.embed.weight": mx.zeros((16, hidden)),
    }
    out = inkling_mod.Model.sanitize(model, weights)
    base = "language_model.mtp.blocks.0."
    assert base + "input_proj.weight" in out
    assert base + "embed_norm.weight" in out
    assert base + "transformer_block.self_attn.q_proj.weight" in out
    assert out[base + "transformer_block.self_attn.k_sconv.conv.weight"].shape == (
        hidden,
        1,
        4,
    )
    gate = out[base + "transformer_block.mlp.gate_proj.weight"]
    ref = w13.reshape(inter, 2, hidden)
    assert mx.array_equal(gate, ref[:, 0, :])
    assert "language_model.model.embed_tokens.weight" in out


def test_prompt_priming_capture(runtime):
    """Chunked prefill through the LanguageModel must fold the prompt
    into a priming context (block-0 head cache) so MTP enters warm."""
    from omlx.patches.mlx_lm_mtp import prompt_priming

    model = _mtp_language_model()
    cache = model.make_cache()
    tokens = mx.array([[3, 9, 4, 7, 1, 8, 2, 6]])
    model(tokens, cache=cache)

    ctx = getattr(model, prompt_priming._CTX_ATTR, None)
    assert ctx is not None, "priming context was not captured"
    assert ctx.mtp_cache[0][0].state[0].shape[2] > 0, (
        "priming fold did not reach the block-0 head cache"
    )
    assert ctx.mtp_cache[1][0].offset == 0
