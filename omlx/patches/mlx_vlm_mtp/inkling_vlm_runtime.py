# SPDX-License-Identifier: Apache-2.0
"""Runtime Lightning MTP for the vendored Inkling VLM.

Single-checkpoint port of mlx-vlm's separate-checkpoint
``speculative/drafters/inkling_mtp`` drafter (commit d688082), matching
vLLM's multi-depth Inkling MTP (vllm-project/vllm#48768): the checkpoint
ships one MTP block per draft depth (``mtp_config.num_nextn_predict_layers``,
8 on Inkling Small), each block being embed/hidden RMSNorms + a 2H->H
input projection + one dense ``InklingDecoderLayer``. The final norm and
LM head are the trunk's own (``speculative_logits_from_hidden``); the
drafter checkpoint's norm is a copy of ``model.llm.norm`` upstream.

Depth mapping onto the omlx chain cycle mirrors the drafter exactly:

* the history fold (``logits_keep=1`` on the persistent
  ``make_mtp_cache()`` list) always runs block 0 and seeds the first
  draft token from its logits;
* chain step j runs block ``min(j, depth-1)`` — the batch generator
  hands the chain a ``_clone_mtp_head_cache`` copy (``_omlx_mtp_head_clone``),
  which is the clone-and-discard equivalent of the drafter's
  snapshot/restore, so blocks past 0 never accumulate committed state.
  The clone comes back as a plain list while the persistent cache is an
  ``_InklingMTPCacheList``, which is how ``mtp_forward`` tells a fold
  from a chain step.

Backbone verify rollback cannot trim the short-conv states, so the
vendored conv stashes its padded input every forward and
``rollback_speculative_cache`` reslices the state at the accepted
boundary (``xp[:, keep : keep + K - 1]``) while trimming the KV subs —
no replay forward, unlike upstream's snapshot/replay.

The head consumes PRE-norm trunk hidden (per-block ``hidden_norm``,
``chain_hidden_post_norm: false``), signalled to the fold via
``_omlx_mtp_head_prenorm``.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_APPLIED = False
_SANITIZE_APPLIED = False

_ATTN = {
    "wq_du": "q_proj",
    "wk_dv": "k_proj",
    "wv_dv": "v_proj",
    "wr_du": "r_proj",
    "wo_ud": "o_proj",
}


class _InklingMTPCacheList(list):
    """Marker list type for the persistent per-block MTP cache."""


def _map_transformer_block(sub: str, v: mx.array) -> dict:
    """Checkpoint ``transformer_block.<sub>`` -> InklingDecoderLayer names.

    Port of the upstream drafter's mapping (drafters/inkling_mtp)."""
    out = {}
    p = "transformer_block."
    if sub.startswith("attn."):
        name, leaf = sub[len("attn.") :].rsplit(".", 1)
        if name in _ATTN:
            out[p + f"self_attn.{_ATTN[name]}.weight"] = v
        elif name in ("q_norm", "k_norm"):
            out[p + f"self_attn.{name}.weight"] = v
        elif name in ("k_sconv", "v_sconv"):
            out[p + f"self_attn.{name}.conv.weight"] = v.transpose(0, 2, 1)
        elif name == "rel_logits_proj":
            out[p + "self_attn.rel_proj"] = v
        else:
            out[p + "self_attn." + name + "." + leaf] = v
    elif sub == "attn_norm.weight":
        out[p + "input_layernorm.weight"] = v
    elif sub == "mlp_norm.weight":
        out[p + "post_attention_layernorm.weight"] = v
    elif sub == "attn_sconv.weight":
        out[p + "attn_sconv.conv.weight"] = v.transpose(0, 2, 1)
    elif sub == "mlp_sconv.weight":
        out[p + "mlp_sconv.conv.weight"] = v.transpose(0, 2, 1)
    elif sub.startswith("mlp."):
        from mlx_vlm.models.inkling.inkling import _split_gate_up

        m = sub[len("mlp.") :]
        mp = p + "mlp."
        if m == "w13_dn.weight":
            g, u = _split_gate_up(v)
            out[mp + "gate_proj.weight"] = g
            out[mp + "up_proj.weight"] = u
        elif m == "w2_md.weight":
            out[mp + "down_proj.weight"] = v
        elif m in ("gate.global_scale", "global_scale"):
            out[mp + "global_scale"] = v
        else:
            out[mp + m] = v
    else:
        out[p + sub] = v
    return out


def _mtp_sanitize_hook(key: str, value: mx.array) -> dict:
    """Map a raw ``model.mtp.layers.N.*`` checkpoint key onto the attached
    ``language_model.mtp.blocks.N.*`` parameter names."""
    for prefix in ("model.mtp.layers.", "mtp.layers."):
        if key.startswith(prefix):
            i, sub = key[len(prefix) :].split(".", 1)
            base = f"language_model.mtp.blocks.{i}."
            if sub in (
                "embed_norm.weight",
                "hidden_norm.weight",
                "input_proj.weight",
            ):
                return {base + sub: value}
            if sub.startswith("transformer_block."):
                tb = sub[len("transformer_block.") :]
                return {
                    base + nk: nv for nk, nv in _map_transformer_block(tb, value).items()
                }
            return {base + sub: value}
    return {}


def apply_sanitize() -> bool:
    """Install the sanitize hook that keeps and remaps mtp.* weights.

    Called from ``apply_mlx_vlm_mtp_patch`` (oQ preserve_mtp path) and
    from ``apply()`` below. Requires the inkling compat vendor namespace.
    """
    global _SANITIZE_APPLIED
    if _SANITIZE_APPLIED:
        return True
    try:
        from omlx.patches.mlx_vlm_inkling_compat import (
            apply_mlx_vlm_inkling_compat_patch,
        )

        apply_mlx_vlm_inkling_compat_patch()
        import mlx_vlm.models.inkling.inkling as inkling_mod
    except Exception as e:
        logger.debug(f"inkling module not importable for MTP sanitize: {e}")
        return False
    inkling_mod._OMLX_MTP_SANITIZE_HOOK = _mtp_sanitize_hook
    _SANITIZE_APPLIED = True
    return True


def apply() -> bool:
    """Apply the inkling runtime Lightning MTP patches. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    if not apply_sanitize():
        return False
    try:
        import mlx_vlm.models.inkling as inkling_pkg
        import mlx_vlm.models.inkling.language as inkling_lang
    except Exception as e:
        logger.debug(f"inkling language module not importable: {e}")
        return False

    _patch_model_config(inkling_pkg)
    _register_mtp_classes(inkling_lang)
    _patch_language_model(inkling_lang)
    _patch_inner_model_capture(inkling_lang)

    # Shared VLMModelAdapter pass-throughs (idempotent).
    from .qwen35_vlm_runtime import _patch_vlm_model_adapter

    _patch_vlm_model_adapter()

    _APPLIED = True
    logger.info("mlx-vlm inkling runtime MTP patch applied")
    return True


def _patch_model_config(inkling_pkg: Any) -> None:
    """Carry the top-level ``mtp_config`` block onto the TextConfig.

    Two hops are required: ``BaseModelConfig.from_dict`` filters by
    dataclass signature (dropping ``mtp_config``), and mlx-vlm's
    ``load_model`` REBUILDS ``text_config`` afterwards via
    ``update_module_configs`` → ``TextConfig.from_dict``, discarding any
    attribute stamped on the first instance. So the ModelConfig wrap
    injects the flattened keys into the raw ``text_config`` dict (the
    same object update_module_configs reads), and the TextConfig wrap
    lifts them onto every rebuilt instance."""
    cls = inkling_pkg.ModelConfig
    if not getattr(cls, "_omlx_mtp_from_dict_patched", False):
        original_from_dict = cls.from_dict.__func__

        def patched_from_dict(cls_inner, params):
            mtp_cfg = (params or {}).get("mtp_config") or {}
            depth = int(
                mtp_cfg.get("num_nextn_predict_layers")
                or mtp_cfg.get("num_mtp_layers")
                or 0
            )
            if depth and isinstance(params.get("text_config"), dict):
                params["text_config"].setdefault("mtp_num_hidden_layers", depth)
                params["text_config"].setdefault(
                    "mtp_local_layer_ids", mtp_cfg.get("local_layer_ids")
                )
            return original_from_dict(cls_inner, params)

        cls.from_dict = classmethod(patched_from_dict)
        cls._omlx_mtp_from_dict_patched = True

    text_cls = inkling_pkg.TextConfig
    if not getattr(text_cls, "_omlx_mtp_from_dict_patched", False):
        original_text_from_dict = text_cls.from_dict.__func__

        def patched_text_from_dict(cls_inner, params):
            instance = original_text_from_dict(cls_inner, params)
            params = params or {}
            instance.mtp_num_hidden_layers = int(
                params.get("mtp_num_hidden_layers", 0) or 0
            )
            instance.mtp_local_layer_ids = params.get("mtp_local_layer_ids")
            return instance

        text_cls.from_dict = classmethod(patched_text_from_dict)
        text_cls._omlx_mtp_from_dict_patched = True


def _register_mtp_classes(inkling_lang: Any) -> None:
    if hasattr(inkling_lang, "InklingMTPModule"):
        return

    from dataclasses import replace

    InklingDecoderLayer = inkling_lang.InklingDecoderLayer

    class InklingMTPBlock(nn.Module):
        def __init__(self, config, layer_idx: int):
            super().__init__()
            self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.input_proj = nn.Linear(
                2 * config.hidden_size, config.hidden_size, bias=False
            )
            self.transformer_block = InklingDecoderLayer(config, layer_idx)

    class InklingMTPModule(nn.Module):
        """Per-depth MTP blocks (upstream drafter layout, trunk-bound)."""

        def __init__(self, text_config):
            super().__init__()
            n = int(getattr(text_config, "mtp_num_hidden_layers", 0) or 0)
            local = set(getattr(text_config, "mtp_local_layer_ids", None) or [])
            layer_config = replace(
                text_config,
                num_hidden_layers=n,
                layer_types=[
                    "hybrid_sliding" if i in local else "hybrid" for i in range(n)
                ],
                mlp_layer_types=["dense"] * n,
                local_layer_ids=None,
            )
            self.blocks = [InklingMTPBlock(layer_config, i) for i in range(n)]

        def __call__(
            self,
            hidden_states: mx.array,
            next_token_ids: mx.array,
            embed_tokens,
            cache,
            block_idx: int = 0,
        ) -> mx.array:
            # block_idx defaults to 0 for generic callers that use the
            # qwen-style 4-arg head signature (oQe's MTP-head imatrix
            # pass); the chain routing in mtp_forward always passes it.
            block = self.blocks[block_idx]
            token_embed = embed_tokens(next_token_ids)
            h = mx.concatenate(
                [block.hidden_norm(hidden_states), block.embed_norm(token_embed)],
                axis=-1,
            )
            h = block.input_proj(h)
            block_cache = cache[block_idx] if cache is not None else None
            return block.transformer_block(h, cache=block_cache)

    inkling_lang.InklingMTPBlock = InklingMTPBlock
    inkling_lang.InklingMTPModule = InklingMTPModule


def _patch_language_model(inkling_lang: Any) -> None:
    cls = inkling_lang.LanguageModel
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, config):
        from . import is_mtp_attach_enabled
        from ..mlx_lm_mtp import is_mtp_active

        original_init(self, config)
        n_mtp = int(getattr(config, "mtp_num_hidden_layers", 0) or 0)
        attach_enabled = bool(is_mtp_attach_enabled())
        self._omlx_mtp_decode_enabled = bool(
            n_mtp > 0 and attach_enabled and is_mtp_active()
        )
        if n_mtp > 0 and attach_enabled:
            self.mtp = inkling_lang.InklingMTPModule(config)
        if self._omlx_mtp_decode_enabled:
            import weakref

            from ..mlx_lm_mtp import get_mtp_depth

            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = min(int(get_mtp_depth()), n_mtp)
            # Chain steps must run on a per-cycle clone: blocks past 0
            # never hold committed state (drafter snapshot/restore
            # equivalent) and block 0's history stays draft-free.
            self._omlx_mtp_head_clone = True
            # The head normalizes its hidden input per block
            # (chain_hidden_post_norm=false) — fold must NOT trunk-norm.
            self._omlx_mtp_head_prenorm = True
            # Prompt-priming capture runs inside InklingModel.__call__,
            # which has no reference back to this LanguageModel; without
            # priming the head enters decode cold (its block-0 cache never
            # saw the prompt) and the adaptive controller parks the
            # request at depth 0 before the fold can warm it up.
            self.model._omlx_mtp_prime_host = weakref.ref(self)

    def __call__(self, inputs=None, cache=None, **kwargs):
        out = original_call(self, inputs, cache=cache, **kwargs)
        if (
            kwargs.get("return_hidden")
            and cache is not None
            and getattr(self, "_omlx_mtp_decode_enabled", False)
            and inputs is not None
            and getattr(out, "gdn_states", None) is None
        ):
            # Rollback marker: the conv-input stashes written during this
            # forward cover exactly these verify positions.
            out.gdn_states = {"verify_len": int(inputs.shape[1])}
        return out

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        if isinstance(mtp_cache, _InklingMTPCacheList):
            # History fold — always block 0; a fold starts a new cycle.
            block_idx = 0
            self._omlx_inkling_chain_step = 0
        else:
            step = int(getattr(self, "_omlx_inkling_chain_step", 0))
            block_idx = min(step, len(self.mtp.blocks) - 1)
            self._omlx_inkling_chain_step = step + 1
        head_hidden = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
            block_idx,
        )
        logits_source = head_hidden
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:, :]
        logits = self.speculative_logits_from_hidden(logits_source)
        if return_hidden:
            return logits, head_hidden
        return logits

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return _InklingMTPCacheList(
                CacheList(KVCache(), ArraysCache(4)) for _ in self.mtp.blocks
            )
        return _InklingMTPCacheList()

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        """Trim KV subs and reslice conv states at the accepted boundary.

        Replaces the vendored snapshot/replay rollback (an extra backbone
        forward per cycle) with an O(1) reslice from the conv-input
        stashes the vendored ``InklingShortConvolution`` records."""
        if isinstance(accepted, mx.array):
            accepted = int(accepted.max().item()) if accepted.size else 0
        elif not isinstance(accepted, int):
            accepted = max(int(a) for a in accepted)
        verify_len = None
        if isinstance(gdn_states, dict):
            verify_len = gdn_states.get("verify_len")
        if verify_len is None:
            verify_len = int(block_size)
        keep = accepted + 1
        trim_n = int(verify_len) - keep
        for c in caches:
            if c is None:
                continue
            kv, conv = c[0], c[1]
            if trim_n > 0 and hasattr(kv, "trim"):
                kv.trim(trim_n)
            stash = getattr(conv, "_omlx_verify_xp", None)
            if not stash:
                continue
            for idx, xp in stash.items():
                k_minus_1 = int(xp.shape[1]) - int(verify_len)
                if k_minus_1 <= 0:
                    continue
                conv[idx] = xp[:, keep : keep + k_minus_1, :]
        return accepted

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.rollback_speculative_cache = rollback_speculative_cache
    cls._omlx_mtp_runtime_patched = True


def _patch_inner_model_capture(inkling_lang: Any) -> None:
    """Wrap ``InklingModel.__call__`` to fold prompt chunks into the head.

    The LanguageModel always calls the inner model with
    ``skip_final_norm=True``, so the captured hidden is the PRE-norm trunk
    output — the convention this head consumes. Image prompts arrive via
    ``input_embeddings`` and are skipped (same limitation as qwen); all
    real gating lives in ``prompt_priming.maybe_capture`` and fails safe
    to an unprimed entry.
    """
    cls = inkling_lang.InklingModel
    if getattr(cls, "_omlx_mtp_prime_capture_patched", False):
        return

    from ..mlx_lm_mtp import prompt_priming

    original_call = cls.__call__

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        skip_final_norm=False,
        **kwargs,
    ):
        out = original_call(
            self,
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            skip_final_norm=skip_final_norm,
            **kwargs,
        )
        if (
            input_embeddings is None
            and cache is not None
            and skip_final_norm
            and inputs is not None
        ):
            host_ref = getattr(self, "_omlx_mtp_prime_host", None)
            host = host_ref() if host_ref is not None else None
            if host is not None:
                try:
                    prompt_priming.maybe_capture(host, inputs, out, cache)
                except Exception:
                    logger.debug(
                        "Inkling MTP prompt-priming capture failed", exc_info=True
                    )
        return out

    cls.__call__ = __call__
    cls._omlx_mtp_prime_capture_patched = True


__all__ = ["apply", "apply_sanitize"]
