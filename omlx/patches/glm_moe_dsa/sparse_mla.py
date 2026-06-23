# Copyright © 2026 Apple Inc.

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx

from .kernels import fast as glm_fast


def scores_to_block_indices(
    scores: mx.array,
    *,
    block_budget: int,
    q_block_size: int = 8,
    k_block_size: int = 16,
    recent_blocks: int = 0,
    sort_indices: bool = True,
) -> Optional[mx.array]:
    """Select a fixed K-block table from dense DSA indexer scores.

    The result is a page table shaped [B, 1, q_blocks, block_budget]. Each row
    contains K-block ids, not token ids. This mirrors the sparse MLA metadata
    consumed by CUDA serving engines while keeping the workspace bounded.
    """

    if scores.ndim != 4 or scores.shape[1] != 1 or block_budget <= 0:
        return None

    B, _, L, K = scores.shape
    q_blocks = (L + q_block_size - 1) // q_block_size
    k_blocks = (K + k_block_size - 1) // k_block_size
    block_budget = min(block_budget, k_blocks)

    pad_value = mx.array(mx.finfo(scores.dtype).min, scores.dtype)
    if K != k_blocks * k_block_size:
        scores = mx.pad(
            scores,
            [(0, 0), (0, 0), (0, 0), (0, k_blocks * k_block_size - K)],
            constant_values=pad_value,
        )
    block_scores = mx.max(
        scores.reshape(B, 1, L, k_blocks, k_block_size),
        axis=-1,
    )
    if L != q_blocks * q_block_size:
        block_scores = mx.pad(
            block_scores,
            [(0, 0), (0, 0), (0, q_blocks * q_block_size - L), (0, 0)],
            constant_values=pad_value,
        )
    block_scores = block_scores.reshape(B, 1, q_blocks, q_block_size, k_blocks)
    block_scores = mx.max(block_scores, axis=3)

    return block_scores_to_indices(
        block_scores,
        block_budget=block_budget,
        recent_blocks=recent_blocks,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        query_length=L,
        key_length=K,
        sort_indices=sort_indices,
    )


def _boost_recent_block_scores(
    block_scores: mx.array,
    *,
    recent_blocks: int,
    q_block_size: int,
    k_block_size: int,
    query_length: int,
    key_length: int,
) -> mx.array:
    if recent_blocks <= 0:
        return block_scores

    q_blocks = block_scores.shape[-2]
    k_blocks = block_scores.shape[-1]
    recent_blocks = min(recent_blocks, k_blocks)

    q_block = mx.arange(q_blocks, dtype=mx.uint32)[:, None]
    k_block = mx.arange(k_blocks, dtype=mx.uint32)[None, :]
    q_end_local = (
        mx.minimum(
            (q_block + 1) * q_block_size,
            mx.array(query_length, dtype=mx.uint32),
        )
        - 1
    )
    q_abs_end = q_end_local + (key_length - query_length)
    end_block = q_abs_end // k_block_size
    recent_mask = (k_block <= end_block) & (k_block + recent_blocks > end_block)
    recent_mask = mx.reshape(recent_mask, (1, 1, q_blocks, k_blocks))
    boost_value = mx.array(mx.finfo(block_scores.dtype).max, block_scores.dtype)
    return mx.where(recent_mask, boost_value, block_scores)


def block_scores_to_indices(
    block_scores: mx.array,
    *,
    block_budget: int,
    recent_blocks: int = 0,
    q_block_size: Optional[int] = None,
    k_block_size: Optional[int] = None,
    query_length: Optional[int] = None,
    key_length: Optional[int] = None,
    sort_indices: bool = True,
) -> Optional[mx.array]:
    """Select block ids from pre-reduced [B, 1, q_blocks, k_blocks] scores."""

    if block_scores.ndim != 4 or block_scores.shape[1] != 1 or block_budget <= 0:
        return None

    block_budget = min(block_budget, block_scores.shape[-1])
    if recent_blocks > 0:
        if (
            q_block_size is None
            or k_block_size is None
            or query_length is None
            or key_length is None
        ):
            return None
        recent_blocks = min(recent_blocks, block_budget)
        block_scores = _boost_recent_block_scores(
            block_scores,
            recent_blocks=recent_blocks,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            query_length=query_length,
            key_length=key_length,
        )
    indices = mx.argpartition(block_scores, kth=-block_budget, axis=-1)[
        ..., -block_budget:
    ].astype(mx.uint32)
    if sort_indices:
        indices = mx.sort(indices, axis=-1)
    return indices


@lru_cache(maxsize=None)
def _make_index_score_reduce_kernel():
    if not mx.metal.is_available():
        return None

    source = r"""
        const uint elem = thread_position_in_grid.x;
        const uint total = B * L * K;
        if (elem >= total) {
          return;
        }

        const uint k_pos = elem % K;
        const uint q_pos = (elem / K) % L;
        const uint b = elem / (L * K);

        if (CAUSAL && k_pos > K - L + q_pos) {
          out[elem] = static_cast<T>(-INFINITY);
          return;
        }

        float acc = 0.0f;
        #pragma clang loop unroll(full)
        for (uint h = 0; h < H; ++h) {
          const uint score_idx = ((b * H + h) * L + q_pos) * K + k_pos;
          const uint weight_idx = (b * H + h) * L + q_pos;
          const float s = static_cast<float>(head_scores[score_idx]);
          const float w = static_cast<float>(weights[weight_idx]);
          acc += metal::max(s, 0.0f) * w;
        }
        out[elem] = static_cast<T>(acc);
    """

    return glm_fast.metal_kernel(
        name="glm_dsa_index_score_reduce",
        input_names=["head_scores", "weights"],
        output_names=["out"],
        source=source,
    )


def fused_index_score_reduce(
    head_scores: mx.array,
    weights: mx.array,
    *,
    causal: bool = False,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    """Fuse ReLU, per-head weighting, causal fill, and head reduction."""

    kernel = _make_index_score_reduce_kernel()
    if (
        kernel is None
        or head_scores.ndim != 4
        or weights.ndim != 4
        or weights.shape[-1] != 1
        or head_scores.shape[:3] != weights.shape[:3]
    ):
        return None

    B, H, L, K = head_scores.shape
    short_k_threadgroup = int(
        os.environ.get("MLX_LM_GLM_DSA_INDEX_REDUCE_THREADGROUP_SIZE", "512")
    )
    long_k_threshold = int(
        os.environ.get("MLX_LM_GLM_DSA_INDEX_REDUCE_LONG_K_THRESHOLD", "32768")
    )
    long_k_threadgroup = int(
        os.environ.get("MLX_LM_GLM_DSA_INDEX_REDUCE_LONG_K_THREADGROUP_SIZE", "256")
    )
    threadgroup_size = (
        long_k_threadgroup if K > long_k_threshold else short_k_threadgroup
    )
    return kernel(
        inputs=[head_scores, weights],
        template=[
            ("T", head_scores.dtype),
            ("B", B),
            ("H", H),
            ("L", L),
            ("K", K),
            ("CAUSAL", causal),
        ],
        grid=(B * L * K, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[(B, 1, L, K)],
        output_dtypes=[head_scores.dtype],
        stream=stream or mx.gpu,
    )[0]


def fused_indexer_scores(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    *,
    causal: bool = False,
    unused_causal_prefix_topk: int = 0,
    skip_causal_future_store: bool = False,
    causal_q_offset: int = -1,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    """Compute GLM DSA indexer logits without materializing per-head scores.

    This is the MLX equivalent of the vLLM/SGLang MQA-logits indexer path:
    the Steel kernel computes ``sum_h relu(q_h @ k.T) * weight_h`` directly
    into [B, 1, L, K]. The Metal kernel is specialized for GLM-5.2's
    [H=32, D=128] indexer and 64-token M/N tiles, so non-multiple prompt
    lengths are padded and sliced back exactly.
    """

    if (
        not hasattr(glm_fast, "dsa_indexer_scores")
        or queries.ndim != 4
        or keys.ndim != 4
        or weights.ndim != 3
        or queries.shape[0] != keys.shape[0]
        or queries.shape[0] != weights.shape[0]
        or queries.shape[1] != 32
        or keys.shape[1] != 1
        or queries.shape[2] != weights.shape[1]
        or queries.shape[1] != weights.shape[2]
        or queries.shape[3] != 128
        or keys.shape[3] != 128
        or keys.shape[2] < int(os.environ.get("MLX_LM_GLM_DSA_INDEXER_MIN_K", "4096"))
        or queries.dtype != keys.dtype
        or queries.dtype != weights.dtype
    ):
        return None

    B, H, L, D = queries.shape
    K = keys.shape[2]
    q_pad = (-L) % 64
    k_pad = (-K) % 64

    q = queries
    k = keys
    w = weights
    if q_pad:
        q = mx.pad(q, [(0, 0), (0, 0), (0, q_pad), (0, 0)])
        w = mx.pad(w, [(0, 0), (0, q_pad), (0, 0)])
    if k_pad:
        k = mx.pad(k, [(0, 0), (0, 0), (0, k_pad), (0, 0)])
    if q_pad or k_pad:
        unused_causal_prefix_topk = 0

    scores = glm_fast.dsa_indexer_scores(
        q,
        k,
        w,
        causal=causal,
        unused_causal_prefix_topk=unused_causal_prefix_topk,
        skip_causal_future_store=skip_causal_future_store,
        causal_q_offset=causal_q_offset,
        stream=stream or mx.gpu,
    )
    if q_pad or k_pad:
        scores = scores[:, :, :L, :K]
    return scores


def fused_indexer_scores_high_histogram(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    *,
    causal: bool = False,
    unused_causal_prefix_topk: int = 0,
    skip_causal_future_store: bool = False,
    causal_q_offset: int = -1,
    stream: Optional[mx.Stream] = None,
) -> Optional[tuple[mx.array, mx.array]]:
    """Compute exact indexer scores plus high-byte top-k histogram state."""

    required = (
        "dsa_indexer_scores_high_histogram",
        "dsa_topk_indices_with_high_state",
    )
    if (
        not all(hasattr(glm_fast, name) for name in required)
        or queries.ndim != 4
        or keys.ndim != 4
        or weights.ndim != 3
        or queries.shape[0] != keys.shape[0]
        or queries.shape[0] != weights.shape[0]
        or queries.shape[1] != 32
        or keys.shape[1] != 1
        or queries.shape[2] != weights.shape[1]
        or queries.shape[1] != weights.shape[2]
        or queries.shape[3] != 128
        or keys.shape[3] != 128
        or queries.shape[2] % 64 != 0
        or keys.shape[2] % 64 != 0
        or keys.shape[2] < int(os.environ.get("MLX_LM_GLM_DSA_INDEXER_MIN_K", "4096"))
        or queries.dtype != keys.dtype
        or queries.dtype != weights.dtype
    ):
        return None

    scores, high_hist = glm_fast.dsa_indexer_scores_high_histogram(
        queries,
        keys,
        weights,
        causal=causal,
        unused_causal_prefix_topk=unused_causal_prefix_topk,
        skip_causal_future_store=skip_causal_future_store,
        causal_q_offset=causal_q_offset,
        stream=stream or mx.gpu,
    )
    return scores, high_hist


def _dsa_histogram_threshold(hist: mx.array, topk: int) -> mx.array:
    rev_cumsum = mx.cumsum(hist[..., ::-1], axis=-1)
    target = mx.array(topk, dtype=mx.uint32)
    rev_idx = mx.argmax((rev_cumsum >= target).astype(mx.uint32), axis=-1)
    threshold_hi = (mx.array(255, dtype=mx.uint32) - rev_idx).astype(mx.uint32)

    prev_idx = mx.maximum(
        rev_idx.astype(mx.int32) - mx.array(1, dtype=mx.int32),
        mx.array(0, dtype=mx.int32),
    ).astype(mx.uint32)
    prev = mx.take_along_axis(rev_cumsum, prev_idx[..., None], axis=-1)[..., 0]
    greater = mx.where(
        rev_idx > mx.array(0, dtype=mx.uint32),
        prev,
        mx.array(0, dtype=mx.uint32),
    )
    return mx.stack([threshold_hi, greater.astype(mx.uint32)], axis=-1)


def _dsa_low_histogram_threshold(
    high_state: mx.array, low_hist: mx.array, topk: int
) -> mx.array:
    greater_hi = high_state[..., 1]
    target = mx.array(topk, dtype=mx.uint32) - greater_hi
    rev_cumsum = mx.cumsum(low_hist[..., ::-1], axis=-1)
    rev_idx = mx.argmax((rev_cumsum >= target[..., None]).astype(mx.uint32), axis=-1)
    threshold_lo = (mx.array(255, dtype=mx.uint32) - rev_idx).astype(mx.uint32)

    prev_idx = mx.maximum(
        rev_idx.astype(mx.int32) - mx.array(1, dtype=mx.int32),
        mx.array(0, dtype=mx.int32),
    ).astype(mx.uint32)
    prev = mx.take_along_axis(rev_cumsum, prev_idx[..., None], axis=-1)[..., 0]
    greater_low = mx.where(
        rev_idx > mx.array(0, dtype=mx.uint32),
        prev,
        mx.array(0, dtype=mx.uint32),
    )
    threshold_key = high_state[..., 0] * mx.array(256, dtype=mx.uint32) + threshold_lo
    greater = greater_hi + greater_low.astype(mx.uint32)
    return mx.stack(
        [threshold_key.astype(mx.uint32), greater.astype(mx.uint32)], axis=-1
    )


def fused_indexer_topk_indices(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    topk: int,
    *,
    causal: bool = False,
    causal_q_offset: int = -1,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    """Exact GLM DSA indexer top-k without a materialized score matrix.

    The implementation mirrors the existing 16-bit radix top-k, but moves the
    score production into Steel MMA passes: high-byte histogram, low-byte
    histogram for the threshold bucket, then index emission. It preserves exact
    top-k semantics up to normal equal-score tie freedom.
    """

    required = (
        "dsa_indexer_score_histogram",
        "dsa_indexer_score_low_histogram",
        "dsa_indexer_topk_emit",
    )
    if (
        not all(hasattr(glm_fast, name) for name in required)
        or queries.ndim != 4
        or keys.ndim != 4
        or weights.ndim != 3
        or queries.shape[0] != keys.shape[0]
        or queries.shape[0] != weights.shape[0]
        or queries.shape[1] != 32
        or keys.shape[1] != 1
        or queries.shape[2] != weights.shape[1]
        or queries.shape[1] != weights.shape[2]
        or queries.shape[3] != 128
        or keys.shape[3] != 128
        or queries.shape[2] % 64 != 0
        or keys.shape[2] % 64 != 0
        or keys.shape[2] < topk
        or keys.shape[2] < int(os.environ.get("MLX_LM_GLM_DSA_INDEXER_MIN_K", "4096"))
        or queries.dtype != keys.dtype
        or queries.dtype != weights.dtype
    ):
        return None

    s = stream or mx.gpu
    high_hist = glm_fast.dsa_indexer_score_histogram(
        queries,
        keys,
        weights,
        causal=causal,
        causal_q_offset=causal_q_offset,
        stream=s,
    )
    high_state = _dsa_histogram_threshold(high_hist, topk)
    low_hist = glm_fast.dsa_indexer_score_low_histogram(
        queries,
        keys,
        weights,
        high_state,
        causal=causal,
        causal_q_offset=causal_q_offset,
        stream=s,
    )
    threshold = _dsa_low_histogram_threshold(high_state, low_hist, topk)
    return glm_fast.dsa_indexer_topk_emit(
        queries,
        keys,
        weights,
        threshold,
        topk,
        causal=causal,
        causal_q_offset=causal_q_offset,
        stream=s,
    )


def fused_topk_indices_with_high_histogram(
    scores: mx.array,
    high_hist: mx.array,
    topk: int,
    *,
    bucketed: bool = False,
    causal_valid_prefix: bool = False,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    if (
        not hasattr(glm_fast, "dsa_topk_indices_with_high_state")
        or scores.ndim != 4
        or scores.shape[1] != 1
        or high_hist.ndim != 3
        or high_hist.shape[0] != scores.shape[0]
        or high_hist.shape[1] != scores.shape[2]
        or high_hist.shape[2] != 256
        or high_hist.dtype != mx.uint32
    ):
        return None
    s = stream or mx.gpu
    high_state = _dsa_histogram_threshold(high_hist, topk)
    return glm_fast.dsa_topk_indices_with_high_state(
        scores,
        high_state,
        topk,
        bucketed=bucketed,
        causal_valid_prefix=causal_valid_prefix,
        stream=s,
    )


def sparse_mla_attention(
    q_latent: mx.array,
    q_pe: mx.array,
    kv_latent: mx.array,
    k_pe: mx.array,
    topk_indices: mx.array,
    scale: float,
    *,
    topk_valid_prefix: bool = False,
    causal_prefix_indices: bool = False,
    topk_length: Optional[mx.array] = None,
    causal_prefix_rows: int = 0,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    """Sparse MLA prefill over per-query DSA top-k indices.

    This mirrors the FlashMLA sparse prefill contract used by vLLM/SGLang:
    attention scores are computed over [latent, rope] keys, values are the
    latent KV cache, and the caller applies the MLA output projection after.

    Shapes:
      q_latent: [B, H, L, 512]
      q_pe: [B, H, L, 64]
      kv_latent: [B, 1, K, 512]
      k_pe: [B, 1, K, 64]
      topk_indices: [B, 1, L, TOPK]
      topk_length: optional [B, L] or [B, 1, L] valid prefix length
    """

    if (
        q_latent.ndim != 4
        or q_pe.ndim != 4
        or kv_latent.ndim != 4
        or k_pe.ndim != 4
        or topk_indices.ndim != 4
        or kv_latent.shape[1] != 1
        or k_pe.shape[1] != 1
    ):
        return None

    B, H, L, D_LATENT = q_latent.shape
    K = kv_latent.shape[2]
    D_PE = q_pe.shape[-1]
    topk_rows = topk_indices.shape[2]
    compact_prefix = causal_prefix_rows > 0 and topk_rows != L

    if (
        L <= 1
        or q_pe.shape[:3] != (B, H, L)
        or kv_latent.shape[:3] != (B, 1, K)
        or k_pe.shape[:3] != (B, 1, K)
        or topk_indices.shape[:2] != (B, 1)
        or not (
            topk_rows == L
            or (
                compact_prefix
                and topk_rows + causal_prefix_rows == L
                and causal_prefix_indices
                and topk_valid_prefix
            )
        )
        or kv_latent.shape[-1] != D_LATENT
        or k_pe.shape[-1] != D_PE
        or D_LATENT != 512
        or D_PE != 64
        or q_latent.dtype not in (mx.float16, mx.bfloat16)
        or q_pe.dtype != q_latent.dtype
        or kv_latent.dtype != q_latent.dtype
        or k_pe.dtype != q_latent.dtype
    ):
        return None

    if not hasattr(glm_fast, "glm_dsa_sparse_mla_attention"):
        return None

    topk = (
        topk_indices
        if topk_indices.dtype == mx.uint32
        else topk_indices.astype(mx.uint32)
    )
    if topk_length is not None and topk_length.dtype != mx.uint32:
        topk_length = topk_length.astype(mx.uint32)
    return glm_fast.glm_dsa_sparse_mla_attention(
        q_latent,
        q_pe,
        kv_latent,
        k_pe,
        topk,
        scale,
        topk_valid_prefix=topk_valid_prefix,
        causal_prefix_indices=causal_prefix_indices,
        topk_length=topk_length,
        causal_prefix_rows=causal_prefix_rows,
        stream=stream or mx.gpu,
    )


def q8_vup_flat(
    x: mx.array,
    unembed_out,
    *,
    key_length: Optional[int] = None,
    stream: Optional[mx.Stream] = None,
) -> Optional[mx.array]:
    """Project GLM sparse MLA latent output directly to [B, L, H * 256]."""

    env = os.environ.get("MLX_LM_GLM_DSA_Q8_VUP_FLAT", "auto").lower()
    if env in {"0", "false", "off", "no"}:
        return None
    if env in {"auto", ""} and (
        key_length is None or key_length < 32768 or key_length > 65536
    ):
        return None
    if env not in {"1", "true", "on", "yes", "auto", ""}:
        return None

    if (
        not hasattr(glm_fast, "glm_dsa_q8_vup_flat")
        or x.ndim != 4
        or x.shape[1] != 64
        or x.shape[-1] != 512
        or getattr(unembed_out, "bits", None) != 8
        or getattr(unembed_out, "group_size", None) != 64
        or getattr(unembed_out, "mode", None) != "affine"
        or not hasattr(unembed_out, "weight")
        or not hasattr(unembed_out, "scales")
    ):
        return None

    biases = unembed_out.get("biases") if hasattr(unembed_out, "get") else None
    if biases is None:
        return None
    weight = unembed_out["weight"]
    scales = unembed_out["scales"]
    if weight.shape != (64, 256, 128) or scales.shape != (64, 256, 8):
        return None
    return glm_fast.glm_dsa_q8_vup_flat(
        x,
        weight,
        scales,
        biases,
        stream=stream or mx.gpu,
    )
