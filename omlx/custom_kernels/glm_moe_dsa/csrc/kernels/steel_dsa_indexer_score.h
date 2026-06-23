// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace mlx::steel;

constant bool do_causal [[function_constant(300)]];
constant bool weights_lh [[function_constant(301)]];
constant bool bucketed_topk_output [[function_constant(302)]];
constant bool packed_block_table_output [[function_constant(303)]];
constant bool pair_head_indexer_score [[function_constant(304)]];
constant bool emit_high_histogram [[function_constant(305)]];

template <typename T>
METAL_FUNC uint dsa_ordered_key_16(T x) {
  const ushort bits = as_type<ushort>(x);
  return (bits & 0x8000) ? uint((~bits) & 0xffff) : uint(bits | 0x8000);
}

template <typename T, typename O, int TOPK, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void dsa_topk_indices_16bit(
    const device T* scores [[buffer(0)]],
    device O* out [[buffer(1)]],
    const constant DSATopKParams* params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(params->rows)) {
    return;
  }

  threadgroup atomic_uint hist[256];
  threadgroup atomic_uint counters[2];
  threadgroup uint state[4];

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  if (tid < 2) {
    atomic_store_explicit(&counters[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device T* row_scores = scores + size_t(row) * params->K;
  device O* row_out = out + size_t(row) * TOPK;

  int scan_limit = params->K;
  if (params->causal_valid_prefix) {
    const int q = int(row % uint(params->L));
    const int valid_length =
        metal::min(params->K, metal::max(0, params->K - params->L + q + 1));
    if (valid_length <= TOPK) {
      for (int i = int(tid); i < TOPK; i += THREADS) {
        row_out[i] = O(i < valid_length ? i : 0);
      }
      return;
    }
    scan_limit = valid_length;
  }

  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    atomic_fetch_add_explicit(&hist[key >> 8], 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = 0;
    uint threshold_hi = 0;
    for (int h = 255; h >= 0; --h) {
      const uint count = atomic_load_explicit(&hist[h], memory_order_relaxed);
      if (greater + count >= uint(TOPK)) {
        threshold_hi = uint(h);
        break;
      }
      greater += count;
    }
    state[0] = threshold_hi;
    state[1] = greater;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_hi = state[0];
  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    if ((key >> 8) == threshold_hi) {
      atomic_fetch_add_explicit(&hist[key & 0xff], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = state[1];
    uint threshold_lo = 0;
    for (int l = 255; l >= 0; --l) {
      const uint count = atomic_load_explicit(&hist[l], memory_order_relaxed);
      if (greater + count >= uint(TOPK)) {
        threshold_lo = uint(l);
        break;
      }
      greater += count;
    }
    const uint threshold_key = (threshold_hi << 8) | threshold_lo;
    state[2] = threshold_key;
    state[3] = greater;
    atomic_store_explicit(&counters[0], 0, memory_order_relaxed);
    atomic_store_explicit(&counters[1], greater, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_key = state[2];
  if (bucketed_topk_output) {
    for (int base = 0; base < scan_limit; base += THREADS) {
      const int i = base + int(tid);
      if (i < scan_limit) {
        const uint key = dsa_ordered_key_16(row_scores[i]);
        if (key > threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = O(i);
          }
        } else if (key == threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = O(i);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (int i = int(tid); i < scan_limit; i += THREADS) {
      const uint key = dsa_ordered_key_16(row_scores[i]);
      if (key > threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = O(i);
        }
      } else if (key == threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = O(i);
        }
      }
    }
  }
}

template <typename T, int TOPK, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
dsa_topk_indices_16bit_split_hist(
    const device T* scores [[buffer(0)]],
    device uint* out [[buffer(1)]],
    const constant DSATopKParams* params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(params->rows)) {
    return;
  }

  constexpr int SGS = THREADS / 32;
  constexpr int HIST_SGS = SGS > 16 ? 16 : SGS;
  threadgroup atomic_uint hist[HIST_SGS * 256];
  threadgroup atomic_uint counters[2];
  threadgroup uint state[4];

  for (int i = int(tid); i < HIST_SGS * 256; i += THREADS) {
    atomic_store_explicit(&hist[i], 0, memory_order_relaxed);
  }
  if (tid < 2) {
    atomic_store_explicit(&counters[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sg = (tid / 32) % HIST_SGS;
  const device T* row_scores = scores + size_t(row) * params->K;
  device uint* row_out = out + size_t(row) * TOPK;

  int scan_limit = params->K;
  if (params->causal_valid_prefix) {
    const int q = int(row % uint(params->L));
    const int valid_length =
        metal::min(params->K, metal::max(0, params->K - params->L + q + 1));
    if (valid_length <= TOPK) {
      for (int i = int(tid); i < TOPK; i += THREADS) {
        row_out[i] = uint(i < valid_length ? i : 0);
      }
      return;
    }
    scan_limit = valid_length;
  }

  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    atomic_fetch_add_explicit(
        &hist[sg * 256 + (key >> 8)], 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = 0;
    uint threshold_hi = 0;
    for (int h = 255; h >= 0; --h) {
      uint count = 0;
      STEEL_PRAGMA_UNROLL
      for (int g = 0; g < HIST_SGS; ++g) {
        count += atomic_load_explicit(&hist[g * 256 + h], memory_order_relaxed);
      }
      if (greater + count >= uint(TOPK)) {
        threshold_hi = uint(h);
        break;
      }
      greater += count;
    }
    state[0] = threshold_hi;
    state[1] = greater;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int i = int(tid); i < HIST_SGS * 256; i += THREADS) {
    atomic_store_explicit(&hist[i], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_hi = state[0];
  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    if ((key >> 8) == threshold_hi) {
      atomic_fetch_add_explicit(
          &hist[sg * 256 + (key & 0xff)], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = state[1];
    uint threshold_lo = 0;
    for (int l = 255; l >= 0; --l) {
      uint count = 0;
      STEEL_PRAGMA_UNROLL
      for (int g = 0; g < HIST_SGS; ++g) {
        count += atomic_load_explicit(&hist[g * 256 + l], memory_order_relaxed);
      }
      if (greater + count >= uint(TOPK)) {
        threshold_lo = uint(l);
        break;
      }
      greater += count;
    }
    const uint threshold_key = (threshold_hi << 8) | threshold_lo;
    state[2] = threshold_key;
    state[3] = greater;
    atomic_store_explicit(&counters[0], 0, memory_order_relaxed);
    atomic_store_explicit(&counters[1], greater, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_key = state[2];
  if (bucketed_topk_output) {
    for (int base = 0; base < scan_limit; base += THREADS) {
      const int i = base + int(tid);
      if (i < scan_limit) {
        const uint key = dsa_ordered_key_16(row_scores[i]);
        if (key > threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = uint(i);
          }
        } else if (key == threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = uint(i);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (int i = int(tid); i < scan_limit; i += THREADS) {
      const uint key = dsa_ordered_key_16(row_scores[i]);
      if (key > threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = uint(i);
        }
      } else if (key == threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = uint(i);
        }
      }
    }
  }
}

template <typename T, int TOPK, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
dsa_topk_indices_16bit_with_high_state(
    const device T* scores [[buffer(0)]],
    const device uint* high_state [[buffer(1)]],
    device uint* out [[buffer(2)]],
    const constant DSATopKParams* params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(params->rows)) {
    return;
  }

  threadgroup atomic_uint hist[256];
  threadgroup atomic_uint counters[2];
  threadgroup uint state[4];

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  if (tid < 2) {
    atomic_store_explicit(&counters[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device T* row_scores = scores + size_t(row) * params->K;
  device uint* row_out = out + size_t(row) * TOPK;

  int scan_limit = params->K;
  if (params->causal_valid_prefix) {
    const int q = int(row % uint(params->L));
    const int valid_length =
        metal::min(params->K, metal::max(0, params->K - params->L + q + 1));
    if (valid_length <= TOPK) {
      for (int i = int(tid); i < TOPK; i += THREADS) {
        row_out[i] = uint(i < valid_length ? i : 0);
      }
      return;
    }
    scan_limit = valid_length;
  }

  const uint threshold_hi = high_state[size_t(row) * 2];
  const uint greater_hi = high_state[size_t(row) * 2 + 1];
  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    if ((key >> 8) == threshold_hi) {
      atomic_fetch_add_explicit(&hist[key & 0xff], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = greater_hi;
    uint threshold_lo = 0;
    for (int l = 255; l >= 0; --l) {
      const uint count = atomic_load_explicit(&hist[l], memory_order_relaxed);
      if (greater + count >= uint(TOPK)) {
        threshold_lo = uint(l);
        break;
      }
      greater += count;
    }
    const uint threshold_key = (threshold_hi << 8) | threshold_lo;
    state[0] = threshold_key;
    state[1] = greater;
    atomic_store_explicit(&counters[0], 0, memory_order_relaxed);
    atomic_store_explicit(&counters[1], greater, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_key = state[0];
  if (bucketed_topk_output) {
    for (int base = 0; base < scan_limit; base += THREADS) {
      const int i = base + int(tid);
      if (i < scan_limit) {
        const uint key = dsa_ordered_key_16(row_scores[i]);
        if (key > threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = uint(i);
          }
        } else if (key == threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            row_out[pos] = uint(i);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (int i = int(tid); i < scan_limit; i += THREADS) {
      const uint key = dsa_ordered_key_16(row_scores[i]);
      if (key > threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = uint(i);
        }
      } else if (key == threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          row_out[pos] = uint(i);
        }
      }
    }
  }
}

template <typename T, int TOPK, int K_BLOCK, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
dsa_topk_block_table_from_scores_16bit(
    const device T* scores [[buffer(0)]],
    device uint* table [[buffer(1)]],
    const constant DSATopKParams* params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(params->rows)) {
    return;
  }

  constexpr uint empty = uint(-1);
  constexpr uint bit_mask =
      uint(K_BLOCK == 32 ? 0xffffffffull : ((1ull << K_BLOCK) - 1ull));
  threadgroup atomic_uint hist[256];
  threadgroup atomic_uint counters[2];
  threadgroup atomic_uint block_slots[TOPK];
  threadgroup atomic_uint bit_slots[TOPK];
  threadgroup atomic_uint out_count;
  threadgroup uint state[4];

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  for (uint slot = tid; slot < uint(TOPK); slot += THREADS) {
    atomic_store_explicit(&block_slots[slot], empty, memory_order_relaxed);
    atomic_store_explicit(&bit_slots[slot], 0, memory_order_relaxed);
  }
  if (tid < 2) {
    atomic_store_explicit(&counters[tid], 0, memory_order_relaxed);
  }
  if (tid == 0) {
    atomic_store_explicit(&out_count, 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device T* row_scores = scores + size_t(row) * params->K;
  device uint* row_table = table + size_t(row) * TOPK;

  int scan_limit = params->K;
  if (params->causal_valid_prefix) {
    const int q = int(row % uint(params->L));
    scan_limit =
        metal::min(params->K, metal::max(0, params->K - params->L + q + 1));
  }

  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    atomic_fetch_add_explicit(&hist[key >> 8], 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = 0;
    uint threshold_hi = 0;
    const uint target = uint(metal::min(TOPK, scan_limit));
    for (int h = 255; h >= 0; --h) {
      const uint count = atomic_load_explicit(&hist[h], memory_order_relaxed);
      if (greater + count >= target) {
        threshold_hi = uint(h);
        break;
      }
      greater += count;
    }
    state[0] = threshold_hi;
    state[1] = greater;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 256) {
    atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_hi = state[0];
  for (int i = int(tid); i < scan_limit; i += THREADS) {
    const uint key = dsa_ordered_key_16(row_scores[i]);
    if ((key >> 8) == threshold_hi) {
      atomic_fetch_add_explicit(&hist[key & 0xff], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint greater = state[1];
    uint threshold_lo = 0;
    const uint target = uint(metal::min(TOPK, scan_limit));
    for (int l = 255; l >= 0; --l) {
      const uint count = atomic_load_explicit(&hist[l], memory_order_relaxed);
      if (greater + count >= target) {
        threshold_lo = uint(l);
        break;
      }
      greater += count;
    }
    const uint threshold_key = (threshold_hi << 8) | threshold_lo;
    state[2] = threshold_key;
    state[3] = greater;
    atomic_store_explicit(&counters[0], 0, memory_order_relaxed);
    atomic_store_explicit(&counters[1], greater, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint threshold_key = state[2];
  const auto add_block = [&](uint k_pos) {
    const uint block = k_pos / uint(K_BLOCK);
    const uint bit = 1u << (k_pos - block * uint(K_BLOCK));
    const uint hash = (block * 2654435761u) % uint(TOPK);
    for (uint probe = 0; probe < uint(TOPK); ++probe) {
      const uint slot = (hash + probe) % uint(TOPK);
      uint expected = empty;
      if (atomic_compare_exchange_weak_explicit(
              &block_slots[slot],
              &expected,
              block,
              memory_order_relaxed,
              memory_order_relaxed) ||
          expected == block) {
        atomic_fetch_or_explicit(
            &bit_slots[slot], bit & bit_mask, memory_order_relaxed);
        break;
      }
    }
  };

  if (bucketed_topk_output) {
    for (int base = 0; base < scan_limit; base += THREADS) {
      const int i = base + int(tid);
      if (i < scan_limit) {
        const uint key = dsa_ordered_key_16(row_scores[i]);
        if (key > threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            add_block(uint(i));
          }
        } else if (key == threshold_key) {
          const uint pos =
              atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
          if (pos < uint(TOPK)) {
            add_block(uint(i));
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    for (int i = int(tid); i < scan_limit; i += THREADS) {
      const uint key = dsa_ordered_key_16(row_scores[i]);
      if (key > threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[0], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          add_block(uint(i));
        }
      } else if (key == threshold_key) {
        const uint pos =
            atomic_fetch_add_explicit(&counters[1], 1, memory_order_relaxed);
        if (pos < uint(TOPK)) {
          add_block(uint(i));
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint slot = tid; slot < uint(TOPK); slot += THREADS) {
    const uint block =
        atomic_load_explicit(&block_slots[slot], memory_order_relaxed);
    const uint bits =
        atomic_load_explicit(&bit_slots[slot], memory_order_relaxed);
    if (block != empty && bits != 0) {
      const uint out_slot =
          atomic_fetch_add_explicit(&out_count, 1, memory_order_relaxed);
      if (out_slot < uint(TOPK)) {
        row_table[out_slot] = uint((ulong(block) << K_BLOCK) | ulong(bits));
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint final_count = 0;
  if (tid == 0) {
    final_count = atomic_load_explicit(&out_count, memory_order_relaxed);
    state[0] = final_count;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  final_count = state[0];

  for (uint slot = final_count + tid; slot < uint(TOPK); slot += THREADS) {
    row_table[slot] = empty;
  }
}

template <int TOPK, int K_BLOCK, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
dsa_topk_to_block_table(
    const device uint* topk [[buffer(0)]],
    device uint* table [[buffer(1)]],
    const constant DSATopKBlockTableParams* params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= uint(params->rows)) {
    return;
  }

  constexpr uint empty = uint(-1);
  const uint table_base = packed_block_table_output ? row * uint(TOPK)
                                                    : row * uint(TOPK) * 2;
  threadgroup atomic_uint block_slots[TOPK];
  threadgroup atomic_uint bit_slots[TOPK];
  threadgroup atomic_uint out_count;
  threadgroup uint final_count;

  for (uint slot = tid; slot < uint(TOPK); slot += THREADS) {
    atomic_store_explicit(&block_slots[slot], empty, memory_order_relaxed);
    atomic_store_explicit(&bit_slots[slot], 0, memory_order_relaxed);
  }
  if (tid == 0) {
    atomic_store_explicit(&out_count, 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int q_pos = int(row % uint(params->L));
  const int q_abs = params->K - params->L + q_pos;
  const device uint* row_topk = topk + size_t(row) * TOPK;

  for (uint j = tid; j < uint(TOPK); j += THREADS) {
    const uint k_pos = row_topk[j];
    if (k_pos >= uint(params->K)) {
      continue;
    }
    if (params->causal && int(k_pos) > q_abs) {
      continue;
    }

    const uint block = k_pos / uint(K_BLOCK);
    const uint bit = 1u << (k_pos - block * uint(K_BLOCK));
    const uint hash = (block * 2654435761u) % uint(TOPK);

    for (uint probe = 0; probe < uint(TOPK); ++probe) {
      const uint slot = (hash + probe) % uint(TOPK);
      uint expected = empty;
      if (atomic_compare_exchange_weak_explicit(
              &block_slots[slot],
              &expected,
              block,
              memory_order_relaxed,
              memory_order_relaxed) ||
          expected == block) {
        atomic_fetch_or_explicit(&bit_slots[slot], bit, memory_order_relaxed);
        break;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint slot = tid; slot < uint(TOPK); slot += THREADS) {
    const uint block =
        atomic_load_explicit(&block_slots[slot], memory_order_relaxed);
    const uint bits =
        atomic_load_explicit(&bit_slots[slot], memory_order_relaxed);
    if (block != empty && bits != 0) {
      const uint out_slot =
          atomic_fetch_add_explicit(&out_count, 1, memory_order_relaxed);
      if (packed_block_table_output) {
        table[table_base + out_slot] =
            uint((ulong(block) << K_BLOCK) | ulong(bits));
      } else {
        table[table_base + out_slot * 2] = block;
        table[table_base + out_slot * 2 + 1] = bits;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    final_count = atomic_load_explicit(&out_count, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint slot = final_count + tid; slot < uint(TOPK); slot += THREADS) {
    if (packed_block_table_output) {
      table[table_base + slot] = empty;
    } else {
      table[table_base + slot * 2] = empty;
      table[table_base + slot * 2 + 1] = 0;
    }
  }
}

template <int TOPK, int PAGE_SIZE, int PAGE_BINS, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
dsa_topk_page_pack(
    const device uint* topk [[buffer(0)]],
    device uint* out [[buffer(1)]],
    const constant int& key_length [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  threadgroup atomic_uint hist[PAGE_BINS];
  threadgroup atomic_uint counters[PAGE_BINS];
  threadgroup uint offsets[PAGE_BINS];

  for (uint i = tid; i < uint(PAGE_BINS); i += THREADS) {
    atomic_store_explicit(&hist[i], 0, memory_order_relaxed);
    atomic_store_explicit(&counters[i], 0, memory_order_relaxed);
    offsets[i] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device uint* row_topk = topk + size_t(row) * TOPK;
  device uint* row_out = out + size_t(row) * TOPK;
  const uint max_page =
      uint(metal::min(PAGE_BINS - 1, (key_length + PAGE_SIZE - 1) / PAGE_SIZE));

  for (uint i = tid; i < uint(TOPK); i += THREADS) {
    const uint token = row_topk[i];
    const uint page = metal::min(token / uint(PAGE_SIZE), max_page);
    atomic_fetch_add_explicit(&hist[page], 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint offset = 0;
    for (uint page = 0; page < uint(PAGE_BINS); ++page) {
      offsets[page] = offset;
      offset += atomic_load_explicit(&hist[page], memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < uint(TOPK); i += THREADS) {
    const uint token = row_topk[i];
    const uint page = metal::min(token / uint(PAGE_SIZE), max_page);
    const uint pos =
        atomic_fetch_add_explicit(&counters[page], 1, memory_order_relaxed);
    row_out[offsets[page] + pos] = token;
  }
}

template <int TOPK, int Q_BLOCK, int CAPACITY, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
dsa_topk_qblock_union(
    const device uint* topk [[buffer(0)]],
    device atomic_uint* union_tokens [[buffer(1)]],
    device atomic_uint* row_bits [[buffer(2)]],
    device atomic_uint* lengths [[buffer(3)]],
    device atomic_uint* overflow [[buffer(4)]],
    const constant DSATopKQBlockUnionParams* params [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_id [[threadgroup_position_in_grid]]) {
  if (block_id >= uint(params->blocks)) {
    return;
  }

  constexpr uint empty = uint(-1);
  const int q_blocks = (params->L + Q_BLOCK - 1) / Q_BLOCK;
  const int b = int(block_id) / q_blocks;
  const int qb = int(block_id) - b * q_blocks;
  const int q_start = qb * Q_BLOCK;
  const size_t out_base = size_t(block_id) * CAPACITY;

  for (uint slot = tid; slot < uint(CAPACITY); slot += THREADS) {
    atomic_store_explicit(
        &union_tokens[out_base + slot], empty, memory_order_relaxed);
    atomic_store_explicit(
        &row_bits[out_base + slot], 0, memory_order_relaxed);
  }
  if (tid == 0) {
    atomic_store_explicit(&lengths[block_id], 0, memory_order_relaxed);
    atomic_store_explicit(&overflow[block_id], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool compact_prefix = params->causal_prefix_rows > 0 &&
      params->topk_rows + params->causal_prefix_rows == params->L &&
      params->causal_prefix_indices && params->topk_valid_prefix;

  for (int q_local = 0; q_local < Q_BLOCK; ++q_local) {
    const int q_pos = q_start + q_local;
    if (q_pos >= params->L) {
      continue;
    }
    const int q_abs = params->K - params->L + q_pos;
    const int valid_slots = params->topk_valid_prefix
        ? metal::min(params->topk, q_abs + 1)
        : params->topk;
    const bool implicit = params->causal_prefix_indices &&
        params->topk_valid_prefix &&
        ((compact_prefix && q_pos < params->causal_prefix_rows) ||
         q_abs < params->topk);
    const int topk_row = compact_prefix ? q_pos - params->causal_prefix_rows
                                        : q_pos;
    const bool has_topk_row = !implicit && topk_row >= 0 &&
        topk_row < params->topk_rows;
    const int safe_topk_row = has_topk_row ? topk_row : 0;
    const device uint* row_topk =
        topk + (size_t(b) * params->topk_rows + safe_topk_row) * TOPK;

    for (uint j = tid; j < uint(valid_slots); j += THREADS) {
      uint token = implicit ? j : (has_topk_row ? row_topk[j] : empty);
      if (token >= uint(params->K)) {
        continue;
      }
      if (params->causal && int(token) > q_abs) {
        continue;
      }

      const uint q_bit = 1u << uint(q_local);
      const uint hash = (token * 2654435761u) % uint(CAPACITY);
      bool inserted_or_found = false;
      for (uint probe = 0; probe < uint(CAPACITY); ++probe) {
        const uint pos = (hash + probe) % uint(CAPACITY);
        uint expected = empty;
        if (atomic_compare_exchange_weak_explicit(
                &union_tokens[out_base + pos],
                &expected,
                token,
                memory_order_relaxed,
                memory_order_relaxed)) {
          atomic_fetch_or_explicit(
              &row_bits[out_base + pos], q_bit, memory_order_relaxed);
          atomic_fetch_add_explicit(&lengths[block_id], 1, memory_order_relaxed);
          inserted_or_found = true;
          break;
        }
        if (expected == token) {
          atomic_fetch_or_explicit(
              &row_bits[out_base + pos], q_bit, memory_order_relaxed);
          inserted_or_found = true;
          break;
        }
      }
      if (!inserted_or_found) {
        atomic_store_explicit(&overflow[block_id], 1, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <int CAPACITY, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void
dsa_qblock_union_compact(
    const device atomic_uint* hash_tokens [[buffer(0)]],
    const device atomic_uint* hash_row_bits [[buffer(1)]],
    device uint* compact_tokens [[buffer(2)]],
    device uint* compact_row_bits [[buffer(3)]],
    device atomic_uint* lengths [[buffer(4)]],
    const constant DSATopKQBlockUnionParams* params [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_id [[threadgroup_position_in_grid]]) {
  if (block_id >= uint(params->blocks)) {
    return;
  }

  constexpr uint empty = uint(-1);
  const size_t base = size_t(block_id) * CAPACITY;

  if (tid == 0) {
    atomic_store_explicit(&lengths[block_id], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint slot = tid; slot < uint(CAPACITY); slot += THREADS) {
    const uint token =
        atomic_load_explicit(&hash_tokens[base + slot], memory_order_relaxed);
    const uint bits =
        atomic_load_explicit(&hash_row_bits[base + slot], memory_order_relaxed);
    if (token != empty && bits != 0) {
      const uint out_slot =
          atomic_fetch_add_explicit(&lengths[block_id], 1, memory_order_relaxed);
      if (out_slot < uint(CAPACITY)) {
        compact_tokens[base + out_slot] = token;
        compact_row_bits[base + out_slot] = bits;
      }
    }
  }
}

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
dsa_indexer_score_histogram(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* W [[buffer(2)]],
    device uint* Hist [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int& H [[buffer(5)]],
    const constant int& causal_q_offset [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      true,
      true,
      float>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;

  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  const int q_offset = causal_q_offset >= 0 ? causal_q_offset : N - M;

  if (do_causal) {
    const int row_limit = metal::min(c_row + BM, M);
    if (c_col > q_offset + row_limit - 1) {
      return;
    }
  }

  Q += size_t(tid.z) * H * M * D;
  K += size_t(tid.z) * N * D;
  W += size_t(tid.z) * H * M;
  Hist += size_t(tid.z) * M * 256 + size_t(c_row) * 256;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  float accum[decltype(mma_op.Ctile)::kElemsPerTile];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) {
    accum[i] = 0.0f;
  }

  if (pair_head_indexer_score) {
    thread mma_t mma_op_1(simd_group_id, simd_lane_id);
    for (int h = 0; h < H; h += 2) {
      mma_op.Ctile.clear();
      mma_op_1.Ctile.clear();

      const device T* A0 = Q + size_t(h) * M * D + size_t(c_row) * D;
      const device T* A1 = Q + size_t(h + 1) * M * D + size_t(c_row) * D;
      const device T* B = K + size_t(c_col) * D;

      thread loader_a_t loader_a0(
          A0, params->lda, As, simd_group_id, simd_lane_id);
      thread loader_a_t loader_a1(
          A1, params->lda, As, simd_group_id, simd_lane_id);
      thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

      for (int d = 0; d < params->gemm_k_iterations_aligned; ++d) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a0.load_unsafe();
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);
        loader_a0.next();

        if (h + 1 < H) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_a1.load_unsafe();

          threadgroup_barrier(mem_flags::mem_threadgroup);
          mma_op_1.mma(As, Bs);
          loader_a1.next();
        }
        loader_b.next();
      }

      threadgroup_barrier(mem_flags::mem_none);

      short ai = 0;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
        const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
        const float weight = weights_lh
            ? static_cast<float>(W[size_t(row) * H + h])
            : static_cast<float>(W[size_t(h) * M + row]);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
          thread const auto& frag = mma_op.Ctile.frag_at(i, j);
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
            accum[ai++] += max(frag[e], 0.0f) * weight;
          }
        }
      }

      if (h + 1 < H) {
        ai = 0;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < decltype(mma_op_1.Ctile)::kTileRows; ++i) {
          const int row = c_row + mma_op_1.sm + i * mma_t::TM_stride;
          const float weight = weights_lh
              ? static_cast<float>(W[size_t(row) * H + h + 1])
              : static_cast<float>(W[size_t(h + 1) * M + row]);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < decltype(mma_op_1.Ctile)::kTileCols; ++j) {
            thread const auto& frag = mma_op_1.Ctile.frag_at(i, j);
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < decltype(mma_op_1.Ctile)::kElemsPerFrag; ++e) {
              accum[ai++] += max(frag[e], 0.0f) * weight;
            }
          }
        }
      }
    }
  } else {
    for (int h = 0; h < H; ++h) {
      mma_op.Ctile.clear();

      const device T* A = Q + size_t(h) * M * D + size_t(c_row) * D;
      const device T* B = K + size_t(c_col) * D;

      thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
      thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

      for (int d = 0; d < params->gemm_k_iterations_aligned; ++d) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(As, Bs);

        loader_a.next();
        loader_b.next();
      }

      threadgroup_barrier(mem_flags::mem_none);

      short ai = 0;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
        const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
        const float weight = weights_lh
            ? static_cast<float>(W[size_t(row) * H + h])
            : static_cast<float>(W[size_t(h) * M + row]);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
          thread const auto& frag = mma_op.Ctile.frag_at(i, j);
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
            accum[ai++] += max(frag[e], 0.0f) * weight;
          }
        }
      }
    }
  }

  device atomic_uint* atomic_hist = reinterpret_cast<device atomic_uint*>(Hist);
  short ai = 0;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
    const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
      const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
        const int col = col_base + e;
        if (row < M && col < N && (!do_causal || col <= q_offset + row)) {
          const uint key = dsa_ordered_key_16(static_cast<T>(accum[ai]));
          atomic_fetch_add_explicit(
              &atomic_hist[size_t(row - c_row) * 256 + (key >> 8)],
              1,
              memory_order_relaxed);
        }
        ai++;
      }
    }
  }
}

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
dsa_indexer_score_low_histogram(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* W [[buffer(2)]],
    const device uint* ThresholdHi [[buffer(3)]],
    device uint* Hist [[buffer(4)]],
    const constant GEMMParams* params [[buffer(5)]],
    const constant int& H [[buffer(6)]],
    const constant int& causal_q_offset [[buffer(7)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      true,
      true,
      float>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;

  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  const int q_offset = causal_q_offset >= 0 ? causal_q_offset : N - M;

  if (do_causal) {
    const int row_limit = metal::min(c_row + BM, M);
    if (c_col > q_offset + row_limit - 1) {
      return;
    }
  }

  Q += size_t(tid.z) * H * M * D;
  K += size_t(tid.z) * N * D;
  W += size_t(tid.z) * H * M;
  ThresholdHi += size_t(tid.z) * M * 2 + size_t(c_row) * 2;
  Hist += size_t(tid.z) * M * 256 + size_t(c_row) * 256;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  float accum[decltype(mma_op.Ctile)::kElemsPerTile];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) {
    accum[i] = 0.0f;
  }

  for (int h = 0; h < H; ++h) {
    mma_op.Ctile.clear();

    const device T* A = Q + size_t(h) * M * D + size_t(c_row) * D;
    const device T* B = K + size_t(c_col) * D;

    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    for (int d = 0; d < params->gemm_k_iterations_aligned; ++d) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    short ai = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
      const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
      const float weight = weights_lh
          ? static_cast<float>(W[size_t(row) * H + h])
          : static_cast<float>(W[size_t(h) * M + row]);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
        thread const auto& frag = mma_op.Ctile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
          accum[ai++] += max(frag[e], 0.0f) * weight;
        }
      }
    }
  }

  device atomic_uint* atomic_hist = reinterpret_cast<device atomic_uint*>(Hist);
  short ai = 0;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
    const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
      const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
        const int col = col_base + e;
        if (row < M && col < N && (!do_causal || col <= q_offset + row)) {
          const uint threshold_hi = ThresholdHi[size_t(row - c_row) * 2];
          const uint key = dsa_ordered_key_16(static_cast<T>(accum[ai]));
          if ((key >> 8) == threshold_hi) {
            atomic_fetch_add_explicit(
                &atomic_hist[size_t(row - c_row) * 256 + (key & 0xff)],
                1,
                memory_order_relaxed);
          }
        }
        ai++;
      }
    }
  }
}

template <typename T, int TOPK, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
dsa_indexer_topk_emit(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* W [[buffer(2)]],
    const device uint* Threshold [[buffer(3)]],
    device uint* O [[buffer(4)]],
    device uint* Counters [[buffer(5)]],
    const constant GEMMParams* params [[buffer(6)]],
    const constant int& H [[buffer(7)]],
    const constant int& causal_q_offset [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      true,
      true,
      float>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;

  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  const int q_offset = causal_q_offset >= 0 ? causal_q_offset : N - M;

  if (do_causal) {
    const int row_limit = metal::min(c_row + BM, M);
    if (c_col > q_offset + row_limit - 1) {
      return;
    }
  }

  Q += size_t(tid.z) * H * M * D;
  K += size_t(tid.z) * N * D;
  W += size_t(tid.z) * H * M;
  Threshold += size_t(tid.z) * M * 2 + size_t(c_row) * 2;
  O += size_t(tid.z) * M * TOPK + size_t(c_row) * TOPK;
  Counters += size_t(tid.z) * M * 2 + size_t(c_row) * 2;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  float accum[decltype(mma_op.Ctile)::kElemsPerTile];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) {
    accum[i] = 0.0f;
  }

  for (int h = 0; h < H; ++h) {
    mma_op.Ctile.clear();

    const device T* A = Q + size_t(h) * M * D + size_t(c_row) * D;
    const device T* B = K + size_t(c_col) * D;

    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    for (int d = 0; d < params->gemm_k_iterations_aligned; ++d) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    short ai = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
      const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
      const float weight = weights_lh
          ? static_cast<float>(W[size_t(row) * H + h])
          : static_cast<float>(W[size_t(h) * M + row]);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
        thread const auto& frag = mma_op.Ctile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
          accum[ai++] += max(frag[e], 0.0f) * weight;
        }
      }
    }
  }

  device atomic_uint* atomic_counters =
      reinterpret_cast<device atomic_uint*>(Counters);
  short ai = 0;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
    const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
      const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
        const int col = col_base + e;
        if (row < M && col < N && (!do_causal || col <= q_offset + row)) {
          const uint threshold_key = Threshold[size_t(row - c_row) * 2];
          const uint greater_count = Threshold[size_t(row - c_row) * 2 + 1];
          device uint* row_out = O + size_t(row - c_row) * TOPK;
          device atomic_uint* row_counters =
              atomic_counters + size_t(row - c_row) * 2;
          const uint key = dsa_ordered_key_16(static_cast<T>(accum[ai]));
          if (key > threshold_key) {
            const uint pos = atomic_fetch_add_explicit(
                &row_counters[0], 1, memory_order_relaxed);
            if (pos < uint(TOPK)) {
              row_out[pos] = uint(col);
            }
          } else if (key == threshold_key) {
            const uint pos = greater_count +
                atomic_fetch_add_explicit(
                    &row_counters[1], 1, memory_order_relaxed);
            if (pos < uint(TOPK)) {
              row_out[pos] = uint(col);
            }
          }
        }
        ai++;
      }
    }
  }
}

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
dsa_indexer_score(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* W [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int& H [[buffer(5)]],
    const constant int& unused_causal_prefix_topk [[buffer(6)]],
    const constant bool& skip_causal_future_store [[buffer(7)]],
    const constant int& causal_q_offset [[buffer(8)]],
    device uint* Hist [[buffer(9), function_constant(emit_high_histogram)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      true,
      true,
      float>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;

  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  const int q_offset = causal_q_offset >= 0 ? causal_q_offset : N - M;
  constexpr int THREADS = WM * WN * 32;
  const int thread_idx = int(simd_group_id) * 32 + int(simd_lane_id);

  if (do_causal) {
    const int row_limit = metal::min(c_row + BM, M);
    if (c_col > q_offset + row_limit - 1) {
      if (skip_causal_future_store) {
        return;
      }
      device T* Dst = O + size_t(tid.z) * M * N + size_t(c_row) * params->ldd +
          c_col;
      device atomic_uint* atomic_hist = nullptr;
      if (emit_high_histogram) {
        atomic_hist = reinterpret_cast<device atomic_uint*>(
            Hist + size_t(tid.z) * M * 256 + size_t(c_row) * 256);
      }
      const uint neg_inf_hi = dsa_ordered_key_16(static_cast<T>(-INFINITY)) >> 8;
      for (int e = thread_idx; e < BM * BN; e += THREADS) {
        const int row = e / BN;
        const int col = e - row * BN;
        if (c_row + row < M && c_col + col < N) {
          Dst[size_t(row) * params->ldd + col] = static_cast<T>(-INFINITY);
          if (emit_high_histogram) {
            atomic_fetch_add_explicit(
                &atomic_hist[size_t(row) * 256 + neg_inf_hi],
                1,
                memory_order_relaxed);
          }
        }
      }
      return;
    }
  }

  if (do_causal && unused_causal_prefix_topk > 0) {
    const int row_limit = metal::min(c_row + BM, M);
    if (q_offset + row_limit <= unused_causal_prefix_topk) {
      return;
    }
  }

  Q += size_t(tid.z) * H * M * D;
  K += size_t(tid.z) * N * D;
  W += size_t(tid.z) * H * M;
  O += size_t(tid.z) * M * N + size_t(c_row) * params->ldd + c_col;
  if (emit_high_histogram) {
    Hist += size_t(tid.z) * M * 256 + size_t(c_row) * 256;
  }

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  float accum[decltype(mma_op.Ctile)::kElemsPerTile];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) {
    accum[i] = 0.0f;
  }

  for (int h = 0; h < H; ++h) {
    mma_op.Ctile.clear();

    const device T* A = Q + size_t(h) * M * D + size_t(c_row) * D;
    const device T* B = K + size_t(c_col) * D;

    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    for (int d = 0; d < params->gemm_k_iterations_aligned; ++d) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    short ai = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
      const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
      const float weight = weights_lh
          ? static_cast<float>(W[size_t(row) * H + h])
          : static_cast<float>(W[size_t(h) * M + row]);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
        thread const auto& frag = mma_op.Ctile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
          accum[ai++] += max(frag[e], 0.0f) * weight;
        }
      }
    }
  }

  device T* Dst = O + size_t(mma_op.sm) * params->ldd + mma_op.sn;
  device atomic_uint* atomic_hist = nullptr;
  if (emit_high_histogram) {
    atomic_hist = reinterpret_cast<device atomic_uint*>(Hist);
  }
  short ai = 0;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
    const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
      const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
      const int out_base =
          (i * decltype(mma_op.Ctile)::kFragRows) * WM * params->ldd +
          (j * decltype(mma_op.Ctile)::kFragCols) * WN;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
        const int col = col_base + e;
        const bool valid = row < M && col < N;
        const bool future = do_causal && col > q_offset + row;
        const T value = future ? static_cast<T>(-INFINITY)
                               : static_cast<T>(accum[ai]);
        Dst[out_base + e] = value;
        if (emit_high_histogram && valid &&
            (!skip_causal_future_store || !future)) {
          const uint key = dsa_ordered_key_16(value);
          atomic_fetch_add_explicit(
              &atomic_hist[size_t(row - c_row) * 256 + (key >> 8)],
              1,
              memory_order_relaxed);
        }
        ai++;
      }
    }
  }
}

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
dsa_indexer_block_score(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* W [[buffer(2)]],
    device float* O [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int& H [[buffer(5)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;

  constexpr int Q_BLOCK = 32;
  constexpr int K_BLOCK = 16;
  constexpr int SUB_M = BM / Q_BLOCK;
  constexpr int SUB_N = BN / K_BLOCK;
  constexpr int SUB_BLOCKS = SUB_M * SUB_N;
  constexpr int THREADS = WM * WN * 32;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      true,
      true,
      float>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;

  const int M = params->M;
  const int N = params->N;
  const int D = params->K;
  const int q_blocks = (M + Q_BLOCK - 1) / Q_BLOCK;
  const int k_blocks = (N + K_BLOCK - 1) / K_BLOCK;

  Q += size_t(tid.z) * H * M * D;
  K += size_t(tid.z) * N * D;
  W += size_t(tid.z) * H * M;
  O += size_t(tid.z) * q_blocks * k_blocks;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
  threadgroup float partial[SUB_BLOCKS * THREADS];

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  float accum[decltype(mma_op.Ctile)::kElemsPerTile];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; ++i) {
    accum[i] = 0.0f;
  }

  for (int h = 0; h < H; ++h) {
    mma_op.Ctile.clear();

    const device T* A = Q + size_t(h) * M * D + size_t(c_row) * D;
    const device T* B = K + size_t(c_col) * D;

    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    for (int d = 0; d < params->gemm_k_iterations_aligned; ++d) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    short ai = 0;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
      const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
      const float weight = weights_lh
          ? static_cast<float>(W[size_t(row) * H + h])
          : static_cast<float>(W[size_t(h) * M + row]);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
        thread const auto& frag = mma_op.Ctile.frag_at(i, j);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
          accum[ai++] += max(frag[e], 0.0f) * weight;
        }
      }
    }
  }

  float local_max[SUB_BLOCKS];
  const int thread_idx = simd_group_id * 32 + simd_lane_id;
  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < SUB_BLOCKS; ++s) {
    local_max[s] = -INFINITY;
  }

  const int offset = N - M;
  short ai = 0;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(mma_op.Ctile)::kTileRows; ++i) {
    const int row = c_row + mma_op.sm + i * mma_t::TM_stride;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(mma_op.Ctile)::kTileCols; ++j) {
      const int col_base = c_col + mma_op.sn + j * mma_t::TN_stride;
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < decltype(mma_op.Ctile)::kElemsPerFrag; ++e) {
        const int col = col_base + e;
        if (row < M && col < N && (!do_causal || col <= offset + row)) {
          const int sub_m = (row - c_row) / Q_BLOCK;
          const int sub_n = (col - c_col) / K_BLOCK;
          const int sub_idx = sub_m * SUB_N + sub_n;
          local_max[sub_idx] = max(local_max[sub_idx], accum[ai]);
        }
        ai++;
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < SUB_BLOCKS; ++s) {
    partial[s * THREADS + thread_idx] = local_max[s];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (thread_idx < SUB_BLOCKS) {
    float block_max = -INFINITY;
    for (short t = 0; t < THREADS; ++t) {
      block_max = max(block_max, partial[thread_idx * THREADS + t]);
    }
    const int out_q_block = c_row / Q_BLOCK + thread_idx / SUB_N;
    const int out_k_block = c_col / K_BLOCK + thread_idx % SUB_N;
    if (out_q_block < q_blocks && out_k_block < k_blocks) {
      O[out_q_block * k_blocks + out_k_block] = block_max;
    }
  }
}
