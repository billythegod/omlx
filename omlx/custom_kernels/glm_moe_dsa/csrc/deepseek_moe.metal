#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void deepseek_mxfp4_gather_blocks_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device uint8_t* scales [[buffer(2)]],
    const device int32_t* block_meta [[buffer(3)]],
    const device int32_t* block_count [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& max_blocks [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& K [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)M;
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int block_id = int(tid.y);
  const int nblocks = block_count[0];
  if (block_id >= max_blocks || block_id >= nblocks) {
    return;
  }

  const int y_col = int(tid.x) * BN;
  if (y_col >= N) {
    return;
  }

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) {
    return;
  }

  const short tgp_bm = short(min(BM, rows));
  const short tgp_bn = short(min(BN, N - y_col));
  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;

  const device T* xl = x + size_t(row_start) * K;
  device T* yl = y + size_t(row_start) * N + y_col;
  const device uint8_t* wl =
      ((const device uint8_t*)w) + size_t(expert) * stride_w +
      size_t(y_col) * K_w;
  const device uint8_t* sl =
      scales + size_t(expert) * stride_s + size_t(y_col) * K_g;

  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl, sl, K, Ws, simd_group_id, simd_lane_id);

  if (rows == BM && tgp_bn == BN) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(yl, N);
  } else if (tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(BN, tgp_bm));
  } else if (rows == BM) {
    gemm_loop_unaligned<true, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, BM));
  } else {
    gemm_loop_unaligned<false, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, tgp_bm));
  }
}

#define instantiate_deepseek_mxfp4_blocks(type, bm, bn, bk, wm, wn)            \
  instantiate_kernel(                                                          \
      "deepseek_mxfp4_gather_blocks_rhs_" #type "_bm_" #bm "_bn_" #bn         \
      "_bk_" #bk "_wm_" #wm "_wn_" #wn,                                       \
      deepseek_mxfp4_gather_blocks_rhs,                                        \
      type,                                                                    \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn)

instantiate_deepseek_mxfp4_blocks(float16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(float16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(float16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(float16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(float16_t, 32, 64, 32, 1, 2);

instantiate_deepseek_mxfp4_blocks(bfloat16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(bfloat16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(bfloat16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(bfloat16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(bfloat16_t, 32, 64, 32, 1, 2);

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void deepseek_mxfp4_gather_pair_blocks_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w0 [[buffer(1)]],
    const device uint8_t* scales0 [[buffer(2)]],
    const device uint32_t* w1 [[buffer(3)]],
    const device uint8_t* scales1 [[buffer(4)]],
    const device int32_t* block_meta [[buffer(5)]],
    const device int32_t* block_count [[buffer(6)]],
    device T* y [[buffer(7)]],
    const constant int& max_blocks [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& N [[buffer(10)]],
    const constant int& K [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int pair_id = int(tid.z);
  const int block_id = int(tid.y);
  const int nblocks = block_count[0];
  if (pair_id >= 2 || block_id >= max_blocks || block_id >= nblocks) {
    return;
  }

  const int y_col = int(tid.x) * BN;
  if (y_col >= N) {
    return;
  }

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) {
    return;
  }

  const short tgp_bm = short(min(BM, rows));
  const short tgp_bn = short(min(BN, N - y_col));
  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;

  const device uint32_t* w = pair_id == 0 ? w0 : w1;
  const device uint8_t* scales = pair_id == 0 ? scales0 : scales1;

  const device T* xl = x + size_t(row_start) * K;
  device T* yl = y + size_t(pair_id) * size_t(M) * N +
      size_t(row_start) * N + y_col;
  const device uint8_t* wl =
      ((const device uint8_t*)w) + size_t(expert) * stride_w +
      size_t(y_col) * K_w;
  const device uint8_t* sl =
      scales + size_t(expert) * stride_s + size_t(y_col) * K_g;

  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl, sl, K, Ws, simd_group_id, simd_lane_id);

  if (rows == BM && tgp_bn == BN) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(yl, N);
  } else if (tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(BN, tgp_bm));
  } else if (rows == BM) {
    gemm_loop_unaligned<true, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, BM));
  } else {
    gemm_loop_unaligned<false, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, tgp_bm));
  }
}

#define instantiate_deepseek_mxfp4_pair_blocks(type, bm, bn, bk, wm, wn)       \
  instantiate_kernel(                                                          \
      "deepseek_mxfp4_gather_pair_blocks_rhs_" #type "_bm_" #bm "_bn_" #bn    \
      "_bk_" #bk "_wm_" #wm "_wn_" #wn,                                       \
      deepseek_mxfp4_gather_pair_blocks_rhs,                                   \
      type,                                                                    \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn)

instantiate_deepseek_mxfp4_pair_blocks(float16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(float16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(float16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(float16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(float16_t, 32, 64, 32, 1, 2);

instantiate_deepseek_mxfp4_pair_blocks(bfloat16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(bfloat16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(bfloat16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(bfloat16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(bfloat16_t, 32, 64, 32, 1, 2);

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void deepseek_mxfp4_gather_pair_concat_blocks_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w0 [[buffer(1)]],
    const device uint8_t* scales0 [[buffer(2)]],
    const device uint32_t* w1 [[buffer(3)]],
    const device uint8_t* scales1 [[buffer(4)]],
    const device int32_t* block_meta [[buffer(5)]],
    const device int32_t* block_count [[buffer(6)]],
    device T* y [[buffer(7)]],
    const constant int& max_blocks [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& N [[buffer(10)]],
    const constant int& K [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)M;
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int block_id = int(tid.y);
  const int nblocks = block_count[0];
  if (block_id >= max_blocks || block_id >= nblocks) {
    return;
  }

  const int y_col_concat = int(tid.x) * BN;
  if (y_col_concat >= 2 * N) {
    return;
  }

  const int pair_id = y_col_concat >= N ? 1 : 0;
  const int y_col = pair_id == 0 ? y_col_concat : y_col_concat - N;
  if (y_col >= N) {
    return;
  }

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) {
    return;
  }

  const short tgp_bm = short(min(BM, rows));
  const short tgp_bn = short(min(BN, N - y_col));
  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const int N_out = 2 * N;

  const device uint32_t* w = pair_id == 0 ? w0 : w1;
  const device uint8_t* scales = pair_id == 0 ? scales0 : scales1;

  const device T* xl = x + size_t(row_start) * K;
  device T* yl = y + size_t(row_start) * N_out +
      size_t(pair_id) * N + y_col;
  const device uint8_t* wl =
      ((const device uint8_t*)w) + size_t(expert) * stride_w +
      size_t(y_col) * K_w;
  const device uint8_t* sl =
      scales + size_t(expert) * stride_s + size_t(y_col) * K_g;

  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl, sl, K, Ws, simd_group_id, simd_lane_id);

  if (rows == BM && tgp_bn == BN) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(yl, N_out);
  } else if (tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N_out, short2(0, 0), short2(BN, tgp_bm));
  } else if (rows == BM) {
    gemm_loop_unaligned<true, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N_out, short2(0, 0), short2(tgp_bn, BM));
  } else {
    gemm_loop_unaligned<false, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N_out, short2(0, 0), short2(tgp_bn, tgp_bm));
  }
}

#define instantiate_deepseek_mxfp4_pair_concat_blocks(type, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                          \
      "deepseek_mxfp4_gather_pair_concat_blocks_rhs_" #type "_bm_" #bm         \
      "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn,                           \
      deepseek_mxfp4_gather_pair_concat_blocks_rhs,                            \
      type,                                                                    \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn)

instantiate_deepseek_mxfp4_pair_concat_blocks(float16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(float16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(float16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(float16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(float16_t, 32, 64, 32, 1, 2);

instantiate_deepseek_mxfp4_pair_concat_blocks(bfloat16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(bfloat16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(bfloat16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(bfloat16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(bfloat16_t, 32, 64, 32, 1, 2);

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void deepseek_mxfp4_gather_expert_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device uint8_t* scales [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& M [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& E [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int y_col = int(tid.x) * BN;
  const int expert = int(tid.y);
  if (y_col >= N || expert >= E) {
    return;
  }

  int lo = 0;
  int hi = M;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (indices[mid] < uint32_t(expert)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int start = lo;

  hi = M;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (indices[mid] <= uint32_t(expert)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int end = lo;
  if (start >= end) {
    return;
  }

  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const short tgp_bn = short(min(BN, N - y_col));
  const short2 tile_w = short2(k_remain, tgp_bn);

  const device uint8_t* wl =
      ((const device uint8_t*)w) + size_t(expert) * stride_w +
      size_t(y_col) * K_w;
  const device uint8_t* sl =
      scales + size_t(expert) * stride_s + size_t(y_col) * K_g;

  for (int row = start; row < end; row += BM) {
    const int rows = min(BM, end - row);
    const short tgp_bm = short(rows);
    const short2 tile_x = short2(k_remain, tgp_bm);

    const device T* xl = x + size_t(row) * K;
    device T* yl = y + size_t(row) * N + y_col;

    thread mma_t mma_op(simd_group_id, simd_lane_id);
    thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(wl, sl, K, Ws, simd_group_id, simd_lane_id);

    if (rows == BM && tgp_bn == BN) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (k_remain != 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result(yl, N);
    } else if (tgp_bn == BN) {
      gemm_loop_unaligned<false, true, true>(
          Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
      if (k_remain != 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result_slice(yl, N, short2(0, 0), short2(BN, tgp_bm));
    } else if (rows == BM) {
      gemm_loop_unaligned<true, false, true>(
          Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
      if (k_remain != 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, BM));
    } else {
      gemm_loop_unaligned<false, false, true>(
          Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
      if (k_remain != 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, tgp_bm));
    }
  }
}

#define instantiate_deepseek_mxfp4_expert(type, bm, bn, bk, wm, wn)            \
  instantiate_kernel(                                                          \
      "deepseek_mxfp4_gather_expert_rhs_" #type "_bm_" #bm "_bn_" #bn         \
      "_bk_" #bk "_wm_" #wm "_wn_" #wn,                                       \
      deepseek_mxfp4_gather_expert_rhs,                                        \
      type,                                                                    \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn)

instantiate_deepseek_mxfp4_expert(float16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(float16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(float16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(float16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(float16_t, 32, 64, 32, 1, 2);

instantiate_deepseek_mxfp4_expert(bfloat16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(bfloat16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(bfloat16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(bfloat16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(bfloat16_t, 32, 64, 32, 1, 2);
