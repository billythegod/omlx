#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::qwen35_prefill_kernels {

// Mirror of mlx::core::metal::is_nax_available() (not exported from libmlx):
// macOS >= 26.2 and an applegpu generation with tensor units (gen >= 17, or
// >= 18 for 'p'-suffix parts).
bool is_nax_available();

// True when the NAX metallib was built next to the extension. Kernel launch
// still degrades to the classic kernels if loading it fails at runtime.
bool nax_qmm_kernels_built();

// False once a NAX kernel launch failed and the op fell back to the classic
// kernels for the rest of the process (diagnostics for the M5 sweep).
bool nax_qmm_runtime_active();

mx::array qwen35_fa256_attention(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    float scale,
    bool causal = true,
    int q_block = 32,
    int k_block = 8,
    mx::StreamOrDevice s = {});

mx::array qwen35_q4_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    mx::StreamOrDevice s = {});

mx::array qwen35_q5_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    mx::StreamOrDevice s = {});

mx::array qwen35_q6_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    mx::StreamOrDevice s = {});

mx::array qwen35_q8_affine_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    mx::StreamOrDevice s = {});

mx::array qwen35_moe_weighted_sum(
    const mx::array& x_sorted,
    const mx::array& inv_order,
    const mx::array& scores,
    mx::StreamOrDevice s = {});

} // namespace omlx::qwen35_prefill_kernels
