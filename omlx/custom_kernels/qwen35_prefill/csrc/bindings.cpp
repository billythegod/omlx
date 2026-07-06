#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "qwen35_prefill.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native Qwen3.5/3.6 prefill kernels for oMLX";

  m.def(
      "qwen35_fa256_attention",
      &omlx::qwen35_prefill_kernels::qwen35_fa256_attention,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "causal"_a = true,
      "q_block"_a = 32,
      "k_block"_a = 8,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q4_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q4_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q5_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q5_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q6_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q6_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q8_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q8_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "stream"_a = nb::none());
  m.def(
      "qwen35_moe_weighted_sum",
      &omlx::qwen35_prefill_kernels::qwen35_moe_weighted_sum,
      "x_sorted"_a,
      "inv_order"_a,
      "scores"_a,
      "stream"_a = nb::none());
}
