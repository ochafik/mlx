#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

using namespace metal;

// SDPA vector instantiations
#define instantiate_sdpa_vector_aggregation(type, value_dim) \
  instantiate_kernel(                                        \
      "sdpa_vector_2pass_2_" #type "_" #value_dim,           \
      sdpa_vector_2pass_2,                                   \
      type,                                                  \
      value_dim)

#define instantiate_sdpa_vector(type, qk_dim, value_dim)       \
  instantiate_kernel(                                          \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,         \
      sdpa_vector,                                             \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1,                                     \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)

#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 256, 256)        \
  instantiate_sdpa_vector_aggregation(type, 64)  \
  instantiate_sdpa_vector_aggregation(type, 96)  \
  instantiate_sdpa_vector_aggregation(type, 128) \
  instantiate_sdpa_vector_aggregation(type, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)

// Affine quantized SDPA vector instantiations
#define instantiate_quant_sdpa_vector(type, head_dim, group_size, bits) \
  instantiate_kernel(                                                   \
    "quant_sdpa_vector_2pass_1_" #type "_" #head_dim "_" #group_size "_" #bits, \
    quant_sdpa_vector_2pass_1, type, head_dim, group_size, bits)

#define instantiate_quant_sdpa_vector_bits(type, heads, group_size) \
  instantiate_quant_sdpa_vector(type, heads, group_size, 4)         \
  instantiate_quant_sdpa_vector(type, heads, group_size, 8)

#define instantiate_quant_sdpa_vector_group_size(type, heads) \
  instantiate_quant_sdpa_vector_bits(type, heads, 32)         \
  instantiate_quant_sdpa_vector_bits(type, heads, 64)         \
  instantiate_quant_sdpa_vector_bits(type, heads, 128)

#define instantiate_quant_sdpa_vector_heads(type) \
  instantiate_quant_sdpa_vector_group_size(type, 64)  \
  instantiate_quant_sdpa_vector_group_size(type, 128)

instantiate_quant_sdpa_vector_heads(float)
instantiate_quant_sdpa_vector_heads(bfloat16_t)
instantiate_quant_sdpa_vector_heads(float16_t)

// Centroid LUT SDPA vector instantiations
#define instantiate_lut_sdpa_vector(type, head_dim, bits) \
  instantiate_kernel(                                      \
    "lut_sdpa_vector_2pass_1_" #type "_" #head_dim "_" #bits, \
    lut_sdpa_vector_2pass_1, type, head_dim, bits)

#define instantiate_lut_sdpa_vector_bits(type, head_dim) \
  instantiate_lut_sdpa_vector(type, head_dim, 3)          \
  instantiate_lut_sdpa_vector(type, head_dim, 4)          \
  instantiate_lut_sdpa_vector(type, head_dim, 8)

#define instantiate_lut_sdpa_vector_heads(type) \
  instantiate_lut_sdpa_vector_bits(type, 64)     \
  instantiate_lut_sdpa_vector_bits(type, 128)

instantiate_lut_sdpa_vector_heads(float)
instantiate_lut_sdpa_vector_heads(bfloat16_t)
instantiate_lut_sdpa_vector_heads(float16_t)

    // clang-format on
