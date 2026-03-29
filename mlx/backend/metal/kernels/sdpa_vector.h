// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

constant bool has_mask [[function_constant(20)]];
constant bool query_transposed [[function_constant(21)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];
constant bool has_sinks [[function_constant(25)]];
constant int blocks [[function_constant(26)]];

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device bool* bmask [[buffer(11), function_constant(bool_mask)]],
    const device T* fmask [[buffer(12), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(13), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(14), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(15), function_constant(has_mask)]],
    const device T* sinks [[buffer(16), function_constant(has_sinks)]],
    const constant int& num_q_heads
    [[buffer(17), function_constant(has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const device T* sinks [[buffer(18), function_constant(has_sinks)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Compute the i-th score
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[i] * keys[i];
      }
      score = simd_sum(score);

      if (float_mask) {
        score += fmask[0];
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * values[i];
      }
    }

    // Move the pointers to the next kv
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
    if (bool_mask) {
      bmask += blocks * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += blocks * mask_kv_seq_stride;
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// ============================================================================
// Affine quantized SDPA helpers (ported from q-sdpa)
// ============================================================================

template <typename T, typename U, int elem_per_thread, int bits>
METAL_FUNC U affine_load_queries(const device T* queries, thread U* q, U scale) {
  U query_sum = 0;
  if (bits == 4) {
    for (int i = 0; i < elem_per_thread; i += 4) {
      q[i] = scale * queries[i];
      q[i + 1] = scale * queries[i + 1];
      q[i + 2] = scale * queries[i + 2];
      q[i + 3] = scale * queries[i + 3];
      query_sum += q[i] + q[i + 1] + q[i + 2] + q[i + 3];
      q[i + 1] /= 16.0f;
      q[i + 2] /= 256.0f;
      q[i + 3] /= 4096.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < elem_per_thread; i++) {
      q[i] = scale * queries[i];
      query_sum += q[i];
    }
  }
  return query_sum;
}

template <typename U, int elem_per_thread, int bits>
METAL_FUNC void affine_load_keys(const device uint32_t* keys, thread U* k) {
  if (bits == 4) {
    auto ks = (const device uint16_t*)keys;
    for (int i = 0; i < elem_per_thread / 4; i++) {
      k[4 * i] = ks[i] & 0x000f;
      k[4 * i + 1] = ks[i] & 0x00f0;
      k[4 * i + 2] = ks[i] & 0x0f00;
      k[4 * i + 3] = ks[i] & 0xf000;
    }
  } else if (bits == 8) {
    auto ks = (const device uint8_t*)keys;
    for (int i = 0; i < elem_per_thread; i++) {
      k[i] = ks[i];
    }
  }
}

template <typename U, int elem_per_thread, int bits>
METAL_FUNC void affine_load_values(
    const device uint32_t* values,
    thread U* v,
    U value_scale,
    U value_bias) {
  auto vs = (const device uint8_t*)values;
  if (bits == 4) {
    U s[2] = {value_scale, value_scale / 16.0f};
    for (int i = 0; i < elem_per_thread / 2; i++) {
      v[2 * i] = s[0] * (vs[i] & 0x0f) + value_bias;
      v[2 * i + 1] = s[1] * (vs[i] & 0xf0) + value_bias;
    }
  } else if (bits == 8) {
    for (int i = 0; i < elem_per_thread; i++) {
      v[i] = value_scale * vs[i] + value_bias;
    }
  }
}

// ============================================================================
// Affine quantized SDPA kernel (2-pass, pass 1)
// ============================================================================

template <typename T, int D, int group_size, int bits>
[[kernel]] void quant_sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device uint32_t* keys [[buffer(1)]],
    const device T* key_scales [[buffer(2)]],
    const device T* key_biases [[buffer(3)]],
    const device uint32_t* values [[buffer(4)]],
    const device T* value_scales [[buffer(5)]],
    const device T* value_biases [[buffer(6)]],
    device float* out [[buffer(7)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& v_stride,
    const constant size_t& k_group_stride,
    const constant size_t& v_group_stride,
    const constant float& scale,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  constexpr int BN = 8;
  constexpr int BD = 4;
  constexpr int elem_per_thread = D / BD;
  const int stride = BN * D;
  constexpr int nblocks = 32;
  constexpr int pack_factor = 32 / bits;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U v[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + quad_lid * elem_per_thread;

  const int kv_idx =
      (block_idx * BN + quad_gid) * D + quad_lid * elem_per_thread;
  const int packed_idx = kv_idx / pack_factor;
  const int k_group_idx = kv_head_idx * k_group_stride + kv_idx / group_size;
  const int v_group_idx = kv_head_idx * v_group_stride + kv_idx / group_size;

  keys += kv_head_idx * k_stride + packed_idx;
  key_scales += k_group_idx;
  key_biases += k_group_idx;
  values += kv_head_idx * v_stride + packed_idx;
  value_scales += v_group_idx;
  value_biases += v_group_idx;

  out += head_idx * nblocks * D + block_idx * D + quad_lid * elem_per_thread;
  sums += head_idx * nblocks + block_idx;
  maxs += head_idx * nblocks + block_idx;

  // Read the query and 0 the output accumulator
  U query_sum = affine_load_queries<T, U, elem_per_thread, bits>(
      queries, q, static_cast<U>(scale));
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -1e9;
  U sum_exp_score = 0;

  // For each key
  for (int i = block_idx * BN + quad_gid; i < N; i += nblocks * BN) {
    // Read the key
    affine_load_keys<U, elem_per_thread, bits>(keys, k);

    // Assume D % group_size == 0 so all the keys are in the same group
    U key_scale = key_scales[0];
    U key_bias = key_biases[0];

    // Compute the i-th score
    U score = 0;
    for (int j = 0; j < elem_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = score * key_scale + query_sum * key_bias;
    score = quad_sum(score);

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    U value_scale = value_scales[0];
    U value_bias = value_biases[0];
    affine_load_values<U, elem_per_thread, bits>(values, v, value_scale, value_bias);

    // Update the output accumulator
    for (int j = 0; j < elem_per_thread; j++) {
      o[j] = o[j] * factor + exp_score * v[j];
    }

    // Move the pointers to the next kv
    keys += nblocks * stride / pack_factor;
    key_scales += nblocks * stride / group_size;
    key_biases += nblocks * stride / group_size;
    values += nblocks * stride / pack_factor;
    value_scales += nblocks * stride / group_size;
    value_biases += nblocks * stride / group_size;
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (quad_lid == 0) {
    max_scores[quad_gid] = max_score;
    sum_exp_scores[quad_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (simd_lid < BN) ? max_scores[simd_lid] : -1e9;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[quad_lid * BN + quad_gid] =
        o[i] * fast::exp(max_scores[quad_gid] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (quad_gid == 0) {
      U output = outputs[quad_lid * BN];
      for (int j = 1; j < BN; j++) {
        output += outputs[quad_lid * BN + j];
      }
      out[i] = static_cast<T>(output);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ============================================================================
// Centroid LUT SDPA helpers (TurboQuant/PolarQuant style)
// ============================================================================

// ============================================================================
// Centroid LUT SDPA kernel (2-pass, pass 1) with sparse-V optimization
//
// Instead of affine dequant (value = scale * packed + bias), this uses:
//   value = centroids[packed_index] * norm
// where centroids is a small LUT (8 or 16 entries) and norm is per-position.
//
// Sparse-V: if attention weight < threshold, skip V dequant+accumulate.
//
// Structure matches sdpa_vector_2pass_1: 1 simdgroup of 32 threads, each
// thread handles D/32 elements. Each threadgroup processes one K/V position
// at a time, striding by nblocks. No tiling needed even for D=256 since
// elem_per_thread = 8.
//
// Byte-level addressing: packed K/V data is accessed via byte pointers to
// avoid alignment issues when elem_per_thread < pack_factor.
// ============================================================================

// LUT load with element offset within a position.
// pos_base: uint32_t pointer to the start of the current position's packed data
// elem_offset: the first element index this thread should read (must be even for 4-bit)
// Reads elem_per_thread elements starting at elem_offset.
template <typename U, int elem_per_thread, int bits>
METAL_FUNC void lut_load_at(
    const device uint32_t* pos_base,
    int elem_offset,
    thread U* out,
    threadgroup const U* centroids) {
  constexpr uint mask = (1u << bits) - 1u;
  if (bits == 3) {
    // 3-bit: bit-level extraction from uint32 words
    int start_bit = elem_offset * 3;
    for (int i = 0; i < elem_per_thread; i++) {
      int bit_pos = start_bit + i * 3;
      int word_idx = bit_pos / 32;
      int bit_offset = bit_pos % 32;
      uint word = pos_base[word_idx];
      uint idx = (word >> bit_offset) & mask;
      // Handle cross-word boundary
      if (bit_offset + bits > 32) {
        uint next_word = pos_base[word_idx + 1];
        int remaining_bits = 32 - bit_offset;
        idx = (idx | (next_word << remaining_bits)) & mask;
      }
      out[i] = centroids[idx];
    }
  } else if (bits == 4) {
    // 4-bit: 2 elements per byte, elem_offset must be even
    const device uint8_t* bytes = (const device uint8_t*)pos_base;
    bytes += elem_offset / 2;
    for (int i = 0; i < elem_per_thread / 2; i++) {
      uint idx0 = bytes[i] & 0x0f;
      uint idx1 = (bytes[i] >> 4) & 0x0f;
      out[2 * i] = centroids[idx0];
      out[2 * i + 1] = centroids[idx1];
    }
  } else if (bits == 8) {
    // 8-bit: 1 element per byte
    const device uint8_t* bytes = (const device uint8_t*)pos_base;
    bytes += elem_offset;
    for (int i = 0; i < elem_per_thread; i++) {
      uint idx = bytes[i];
      out[i] = centroids[idx];
    }
  }
}

template <typename T, int D, int bits>
[[kernel]] void lut_sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device uint32_t* keys [[buffer(1)]],
    const device T* k_norms [[buffer(2)]],
    const device uint32_t* values [[buffer(3)]],
    const device T* v_norms [[buffer(4)]],
    const device float* centroids_k_in [[buffer(5)]],
    const device float* centroids_v_in [[buffer(6)]],
    device float* out [[buffer(7)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& v_stride,
    const constant float& scale,
    const constant float& sparse_v_threshold,
    const constant int& n_centroids,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int nblocks = 32;
  constexpr int pack_factor = 32 / bits;
  constexpr int max_centroids = (1 << bits);
  // Packed uint32s per position
  constexpr int packed_per_pos = D / pack_factor;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread] = {0};

  // Centroid LUTs in threadgroup memory for fast access
  threadgroup U tg_centroids_k[max_centroids];
  threadgroup U tg_centroids_v[max_centroids];

  // Load centroids into threadgroup memory (32 threads cooperate)
  for (int c = simd_lid; c < max_centroids; c += BD) {
    tg_centroids_k[c] = (c < n_centroids) ?
        static_cast<U>(centroids_k_in[c]) : U(0);
    tg_centroids_v[c] = (c < n_centroids) ?
        static_cast<U>(centroids_v_in[c]) : U(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * elem_per_thread;

  // Position-level pointer: always uint32-aligned since each position's
  // packed data occupies packed_per_pos uint32s.
  // k_stride is in uint32 units for the head dimension.
  const device uint32_t* keys_pos =
      keys + kv_head_idx * k_stride + block_idx * packed_per_pos;
  const device uint32_t* values_pos =
      values + kv_head_idx * v_stride + block_idx * packed_per_pos;

  // Per-thread element offset within each position
  const int elem_offset = simd_lid * elem_per_thread;

  // k_norms/v_norms are per-position: shape [B, n_kv_heads, N]
  k_norms += kv_head_idx * N + block_idx;
  v_norms += kv_head_idx * N + block_idx;

  out += head_idx * nblocks * D + block_idx * D + simd_lid * elem_per_thread;
  sums += head_idx * nblocks + block_idx;
  maxs += head_idx * nblocks + block_idx;

  // Read the query
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = -1e9;
  U sum_exp_score = 0;

  // Stride for advancing to the next position in packed uint32 units
  const int pos_advance = nblocks * packed_per_pos;

  // For each key (one position at a time, striding by nblocks)
  for (int i = block_idx; i < N; i += nblocks) {
    // Load key via centroid LUT and compute dot product
    lut_load_at<U, elem_per_thread, bits>(
        keys_pos, elem_offset, k, tg_centroids_k);
    U score = 0;
    for (int j = 0; j < elem_per_thread; j++) {
      score += q[j] * k[j];
    }

    // Reduce across all 32 threads and apply per-position norm
    U kn = static_cast<U>(k_norms[0]);
    score = simd_sum(score) * kn;

    // Update the accumulators (online softmax)
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Sparse-V optimization: skip V dequant+accumulate if weight is negligible
    bool do_accumulate = (sparse_v_threshold <= 0.0f) ||
                         (exp_score > sparse_v_threshold * sum_exp_score);

    if (do_accumulate) {
      U vn = static_cast<U>(v_norms[0]);
      thread U v[elem_per_thread];
      lut_load_at<U, elem_per_thread, bits>(
          values_pos, elem_offset, v, tg_centroids_v);
      for (int j = 0; j < elem_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * v[j] * vn;
      }
    } else {
      // Still need to rescale existing output for the new max
      for (int j = 0; j < elem_per_thread; j++) {
        o[j] = o[j] * factor;
      }
    }

    // Move the pointers to the next kv position
    keys_pos += pos_advance;
    values_pos += pos_advance;
    k_norms += nblocks;
    v_norms += nblocks;
  }

  // Write the sum and max (only thread 0 needs to write these scalars)
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  // Each thread writes its portion of the output directly (no reduction needed
  // since each thread owns distinct output elements, like sdpa_vector_2pass_1)
  for (int i = 0; i < elem_per_thread; i++) {
    out[i] = o[i];
  }
}

// ============================================================================
// Phase 3: Compact-then-compute LUT SDPA kernel
//
// Separates K-score computation from V accumulation within a single kernel
// using batched stream compaction. Positions are processed in chunks:
//   1. Compute K-scores for a chunk of COMPACT_CHUNK positions
//   2. Find chunk_max, compute exp_scores, identify active positions
//   3. Only load and accumulate V for active positions (dense, no divergence)
//
// Key insight: Online softmax can be decomposed per-chunk. For a chunk of
// scores, we find chunk_max = max(scores), then:
//   new_global_max = max(old_max, chunk_max)
//   o *= exp(old_max - new_global_max)          // rescale existing output
//   For each active pos: o += exp(score - new_global_max) * v
//   sum_exp *= exp(old_max - new_global_max) + sum(exp(scores - new_global_max))
//
// This is mathematically equivalent to per-position online softmax because
// the max subtraction is purely for numerical stability.
//
// Same thread layout as lut_sdpa_vector_2pass_1: 1 simdgroup of 32 threads,
// each thread handles D/32 elements, nblocks=32 parallel blocks.
// ============================================================================

constant constexpr int COMPACT_CHUNK = 256;

template <typename T, int D, int bits>
[[kernel]] void lut_sdpa_compact_v_2pass_1(
    const device T* queries [[buffer(0)]],
    const device uint32_t* keys [[buffer(1)]],
    const device T* k_norms [[buffer(2)]],
    const device uint32_t* values [[buffer(3)]],
    const device T* v_norms [[buffer(4)]],
    const device float* centroids_k_in [[buffer(5)]],
    const device float* centroids_v_in [[buffer(6)]],
    device float* out [[buffer(7)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& v_stride,
    const constant float& scale,
    const constant float& sparse_v_threshold,
    const constant int& n_centroids,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int nblocks = 32;
  constexpr int pack_factor = 32 / bits;
  constexpr int max_centroids = (1 << bits);
  constexpr int packed_per_pos = D / pack_factor;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread] = {0};

  // Threadgroup memory:
  //   centroids: 2 * max_centroids floats (K and V LUTs)
  //   scores: COMPACT_CHUNK floats (raw K-scores for current chunk)
  //   active_indices: COMPACT_CHUNK ints (compacted position indices)
  //   shared scalars: chunk_sum, active_count
  threadgroup U tg_centroids_k[max_centroids];
  threadgroup U tg_centroids_v[max_centroids];
  threadgroup U tg_scores[COMPACT_CHUNK];
  threadgroup int tg_active_indices[COMPACT_CHUNK];
  threadgroup U tg_chunk_sum[1];
  threadgroup int tg_active_count[1];

  // Load centroids into threadgroup memory
  for (int c = simd_lid; c < max_centroids; c += BD) {
    tg_centroids_k[c] = (c < n_centroids) ?
        static_cast<U>(centroids_k_in[c]) : U(0);
    tg_centroids_v[c] = (c < n_centroids) ?
        static_cast<U>(centroids_v_in[c]) : U(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + simd_lid * elem_per_thread;

  const device uint32_t* keys_base = keys + kv_head_idx * k_stride;
  const device uint32_t* values_base = values + kv_head_idx * v_stride;
  const int elem_offset = simd_lid * elem_per_thread;
  const device T* k_norms_base = k_norms + kv_head_idx * N;
  const device T* v_norms_base = v_norms + kv_head_idx * N;

  out += head_idx * nblocks * D + block_idx * D + simd_lid * elem_per_thread;
  sums += head_idx * nblocks + block_idx;
  maxs += head_idx * nblocks + block_idx;

  // Read the query
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = -1e9;
  U sum_exp_score = 0;

  // Compute total positions this block processes
  int my_pos_count = (N - block_idx + nblocks - 1) / nblocks;
  if (block_idx >= N) my_pos_count = 0;

  // Global position indices for this block: block_idx, block_idx+nblocks,
  // block_idx+2*nblocks, ...
  // For chunk c within a batch, global_pos = block_idx + (pos_processed + c) * nblocks

  int pos_processed = 0;

  while (pos_processed < my_pos_count) {
    int chunk_size = min(COMPACT_CHUNK, my_pos_count - pos_processed);

    // ================================================================
    // Phase A: Compute K-scores for all positions in this chunk
    // ================================================================
    // All 32 threads cooperate on each position (simd_sum for dot product)
    for (int c = 0; c < chunk_size; c++) {
      int global_pos_idx = block_idx + (pos_processed + c) * nblocks;
      const device uint32_t* keys_pos =
          keys_base + global_pos_idx * packed_per_pos;

      lut_load_at<U, elem_per_thread, bits>(
          keys_pos, elem_offset, k, tg_centroids_k);

      U score = 0;
      for (int j = 0; j < elem_per_thread; j++) {
        score += q[j] * k[j];
      }

      U kn = static_cast<U>(k_norms_base[global_pos_idx]);
      score = simd_sum(score) * kn;

      // Thread 0 stores the score
      if (simd_lid == 0) {
        tg_scores[c] = score;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ================================================================
    // Phase B: Chunk-level softmax + compaction
    // ================================================================

    // Step 1: Find chunk_max using all 32 threads (parallel reduction)
    U chunk_max = -1e9;
    for (int c = simd_lid; c < chunk_size; c += BD) {
      chunk_max = max(chunk_max, tg_scores[c]);
    }
    chunk_max = simd_max(chunk_max);

    // Step 2: New global max and rescale factor
    U new_global_max = max(max_score, chunk_max);
    U global_rescale = fast::exp(max_score - new_global_max);

    // Step 3: Rescale existing output accumulator
    for (int j = 0; j < elem_per_thread; j++) {
      o[j] *= global_rescale;
    }
    sum_exp_score *= global_rescale;

    // Step 4: Compute exp_scores, identify active positions, compute chunk_sum
    // Thread 0 does the serial compaction (fast: just exp + compare on scalars)
    if (simd_lid == 0) {
      int active = 0;
      U chunk_sum = 0;
      for (int c = 0; c < chunk_size; c++) {
        U exp_s = fast::exp(tg_scores[c] - new_global_max);
        tg_scores[c] = exp_s;  // overwrite with exp_score for V pass
        chunk_sum += exp_s;

        // Position is active if its weight exceeds threshold * running_sum
        bool is_active = (sparse_v_threshold <= 0.0f) ||
                         (exp_s > sparse_v_threshold * (sum_exp_score + chunk_sum));
        if (is_active) {
          tg_active_indices[active] = c;
          active++;
        }
      }
      tg_chunk_sum[0] = chunk_sum;
      tg_active_count[0] = active;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: All threads read compaction results
    int active_count = tg_active_count[0];
    U chunk_sum = tg_chunk_sum[0];
    sum_exp_score += chunk_sum;
    max_score = new_global_max;

    // ================================================================
    // Phase C: Dense V accumulation for active positions only
    // ================================================================
    for (int a = 0; a < active_count; a++) {
      int c = tg_active_indices[a];
      int global_pos_idx = block_idx + (pos_processed + c) * nblocks;

      const device uint32_t* values_pos =
          values_base + global_pos_idx * packed_per_pos;
      U vn = static_cast<U>(v_norms_base[global_pos_idx]);
      U exp_s = tg_scores[c];

      thread U v[elem_per_thread];
      lut_load_at<U, elem_per_thread, bits>(
          values_pos, elem_offset, v, tg_centroids_v);

      for (int j = 0; j < elem_per_thread; j++) {
        o[j] += exp_s * v[j] * vn;
      }
    }

    pos_processed += chunk_size;
  }

  // Write the sum and max
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < elem_per_thread; i++) {
    out[i] = o[i];
  }
}
