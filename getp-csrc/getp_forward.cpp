#include <hip/hip_runtime.h>
#include <malloc.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "collectives.hpp"
#include "getp_transformer.hpp"
#include "profiler.hpp"

extern CollectiveGroup g_world;

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size) {
  extern __shared__ float smem[];
  float *ss_ptr = &smem[blockDim.x];
  int tid = threadIdx.x;
  smem[tid] = 0;
  for (int offset = 0; offset < size; offset += blockDim.x) {
    if (tid + offset < size) {
      float t = x[tid + offset];
      smem[tid] += t * t;
    }
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }
  if (tid == 0) *ss_ptr = smem[0];
  __syncthreads();
  float ss = *ss_ptr;
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  for (int offset = 0; offset < size; offset += blockDim.x) {
    if (tid + offset < size) {
      o[tid + offset] = weight[tid + offset] * (ss * x[tid + offset]);
    }
  }
}
void getp_rmsnorm(float *o, float *x, float *weight, int size) {
  PROFILE_FUNCTION();
  dim3 blockDim(1024);
  dim3 gridDim(1);
  rmsnorm_kernel<<<gridDim, blockDim, sizeof(float) * (blockDim.x + 1)>>>(
      o, x, weight, size);
  HIP_CHECK(hipDeviceSynchronize());
}

// Warp reduce sum (HIP warpSize = 64)
__device__ inline float warp_sum_f32(float v) {
#pragma unroll
  for (int off = warpSize >> 1; off > 0; off >>= 1) v += __shfl_down(v, off);
  return v;
}

// Quick BF16 bits -> float (no library)
union __bf16_bits_u {
  __hip_bfloat16 b;
  unsigned short u;
};
__device__ inline float bf16bits_to_f32(unsigned short u) {
  __bf16_bits_u t;
  t.u = u;
  return __bfloat162float(t.b);
}
// Vectorized, cache x in shared, each warp computes one output row
template <int TILE_N = 512, int WARPS_PER_BLOCK = 8>
__global__ void gemv_fp32_vec_kernel(float *__restrict__ y,
                                     const float *__restrict__ x,
                                     const float *__restrict__ W,
                                     const float *__restrict__ B, int n,
                                     int d) {
  extern __shared__ float sX[];  // TILE_N floats
  const int lane = threadIdx.x;  // 0..63
  const int warp = threadIdx.y;  // 0..WARPS_PER_BLOCK-1
  const int out_row = blockIdx.x * WARPS_PER_BLOCK + warp;

  float acc = 0.f;

  for (int tile = 0; tile < n; tile += TILE_N) {
    const int tile_len = min(TILE_N, n - tile);

    // cooperative load of x into shared (vectorized float4 where possible)
    for (int t = warp * warpSize * 4 + lane * 4; t < tile_len;
         t += WARPS_PER_BLOCK * warpSize * 4) {
      if (t + 3 < tile_len) {
        reinterpret_cast<float4 &>(sX[t]) =
            *reinterpret_cast<const float4 *>(&x[tile + t]);
      } else {
        for (int j = 0; j < 4 && t + j < tile_len; ++j)
          sX[t + j] = x[tile + t + j];
      }
    }
    __syncthreads();

    if (out_row < d) {
      const float *wrow = W + (size_t)out_row * n + tile;
      const float4 *x4 = reinterpret_cast<const float4 *>(sX);
      const float4 *w4 = reinterpret_cast<const float4 *>(wrow);

      float partial = 0.f;
      const int it4 = tile_len >> 2;
#pragma unroll 4
      for (int k4 = lane; k4 < it4; k4 += warpSize) {
        float4 a = w4[k4];
        float4 b = x4[k4];
        partial += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
      }
      // tail
      for (int k = (it4 << 2) + lane; k < tile_len; k += warpSize) {
        partial += wrow[k] * sX[k];
      }
      acc += warp_sum_f32(partial);
    }
    __syncthreads();
  }

  if (out_row < d && lane == 0) {
    y[out_row] = acc + (B ? B[out_row] : 0.f);
  }
}
template <int TILE_N = 512, int WARPS_PER_BLOCK = 8>
__global__ void gemv_bf16_vec_kernel(float *__restrict__ y,
                                     const float *__restrict__ x,
                                     const __hip_bfloat16 *__restrict__ W,
                                     const __hip_bfloat16 *__restrict__ B,
                                     int n, int d) {
  extern __shared__ float sX[];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int out_row = blockIdx.x * WARPS_PER_BLOCK + warp;

  float acc = 0.f;

  for (int tile = 0; tile < n; tile += TILE_N) {
    const int tile_len = min(TILE_N, n - tile);

    // cooperative load x -> shared, vectorized
    for (int t = warp * warpSize * 4 + lane * 4; t < tile_len;
         t += WARPS_PER_BLOCK * warpSize * 4) {
      if (t + 3 < tile_len) {
        reinterpret_cast<float4 &>(sX[t]) =
            *reinterpret_cast<const float4 *>(&x[tile + t]);
      } else {
        for (int j = 0; j < 4 && t + j < tile_len; ++j)
          sX[t + j] = x[tile + t + j];
      }
    }
    __syncthreads();

    if (out_row < d) {
      const __hip_bfloat16 *wrow_b = W + (size_t)out_row * n + tile;

      // reinterpret as ushort4 (4 * 16-bit = 64b = coalesced)
      const ushort4 *w4 = reinterpret_cast<const ushort4 *>(wrow_b);
      const float4 *x4 = reinterpret_cast<const float4 *>(sX);

      float partial = 0.f;
      const int it4 = tile_len >> 2;
#pragma unroll 4
      for (int k4 = lane; k4 < it4; k4 += warpSize) {
        ushort4 wb = w4[k4];
        float4 xb = x4[k4];
        partial += bf16bits_to_f32(wb.x) * xb.x + bf16bits_to_f32(wb.y) * xb.y +
                   bf16bits_to_f32(wb.z) * xb.z + bf16bits_to_f32(wb.w) * xb.w;
      }
      // tail
      for (int k = (it4 << 2) + lane; k < tile_len; k += warpSize) {
        __bf16_bits_u u;
        u.b = wrow_b[k];
        partial += __bfloat162float(u.b) * sX[k];
      }

      acc += warp_sum_f32(partial);
    }
    __syncthreads();
  }

  if (out_row < d && lane == 0) {
    float bias = 0.f;
    if (B) {
      __bf16_bits_u u;
      u.b = B[out_row];
      bias = __bfloat162float(u.b);
    }
    y[out_row] = acc + bias;
  }
}
template <typename T>
void getp_matmul(float *xout, float *x, T *w, T *b, int n, int d) {
  PROFILE_FUNCTION();
  constexpr int WARPS = 8;     // 8 warps/block (512 threads)
  constexpr int TILE_N = 512;  // 512 elements of x/tile in shared
  dim3 block(64, WARPS);
  dim3 grid((d + WARPS - 1) / WARPS);
  size_t shmem = TILE_N * sizeof(float);

  if constexpr (std::is_same<T, float>::value) {
    gemv_fp32_vec_kernel<TILE_N, WARPS><<<grid, block, shmem>>>(
        xout, x, (const float *)w, (const float *)b, n, d);
  } else {  // __hip_bfloat16
    gemv_bf16_vec_kernel<TILE_N, WARPS><<<grid, block, shmem>>>(
        xout, x, (const __hip_bfloat16 *)w, (const __hip_bfloat16 *)b, n, d);
  }
  HIP_CHECK(hipDeviceSynchronize());
}
template <int TILE_N = 512, int WARPS_PER_BLOCK = 8>
__global__ void gemv_qkv_vec_kernel(float *__restrict__ q_out,
                                    float *__restrict__ k_out,
                                    float *__restrict__ v_out,
                                    const float *__restrict__ x,
                                    const float *__restrict__ W,
                                    const float *__restrict__ B, int n,
                                    int q_len, int k_len, int v_len) {
  extern __shared__ float sX[];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int out_row = blockIdx.x * WARPS_PER_BLOCK + warp;
  const int d_total = q_len + k_len + v_len;

  float acc = 0.f;

  for (int tile = 0; tile < n; tile += TILE_N) {
    const int tile_len = min(TILE_N, n - tile);

    for (int t = warp * warpSize * 4 + lane * 4; t < tile_len;
         t += WARPS_PER_BLOCK * warpSize * 4) {
      if (t + 3 < tile_len) {
        reinterpret_cast<float4 &>(sX[t]) =
            *reinterpret_cast<const float4 *>(&x[tile + t]);
      } else {
        for (int j = 0; j < 4 && t + j < tile_len; ++j)
          sX[t + j] = x[tile + t + j];
      }
    }
    __syncthreads();

    if (out_row < d_total) {
      const float *wrow = W + (size_t)out_row * n + tile;
      const float4 *x4 = reinterpret_cast<const float4 *>(sX);
      const float4 *w4 = reinterpret_cast<const float4 *>(wrow);

      float partial = 0.f;
      const int it4 = tile_len >> 2;
#pragma unroll 4
      for (int k4 = lane; k4 < it4; k4 += warpSize) {
        float4 a = w4[k4];
        float4 b = x4[k4];
        partial += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
      }
      for (int k = (it4 << 2) + lane; k < tile_len; k += warpSize) {
        partial += wrow[k] * sX[k];
      }
      acc += warp_sum_f32(partial);
    }
    __syncthreads();
  }

  if (out_row < d_total && lane == 0) {
    const float bias = (B ? B[out_row] : 0.f);
    float *dst = nullptr;
    int idx = 0;
    if (out_row < q_len) {
      dst = q_out;
      idx = out_row;
    } else if (out_row < q_len + k_len) {
      dst = k_out;
      idx = out_row - q_len;
    } else {
      dst = v_out;
      idx = out_row - q_len - k_len;
    }
    dst[idx] = acc + bias;
  }
}

void getp_matmul_qkv_fused(float *q, float *k, float *v, float *x,
                           const float *w_qkv, const float *b_qkv, int n,
                           int head_dim, int n_attn_heads, int n_kv_heads) {
  PROFILE_FUNCTION();
  const int q_len = head_dim * n_attn_heads;
  const int k_len = head_dim * n_kv_heads;
  const int v_len = head_dim * n_kv_heads;
  const int d_total = q_len + k_len + v_len;

  constexpr int WARPS = 8;
  constexpr int TILE_N = 512;
  dim3 block(64, WARPS);
  dim3 grid((d_total + WARPS - 1) / WARPS);
  size_t shmem = TILE_N * sizeof(float);

  gemv_qkv_vec_kernel<TILE_N, WARPS><<<grid, block, shmem>>>(
      q, k, v, x, w_qkv, b_qkv, n, q_len, k_len, v_len);
  HIP_CHECK(hipDeviceSynchronize());
}

__global__ void compute_inv_freq_kernel(float base, int head_dim,
                                        float scaling_factor,
                                        float initial_context_length,
                                        float ntk_beta, float ntk_alpha,
                                        float *inv_freq_out) {
  const int d_half = head_dim / 2;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= d_half) return;
  float freq = powf(base, ((float)(2 * i)) / (float)head_dim);
  float inv_freq;
  if (scaling_factor > 1.0f) {
    float low = d_half *
                logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) /
                logf(base);
    float high = d_half *
                 logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) /
                 logf(base);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)i - low) / (high - low);
    if (ramp < 0) ramp = 0;
    if (ramp > 1) ramp = 1;
    float mask = 1.0f - ramp;
    inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
  } else {
    inv_freq = 1.0f / freq;
  }
  inv_freq_out[i] = inv_freq;
}
__global__ void compute_cos_sin_kernel(float *cos_out, float *sin_out,
                                       float *inv_freq, float concentration,
                                       int pos, int d_half) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= d_half) return;
  float val = (float)pos * inv_freq[i];
  cos_out[i] = cosf(val) * concentration;
  sin_out[i] = sinf(val) * concentration;
}
void getp_compute_cos_sin(int pos, float base, int head_dim,
                          float scaling_factor, float initial_context_length,
                          float ntk_beta, float ntk_alpha, float *cos_out,
                          float *sin_out) {
  PROFILE_FUNCTION();
  int d_half = head_dim / 2;
  float concentration =
      scaling_factor > 1.0f ? 0.1f * logf(scaling_factor) + 1.0f : 1.0f;
  float *inv_freq;
  HIP_CHECK(hipMalloc(&inv_freq, d_half * sizeof(float)));
  {
    dim3 blockDim(256);
    dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x);
    compute_inv_freq_kernel<<<gridDim, blockDim>>>(
        base, head_dim, scaling_factor, initial_context_length, ntk_beta,
        ntk_alpha, inv_freq);
  }
  {
    dim3 blockDim(256);
    dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x);
    compute_cos_sin_kernel<<<gridDim, blockDim>>>(
        cos_out, sin_out, inv_freq, concentration, pos, head_dim / 2);
  }
  HIP_CHECK(hipFree(inv_freq));
  HIP_CHECK(hipDeviceSynchronize());
}

__global__ void apply_rotary_emb_kernel(float *x, float *cos, float *sin,
                                        int n_heads, int head_dim) {
  const int half = head_dim / 2;
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (h >= n_heads || i >= half) return;
  float x1 = x[h * head_dim + i];
  float x2 = x[h * head_dim + half + i];
  float c = cos[i];
  float s = sin[i];
  float o1 = x1 * c - x2 * s;
  float o2 = x2 * c + x1 * s;
  x[h * head_dim + i] = o1;
  x[h * head_dim + half + i] = o2;
}
void getp_apply_rotary_emb(float *x, float *cos, float *sin, int n_heads,
                           int head_dim) {
  PROFILE_FUNCTION();
  dim3 blockDim(16, 16);
  dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x,
               (n_heads + blockDim.y - 1) / blockDim.y);
  apply_rotary_emb_kernel<<<gridDim, blockDim>>>(x, cos, sin, n_heads,
                                                 head_dim);
  HIP_CHECK(hipDeviceSynchronize());
}

__global__ void multihead_attention_kernel(
    float *key_cache, float *value_cache, float *query, float *mask,
    float *attn_sinks, float *attn, int l, int head_dim, int n_attn_heads,
    int n_kv_heads, int pos, int att_lda, int mask_lda, int apply_mask) {
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  if (h >= n_attn_heads || t > pos + 1) return;
  if (t == pos + 1) {
    if (t < att_lda) attn[h * att_lda + t] = attn_sinks[l * n_attn_heads + h];
    return;
  }
  const int kv_dim = head_dim * n_kv_heads;
  const int kv_mul = n_attn_heads / n_kv_heads;
  const float *q = query + h * head_dim;
  const float *k = key_cache + t * kv_dim + (h / kv_mul) * head_dim;
  float score = 0;
  for (int i = 0; i < head_dim; ++i) score += q[i] * k[i];
  score = score / sqrtf((float)head_dim);
  if (apply_mask && t <= pos) score += mask[pos * mask_lda + t];
  attn[h * att_lda + t] = score;
}
void getp_multihead_attention(float *key_cache, float *value_cache,
                              float *query, float *mask, float *attn_sinks,
                              float *attn, int l, int head_dim,
                              int n_attn_heads, int n_kv_heads, int pos,
                              int att_lda, int mask_lda, int sliding_window) {
  PROFILE_FUNCTION();
  int apply_mask = (sliding_window > 0 && (l % 2 == 0) ? 1 : 0);
  dim3 blockDim(16, 16);
  dim3 gridDim((pos + 2 + blockDim.x - 1) / blockDim.x,
               (n_attn_heads + blockDim.y - 1) / blockDim.y);
  multihead_attention_kernel<<<gridDim, blockDim>>>(
      key_cache, value_cache, query, mask, attn_sinks, attn, l, head_dim,
      n_attn_heads, n_kv_heads, pos, att_lda, mask_lda, apply_mask);
  HIP_CHECK(hipDeviceSynchronize());
}

template <int TILE_T = 128>
__global__ void weighted_sum_tiled_kernel(float *__restrict__ tb,
                                          const float *__restrict__ value_cache,
                                          const float *__restrict__ att,
                                          int seq_len, int n_attn_heads,
                                          int n_kv_heads, int pos,
                                          int head_dim) {
  const int h = blockIdx.y;
  const int i0 = blockIdx.x * blockDim.x;
  const int i = i0 + threadIdx.x;
  if (h >= n_attn_heads) return;
  extern __shared__ float s_att[];
  const int kv_dim = head_dim * n_kv_heads;
  const int kv_mul = n_attn_heads / n_kv_heads;
  const int kv_h = h / kv_mul;
  float acc = 0.f;
  for (int t0 = 0; t0 <= pos; t0 += TILE_T) {
    const int tlen = min(TILE_T, pos - t0 + 1);
    for (int tt = threadIdx.x; tt < tlen; tt += blockDim.x)
      s_att[tt] = att[h * seq_len + (t0 + tt)];
    __syncthreads();
    if (i < head_dim) {
      const float *vptr =
          value_cache + (size_t)t0 * kv_dim + kv_h * head_dim + i;
      for (int tt = 0; tt < tlen; ++tt) {
        acc += s_att[tt] * (*vptr);
        vptr += kv_dim;
      }
    }
    __syncthreads();
  }
  if (i < head_dim) tb[h * head_dim + i] = acc;
}
void getp_weighted_sum(float *tb, float *value_cache, float *att, int seq_len,
                       int n_attn_heads, int n_kv_heads, int pos,
                       int head_dim) {
  PROFILE_FUNCTION();
  const int BLK_X = 32;
  dim3 block(BLK_X);
  dim3 grid((head_dim + BLK_X - 1) / BLK_X, n_attn_heads);
  const size_t shmem = (size_t)min(128, pos + 1) * sizeof(float);
  weighted_sum_tiled_kernel<128><<<grid, block, shmem>>>(
      tb, value_cache, att, seq_len, n_attn_heads, n_kv_heads, pos, head_dim);
  HIP_CHECK(hipDeviceSynchronize());
}

__global__ void softmax_kernel(float *A, int M, int N, int lda) {
  extern __shared__ float smem[];
  float *max_ptr = &smem[blockDim.x];
  float *sum_ptr = &smem[blockDim.x + 1];
  A += blockIdx.y * lda;
  int tid = threadIdx.x;
  smem[tid] = -INFINITY;
  for (int offset = 0; offset < N; offset += blockDim.x) {
    if (tid + offset < N) {
      float t = A[tid + offset];
      if (t > smem[tid]) smem[tid] = t;
    }
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      float t = smem[tid + stride];
      if (t > smem[tid]) smem[tid] = t;
    }
    __syncthreads();
  }
  if (tid == 0) *max_ptr = smem[0];
  __syncthreads();
  float max_val = *max_ptr;
  smem[tid] = 0;
  for (int offset = 0; offset < N; offset += blockDim.x) {
    if (tid + offset < N) smem[tid] += expf(A[tid + offset] - max_val);
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }
  if (tid == 0) *sum_ptr = smem[0];
  __syncthreads();
  float sum = *sum_ptr;
  for (int offset = 0; offset < N; offset += blockDim.x) {
    if (tid + offset < N) {
      float x = A[tid + offset];
      x = expf(x - max_val) / sum;
      A[tid + offset] = x;
    }
  }
}
void getp_softmax(float *A, int M, int N, int lda) {
  PROFILE_FUNCTION();
  dim3 blockDim(1024);
  dim3 gridDim(1, M);
  softmax_kernel<<<gridDim, blockDim, sizeof(float) * (blockDim.x + 2)>>>(
      A, M, N, lda);
  HIP_CHECK(hipDeviceSynchronize());
}

__global__ void vecadd_kernel(float *x, float *y, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= size) return;
  x[i] += y[i];
}
void getp_vecadd(float *x, float *y, int size) {
  PROFILE_FUNCTION();
  dim3 blockDim(1024);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
  vecadd_kernel<<<gridDim, blockDim>>>(x, y, size);
  HIP_CHECK(hipDeviceSynchronize());
}

__global__ void topk_kernel(float *router_score, float *topk_values,
                            int *topk_indices, int n_experts,
                            int experts_per_token) {
  int tid = threadIdx.x;
  if (tid >= experts_per_token) return;
  if (tid < n_experts) {
    topk_values[tid] = router_score[tid];
    topk_indices[tid] = tid;
  } else {
    topk_values[tid] = -INFINITY;
    topk_indices[tid] = -1;
  }
  __syncthreads();
  for (int i = experts_per_token; i < n_experts; i++) {
    __syncthreads();
    float current_score = router_score[i];
    if (tid == 0) {
      int min_idx = 0;
      float min_val = topk_values[0];
      for (int j = 1; j < experts_per_token; j++)
        if (topk_values[j] < min_val) {
          min_val = topk_values[j];
          min_idx = j;
        }
      if (current_score > min_val) {
        topk_values[min_idx] = current_score;
        topk_indices[min_idx] = i;
      }
    }
  }
}

__global__ void weighted_aggregate_kernel(float *e_agg, float *expert_out,
                                          float expert_weight, int hidden_dim) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= hidden_dim) return;
  atomicAdd(&e_agg[i], expert_out[i] * expert_weight);
}
void getp_topk(float *topk_values, int *topk_indices, float *router_score,
               int n_experts, int experts_per_token) {
  PROFILE_FUNCTION();
  dim3 blockDim(experts_per_token);
  dim3 gridDim(1);
  topk_kernel<<<gridDim, blockDim>>>(router_score, topk_values, topk_indices,
                                     n_experts, experts_per_token);
  HIP_CHECK(hipDeviceSynchronize());
}

void getp_weighted_aggregate(float *e_agg, float *expert_out,
                             float expert_weight, int hidden_dim) {
  PROFILE_FUNCTION();
  dim3 blockDim(256);
  dim3 gridDim((hidden_dim + blockDim.x - 1) / blockDim.x);
  weighted_aggregate_kernel<<<gridDim, blockDim>>>(e_agg, expert_out,
                                                   expert_weight, hidden_dim);
  HIP_CHECK(hipDeviceSynchronize());
}
__global__ void swiglu_fused_kernel(float *gate_up, const float *mlp1_out,
                                    int intermediate_dim, float swiglu_limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= intermediate_dim) return;
  float g = mlp1_out[2 * i + 0];
  float u = mlp1_out[2 * i + 1];
  if (g > swiglu_limit) g = swiglu_limit;
  if (u > swiglu_limit) u = swiglu_limit;
  if (u < -swiglu_limit) u = -swiglu_limit;
  const float a = 1.702f;
  float s = 1.f / (1.f + expf(-a * g));
  float v = (g * s) * (u + 1.f);
  gate_up[i] = v;
}

void getp_swiglu_fused(float *gate_up, float *mlp1_out, int intermediate_dim,
                       float swiglu_limit) {
  PROFILE_FUNCTION();
  dim3 block(256);
  dim3 grid((intermediate_dim + block.x - 1) / block.x);
  swiglu_fused_kernel<<<grid, block>>>(gate_up, mlp1_out, intermediate_dim,
                                       swiglu_limit);
  HIP_CHECK(hipDeviceSynchronize());
}

float *getp_forward(Transformer *transformer,
                    DeviceTransformer **dev_transformeres, int token, int pos) {
  PROFILE_FUNCTION();
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  DeviceTransformerWeights *dev_w = &dev_transformeres[0]->weights;
  RunState *dev_s = &dev_transformeres[0]->state;
  int n_devices;
  HIP_CHECK(hipGetDeviceCount(&n_devices));
  float *x = s->x;
  float *dev_x = dev_s->x;
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul = p->n_attn_heads / p->n_kv_heads;
  int intermediate_dim = p->intermediate_dim;
  int n_experts = p->n_experts;

  float *content_row = w->token_embedding_table + token * hidden_dim;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemcpy(dev_x, content_row, hidden_dim * sizeof(*x),
                      hipMemcpyHostToDevice));

  float *dev_cos_vals, *dev_sin_vals;
  HIP_CHECK(hipMalloc(&dev_cos_vals, sizeof(float) * (head_dim / 2)));
  HIP_CHECK(hipMalloc(&dev_sin_vals, sizeof(float) * (head_dim / 2)));
  float ntk_beta = 32.0f, ntk_alpha = 1.0f;
  getp_compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                       p->initial_context_length, ntk_beta, ntk_alpha,
                       dev_cos_vals, dev_sin_vals);

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    HIP_CHECK(hipSetDevice(0));
    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_attn_w + 1ll * l * hidden_dim,
                 hidden_dim);
    int loff = l * p->seq_len * kv_dim;
    dev_s->k = dev_s->key_cache + loff + pos * kv_dim;
    dev_s->v = dev_s->value_cache + loff + pos * kv_dim;

    float *dev_w_qkv = dev_w->w_qkv + 1ll * l * hidden_dim *
                                          (head_dim * p->n_attn_heads +
                                           2 * head_dim * p->n_kv_heads);
    float *dev_b_qkv =
        dev_w->b_qkv +
        1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    getp_matmul_qkv_fused(dev_s->q, dev_s->k, dev_s->v, dev_s->t, dev_w_qkv,
                          dev_b_qkv, hidden_dim, head_dim, p->n_attn_heads,
                          p->n_kv_heads);

    getp_apply_rotary_emb(dev_s->q, dev_cos_vals, dev_sin_vals, p->n_attn_heads,
                          head_dim);
    getp_apply_rotary_emb(dev_s->k, dev_cos_vals, dev_sin_vals, p->n_kv_heads,
                          head_dim);

    int att_lda = p->seq_len + 1;
    int mask_lda = p->seq_len;
    getp_multihead_attention(dev_s->key_cache + loff, dev_s->value_cache + loff,
                             dev_s->q, dev_s->mask, dev_w->attn_sinks,
                             dev_s->att, l, head_dim, p->n_attn_heads,
                             p->n_kv_heads, pos, att_lda, mask_lda,
                             p->sliding_window);
    int Nsoft = MIN(pos + 2, att_lda);
    getp_softmax(dev_s->att, p->n_attn_heads, Nsoft, att_lda);
    getp_weighted_sum(dev_s->tb, dev_s->value_cache + loff, dev_s->att, att_lda,
                      p->n_attn_heads, p->n_kv_heads, pos, head_dim);

    float *dev_w_o =
        dev_w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    float *dev_b_o = dev_w->b_o + 1ll * l * hidden_dim;
    getp_matmul<float>(dev_s->tb2, dev_s->tb, dev_w_o, dev_b_o,
                       head_dim * p->n_attn_heads, hidden_dim);
    getp_vecadd(dev_x, dev_s->tb2, hidden_dim);

    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim,
                 hidden_dim);

    float *dev_w_router = dev_w->w_router + 1ll * l * hidden_dim * n_experts;
    float *dev_b_router = dev_w->b_router + 1ll * l * n_experts;
    getp_matmul(dev_s->router_score, dev_s->t, dev_w_router, dev_b_router,
                hidden_dim, n_experts);
    getp_topk(dev_s->topk_v, dev_s->topk_i, dev_s->router_score, n_experts,
              p->experts_per_token);
    getp_softmax(dev_s->topk_v, 1, p->experts_per_token, p->experts_per_token);

    HIP_CHECK(hipMemset(dev_s->e_agg, 0, hidden_dim * sizeof(float)));
    HIP_CHECK(hipMemcpy(s->topk_v, dev_s->topk_v,
                        p->experts_per_token * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(s->topk_i, dev_s->topk_i,
                        p->experts_per_token * sizeof(int),
                        hipMemcpyDeviceToHost));

    for (int device_id = 0; device_id < n_devices; device_id++) {
      HIP_CHECK(hipSetDevice(device_id));
      int experts_per_device = n_experts / n_devices;
      int expert_start = device_id * experts_per_device;
      int expert_end = expert_start + experts_per_device;
      DeviceTransformerWeights *curr_dev_w =
          &dev_transformeres[device_id]->weights;
      RunState *curr_dev_s = &dev_transformeres[device_id]->state;
      if (device_id != 0)
        HIP_CHECK(hipMemcpyPeer(curr_dev_s->t, device_id, dev_s->t, 0,
                                sizeof(float) * hidden_dim));
      HIP_CHECK(hipMemset(curr_dev_s->e_agg, 0, hidden_dim * sizeof(float)));
      for (int idx = 0; idx < p->experts_per_token; idx++) {
        int expert_id = s->topk_i[idx];
        float expert_weight = s->topk_v[idx];
        if (expert_id >= expert_start && expert_id < expert_end) {
          int local_expert_id = expert_id - expert_start;
          __hip_bfloat16 *dev_w_mlp1 =
              curr_dev_w->w_mlp1 +
              1ll * (l * experts_per_device + local_expert_id) *
                  (2 * p->intermediate_dim) * hidden_dim;
          __hip_bfloat16 *dev_b_mlp1 =
              curr_dev_w->b_mlp1 +
              1ll * (l * experts_per_device + local_expert_id) *
                  (2 * p->intermediate_dim);
          getp_matmul<__hip_bfloat16>(curr_dev_s->mlp1_out, curr_dev_s->t,
                                      dev_w_mlp1, dev_b_mlp1, hidden_dim,
                                      2 * p->intermediate_dim);
          // getp_split_gate_up(curr_dev_s->mlp1_out, curr_dev_s->gate,
          // curr_dev_s->up, p->intermediate_dim);
          // getp_swiglu(curr_dev_s->gate_up, curr_dev_s->gate, curr_dev_s->up,
          // p->intermediate_dim, p->swiglu_limit);
          getp_swiglu_fused(curr_dev_s->gate_up, curr_dev_s->mlp1_out,
                            p->intermediate_dim, p->swiglu_limit);

          __hip_bfloat16 *dev_w_mlp2 =
              curr_dev_w->w_mlp2 +
              1ll * (l * experts_per_device + local_expert_id) * hidden_dim *
                  p->intermediate_dim;
          __hip_bfloat16 *dev_b_mlp2 =
              curr_dev_w->b_mlp2 +
              1ll * (l * experts_per_device + local_expert_id) * hidden_dim;
          getp_matmul<__hip_bfloat16>(curr_dev_s->tb2, curr_dev_s->gate_up,
                                      dev_w_mlp2, dev_b_mlp2,
                                      p->intermediate_dim, hidden_dim);
          getp_weighted_aggregate(curr_dev_s->e_agg, curr_dev_s->tb2,
                                  expert_weight, hidden_dim);
        }
      }
      if (device_id != 0) {
        HIP_CHECK(hipMemcpyPeer(dev_s->tb2, 0, curr_dev_s->e_agg, device_id,
                                sizeof(float) * hidden_dim));
        HIP_CHECK(hipSetDevice(0));
        getp_vecadd(dev_s->e_agg, dev_s->tb2, hidden_dim);
      }
    }
    HIP_CHECK(hipSetDevice(0));
    getp_vecadd(dev_x, dev_s->e_agg, hidden_dim);
  }

  HIP_CHECK(hipFree(dev_cos_vals));
  HIP_CHECK(hipFree(dev_sin_vals));
  getp_rmsnorm(dev_x, dev_x, dev_w->rms_out_w, hidden_dim);
  getp_matmul(dev_s->logits, dev_x, dev_w->out, (float *)NULL, hidden_dim,
              p->vocab_size);
  HIP_CHECK(hipMemcpy(s->logits, dev_s->logits, sizeof(float) * p->vocab_size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  return s->logits;
}
