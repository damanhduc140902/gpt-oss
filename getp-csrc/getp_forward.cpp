#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <malloc.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "collectives.hpp"
#include "getp_transformer.cpp"
#include "getp_transformer.hpp"
#include "profiler.hpp"
#ifndef GEMV_TILE_N
#define GEMV_TILE_N 512
#endif
#ifndef GEMV_WARPS_PER_BLOCK
#define GEMV_WARPS_PER_BLOCK 4
#endif
extern CollectiveGroup g_world;

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size) {
  extern __shared__ float smem[];
  float *ss_ptr = &smem[blockDim.x];
  int tid = threadIdx.x;
  // blockIdx.y equal to batch index
  // shift o,x to the corresponding batch
  o += blockIdx.y * size;
  x += blockIdx.y * size; 
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
void getp_rmsnorm(float *o, float *x, float *weight, int batch_size, int dim) {
  PROFILE_FUNCTION();
  dim3 blockDim(1024);
  dim3 gridDim(1, batch_size);
  rmsnorm_kernel<<<gridDim, blockDim, sizeof(float) * (blockDim.x + 1)>>>(o, x, weight, dim);
  // HIP_CHECK(hipDeviceSynchronize());
}

// Warp reduce sum (HIP wavefront = 64)
__device__ inline float warp_sum_f32(float v) {
#pragma unroll
  for (int off = warpSize >> 1; off > 0; off >>= 1) v += __shfl_down(v, off);
  return v;
}

union __bf16_bits_u {
  __hip_bfloat16 b;
  unsigned short u;
};
__device__ inline float bf16bits_to_f32(unsigned short u) {
  __bf16_bits_u t;
  t.u = u;
  return __bfloat162float(t.b);
}
template <int TILE_N, int WARPS_PER_BLOCK>
__global__ void gemv_qkv_bf16_vec(float *__restrict__ q_out,
                                  float *__restrict__ k_out,
                                  float *__restrict__ v_out,
                                  const float *__restrict__ x,
                                  const __hip_bfloat16 *__restrict__ W,
                                  const __hip_bfloat16 *__restrict__ B, int n,
                                  int q_len, int k_len, int v_len) {
  extern __shared__ float sX[];
  const int lane = threadIdx.x;  // 0..63
  const int warp = threadIdx.y;  // 0..WARPS_PER_BLOCK-1
  const int out_row = blockIdx.x * WARPS_PER_BLOCK + warp;
  const int d_total = q_len + k_len + v_len;

  q_out += blockIdx.y * q_len;
  k_out += blockIdx.y * k_len;
  v_out += blockIdx.y * v_len;
  x     += blockIdx.y * n;

  float partial_total = 0.f;

  for (int tile = 0; tile < n; tile += TILE_N) {
    const int tile_len = min(TILE_N, n - tile);

    // x -> shared, vector 16B
    for (int t = warp * warpSize * 4 + lane * 4; t < tile_len;
         t += WARPS_PER_BLOCK * warpSize * 4) {
      if (t + 3 < tile_len)
        reinterpret_cast<float4 &>(sX[t]) =
            *reinterpret_cast<const float4 *>(&x[tile + t]);
      else {
        for (int j = 0; j < 4 && t + j < tile_len; ++j)
          sX[t + j] = x[tile + t + j];
      }
    }
    __syncthreads();

    if (out_row < d_total) {
      const __hip_bfloat16 *__restrict__ wrow_b =
          W + (size_t)out_row * n + tile;

      // 128-bit load: 8 bf16 / lần
      const uint4 *w8 = reinterpret_cast<const uint4 *>(wrow_b);
      const float4 *x4 = reinterpret_cast<const float4 *>(sX);

      float partial = 0.f;
      const int it8 = tile_len >> 3;
#pragma unroll 4
      for (int k8 = lane; k8 < it8; k8 += warpSize) {
        uint4 wb = w8[k8];
        unsigned short w01 = (unsigned short)(wb.x & 0xFFFF);
        unsigned short w02 = (unsigned short)(wb.x >> 16);
        unsigned short w11 = (unsigned short)(wb.y & 0xFFFF);
        unsigned short w12 = (unsigned short)(wb.y >> 16);
        unsigned short w21 = (unsigned short)(wb.z & 0xFFFF);
        unsigned short w22 = (unsigned short)(wb.z >> 16);
        unsigned short w31 = (unsigned short)(wb.w & 0xFFFF);
        unsigned short w32 = (unsigned short)(wb.w >> 16);

        float4 xb0 = x4[(k8 << 1) + 0];
        float4 xb1 = x4[(k8 << 1) + 1];

        // partial += bf16bits_to_f32(w01) * xb0.x + bf16bits_to_f32(w02) * xb0.y +
        //            bf16bits_to_f32(w11) * xb0.z + bf16bits_to_f32(w12) * xb0.w +
        //            bf16bits_to_f32(w21) * xb1.x + bf16bits_to_f32(w22) * xb1.y +
        //            bf16bits_to_f32(w31) * xb1.z + bf16bits_to_f32(w32) * xb1.w;
        partial = fmaf(bf16bits_to_f32(w01), xb0.x, partial);
        partial = fmaf(bf16bits_to_f32(w02), xb0.y, partial);
        partial = fmaf(bf16bits_to_f32(w11), xb0.z, partial);
        partial = fmaf(bf16bits_to_f32(w12), xb0.w, partial);
        partial = fmaf(bf16bits_to_f32(w21), xb1.x, partial);
        partial = fmaf(bf16bits_to_f32(w22), xb1.y, partial);
        partial = fmaf(bf16bits_to_f32(w31), xb1.z, partial);
        partial = fmaf(bf16bits_to_f32(w32), xb1.w, partial);
      }
      for (int k = (it8 << 3) + lane; k < tile_len; k += warpSize) {
        __bf16_bits_u u;
        u.b = wrow_b[k];
        partial += __bfloat162float(u.b) * sX[k];
      }
      partial_total += partial;
    }
    __syncthreads();
  }

  if (out_row < d_total) {
    float acc = warp_sum_f32(partial_total);
    if (threadIdx.x == 0) {
      float bias = 0.f;
      if (B) {
        __bf16_bits_u u;
        u.b = B[out_row];
        bias = __bfloat162float(u.b);
      }
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
}

static inline void getp_matmul_qkv_fused_bf16(
    float *q, float *k, float *v, float *x, const __hip_bfloat16 *w_qkv_bf16,
    const __hip_bfloat16 *b_qkv_bf16, int n, int head_dim, int n_attn_heads,
    int n_kv_heads, int batch_size) {
  PROFILE_FUNCTION();
  const int q_len = head_dim * n_attn_heads;
  const int k_len = head_dim * n_kv_heads;
  const int v_len = head_dim * n_kv_heads;
  const int d_total = q_len + k_len + v_len;

  constexpr int WARPS = GEMV_WARPS_PER_BLOCK;
  constexpr int TILE = GEMV_TILE_N;
  dim3 block(64, WARPS);
  dim3 grid((d_total + WARPS - 1) / WARPS, batch_size);
  size_t shmem = TILE * sizeof(float);
  gemv_qkv_bf16_vec<TILE, WARPS><<<grid, block, shmem>>>(
      q, k, v, x, w_qkv_bf16, b_qkv_bf16, n, q_len, k_len, v_len);
  HIP_CHECK(hipDeviceSynchronize());
}

template <int TILE_N, int WARPS_PER_BLOCK>
__global__ void gemv_bf16_vec_v2(float *__restrict__ y,
                                 const float *__restrict__ x,
                                 const __hip_bfloat16 *__restrict__ W,
                                 const __hip_bfloat16 *__restrict__ B, int n,
                                 int d) {
  extern __shared__ float sX[];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int out_row = blockIdx.x * WARPS_PER_BLOCK + warp;
  // blockIdx.y equal to batch index
  // shift x,y to the corresponding batch
  y += blockIdx.y * d;
  x += blockIdx.y * n;

  float partial_total = 0.f;

  for (int tile = 0; tile < n; tile += TILE_N) {
    const int tile_len = min(TILE_N, n - tile);

    // x -> shared (vector 16B)
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

      // 128-bit load: 8×bf16 mỗi lần
      const uint4 *w8 = reinterpret_cast<const uint4 *>(wrow_b);
      const float4 *x4 = reinterpret_cast<const float4 *>(sX);

      float partial = 0.f;
      const int it8 = tile_len >> 3;  // 8 elements per 128-bit load
#pragma unroll 4
      for (int k8 = lane; k8 < it8; k8 += warpSize) {
        uint4 wb = w8[k8];
        // tách 8 bf16: wb.x,y,z,w mỗi cái chứa 2 bf16 (4 bytes)
        unsigned short w01 = (unsigned short)(wb.x & 0xFFFF);
        unsigned short w02 = (unsigned short)(wb.x >> 16);
        unsigned short w11 = (unsigned short)(wb.y & 0xFFFF);
        unsigned short w12 = (unsigned short)(wb.y >> 16);
        unsigned short w21 = (unsigned short)(wb.z & 0xFFFF);
        unsigned short w22 = (unsigned short)(wb.z >> 16);
        unsigned short w31 = (unsigned short)(wb.w & 0xFFFF);
        unsigned short w32 = (unsigned short)(wb.w >> 16);

        // x tương ứng: 8 phần tử = 2 * float4
        float4 xb0 = x4[(k8 << 1) + 0];
        float4 xb1 = x4[(k8 << 1) + 1];

        partial += bf16bits_to_f32(w01) * xb0.x + bf16bits_to_f32(w02) * xb0.y +
                   bf16bits_to_f32(w11) * xb0.z + bf16bits_to_f32(w12) * xb0.w +
                   bf16bits_to_f32(w21) * xb1.x + bf16bits_to_f32(w22) * xb1.y +
                   bf16bits_to_f32(w31) * xb1.z + bf16bits_to_f32(w32) * xb1.w;
      }
      // tail còn lại (<8)
      for (int k = (it8 << 3) + lane; k < tile_len; k += warpSize) {
        __bf16_bits_u u;
        u.b = wrow_b[k];
        partial += __bfloat162float(u.b) * sX[k];
      }
      partial_total += partial;
    }
    __syncthreads();
  }

  if (out_row < d) {
    float acc = warp_sum_f32(partial_total);
    if (lane == 0) {
      float bias = 0.f;
      if (B) {
        __bf16_bits_u u;
        u.b = B[out_row];
        bias = __bfloat162float(u.b);
      }
      y[out_row] = acc + bias;
    }
  }
}
template <typename T>
void getp_matmul(float *xout, float *x, T *w, T *b, int n, int d, int batch_size) {
  PROFILE_FUNCTION();
  constexpr int WARPS = GEMV_WARPS_PER_BLOCK;
  constexpr int TILE_N = GEMV_TILE_N;
  dim3 block(64, WARPS);
  dim3 grid((d + WARPS - 1) / WARPS, batch_size);
  size_t shmem = TILE_N * sizeof(float);

  gemv_bf16_vec_v2<TILE_N, WARPS><<<grid, block, shmem>>>(
      xout, x, (const __hip_bfloat16 *)w, (const __hip_bfloat16 *)b, n, d);

  HIP_CHECK(hipDeviceSynchronize());
}

template <int TILE_N, int WARPS_PER_BLOCK, int MAX_E = 4>
__global__ void mlp1_swiglu_bf16_kernel_batch(
    float *__restrict__ gate_up_all,             // [n_active, intermediate_dim]
    const float *__restrict__ x,                 // [hidden_dim]
    const __hip_bfloat16 *__restrict__ W_layer,  // layer-base: [E_dev, 2D, H]
    const __hip_bfloat16 *__restrict__ B_layer,  // layer-base: [E_dev, 2D]
    int hidden_dim, int intermediate_dim, int experts_per_token,
    int4 expert_local_ids_0, int n_active_0, 
    int4 expert_local_ids_1, int n_active_1, 
    float swiglu_limit) {
  extern __shared__ float sX[];  // TILE_N floats
  const int lane = threadIdx.x;  // 0..63
  const int warp = threadIdx.y;  // 0..WARPS_PER_BLOCK-1
  const int j = blockIdx.x * WARPS_PER_BLOCK + warp;

  int4 expert_local_ids = blockIdx.y == 0 ? expert_local_ids_0 : expert_local_ids_1;
  int n_active = blockIdx.y == 0 ? n_active_0 : n_active_1;

  gate_up_all += blockIdx.y * experts_per_token * intermediate_dim;
  x           += blockIdx.y * hidden_dim;

  float g_tot[MAX_E], u_tot[MAX_E];
#pragma unroll
  for (int e = 0; e < MAX_E; ++e) {
    g_tot[e] = 0.f;
    u_tot[e] = 0.f;
  }

  // tile theo K=hidden_dim
  for (int tile = 0; tile < hidden_dim; tile += TILE_N) {
    const int tlen = min(TILE_N, hidden_dim - tile);

    // nạp x -> shared (vector 16B)
    for (int t = warp * warpSize * 4 + lane * 4; t < tlen;
         t += WARPS_PER_BLOCK * warpSize * 4) {
      if (t + 3 < tlen)
        reinterpret_cast<float4 &>(sX[t]) =
            *reinterpret_cast<const float4 *>(&x[tile + t]);
      else
        for (int k = 0; k < 4 && t + k < tlen; ++k) sX[t + k] = x[tile + t + k];
    }
    __syncthreads();

    if (j < intermediate_dim) {
      const float4 *x4 = reinterpret_cast<const float4 *>(sX);
      const int it8 = tlen >> 3;  // 8 bf16 per 128-bit

#pragma unroll
      for (int ee = 0; ee < MAX_E; ++ee) {
        if (ee >= n_active) break;
        const int local_id =
            ((const int *)&expert_local_ids)[ee];  // 0..E_dev-1
        if (local_id < 0) continue;

        // offset tới expert ee trong layer hiện tại
        const size_t row_stride = (size_t)hidden_dim;
        const size_t expert_base = (size_t)local_id *
                                   (size_t)(2 * intermediate_dim) *
                                   (size_t)hidden_dim;
        const __hip_bfloat16 *wg =
            W_layer + expert_base + (size_t)(2 * j + 0) * row_stride + tile;
        const __hip_bfloat16 *wu =
            W_layer + expert_base + (size_t)(2 * j + 1) * row_stride + tile;

        const uint4 *wg8 = reinterpret_cast<const uint4 *>(wg);
        const uint4 *wu8 = reinterpret_cast<const uint4 *>(wu);

        float g = 0.f, u = 0.f;
#pragma unroll 4
        for (int k8 = lane; k8 < it8; k8 += warpSize) {
          uint4 ag = wg8[k8], au = wu8[k8];
          // unpack 8 bf16/gói 128-bit
          unsigned short g01 = (unsigned short)(ag.x & 0xFFFF);
          unsigned short g02 = (unsigned short)(ag.x >> 16);
          unsigned short g11 = (unsigned short)(ag.y & 0xFFFF);
          unsigned short g12 = (unsigned short)(ag.y >> 16);
          unsigned short g21 = (unsigned short)(ag.z & 0xFFFF);
          unsigned short g22 = (unsigned short)(ag.z >> 16);
          unsigned short g31 = (unsigned short)(ag.w & 0xFFFF);
          unsigned short g32 = (unsigned short)(ag.w >> 16);

          unsigned short u01 = (unsigned short)(au.x & 0xFFFF);
          unsigned short u02 = (unsigned short)(au.x >> 16);
          unsigned short u11 = (unsigned short)(au.y & 0xFFFF);
          unsigned short u12 = (unsigned short)(au.y >> 16);
          unsigned short u21 = (unsigned short)(au.z & 0xFFFF);
          unsigned short u22 = (unsigned short)(au.z >> 16);
          unsigned short u31 = (unsigned short)(au.w & 0xFFFF);
          unsigned short u32 = (unsigned short)(au.w >> 16);

          float4 xb0 = x4[(k8 << 1) + 0];
          float4 xb1 = x4[(k8 << 1) + 1];

          g = fmaf(bf16bits_to_f32(g01), xb0.x, g);
          g = fmaf(bf16bits_to_f32(g02), xb0.y, g);
          g = fmaf(bf16bits_to_f32(g11), xb0.z, g);
          g = fmaf(bf16bits_to_f32(g12), xb0.w, g);
          g = fmaf(bf16bits_to_f32(g21), xb1.x, g);
          g = fmaf(bf16bits_to_f32(g22), xb1.y, g);
          g = fmaf(bf16bits_to_f32(g31), xb1.z, g);
          g = fmaf(bf16bits_to_f32(g32), xb1.w, g);

          u = fmaf(bf16bits_to_f32(u01), xb0.x, u);
          u = fmaf(bf16bits_to_f32(u02), xb0.y, u);
          u = fmaf(bf16bits_to_f32(u11), xb0.z, u);
          u = fmaf(bf16bits_to_f32(u12), xb0.w, u);
          u = fmaf(bf16bits_to_f32(u21), xb1.x, u);
          u = fmaf(bf16bits_to_f32(u22), xb1.y, u);
          u = fmaf(bf16bits_to_f32(u31), xb1.z, u);
          u = fmaf(bf16bits_to_f32(u32), xb1.w, u);
        }
        for (int k = (it8 << 3) + lane; k < tlen; k += warpSize) {
          g += __bfloat162float(wg[k]) * sX[k];
          u += __bfloat162float(wu[k]) * sX[k];
        }
        g_tot[ee] += g;
        u_tot[ee] += u;
      }  // end loop expert
    }
    __syncthreads();
  }

  if (j < intermediate_dim) {
#pragma unroll
    for (int ee = 0; ee < MAX_E; ++ee) {
      if (ee >= n_active) break;
      const int local_id = ((const int *)&expert_local_ids)[ee];
      if (local_id < 0) continue;

      float g = g_tot[ee], u = u_tot[ee];
      // reduce trong warp
#pragma unroll
      for (int off = warpSize >> 1; off > 0; off >>= 1) {
        g += __shfl_down(g, off);
        u += __shfl_down(u, off);
      }
      if (lane == 0) {
        const size_t b_off = (size_t)local_id * (size_t)(2 * intermediate_dim);
        float bg =
            B_layer ? __bfloat162float(B_layer[b_off + (2 * j + 0)]) : 0.f;
        float bu =
            B_layer ? __bfloat162float(B_layer[b_off + (2 * j + 1)]) : 0.f;
        g += bg;
        u += bu;

        if (g > swiglu_limit) g = swiglu_limit;
        if (u > swiglu_limit) u = swiglu_limit;
        if (u < -swiglu_limit) u = -swiglu_limit;

        const float a = 1.702f;
        float s = 1.f / (1.f + expf(-a * g));
        gate_up_all[(size_t)ee * (size_t)intermediate_dim + j] =
            (g * s) * (u + 1.f);
      }
    }
  }
}

static inline void getp_mlp1_swiglu_bf16_batch(
    float *gate_up_all, float *x, const __hip_bfloat16 *w_mlp1_layer_base,
    const __hip_bfloat16 *b_mlp1_layer_base, int hidden_dim,
    int intermediate_dim, int experts_per_token, 
    int4 expert_local_ids_0, int n_active_0, 
    int4 expert_local_ids_1, int n_active_1,
    float swiglu_limit, int batch_size) {
  PROFILE_FUNCTION();
  constexpr int WARPS = GEMV_WARPS_PER_BLOCK;
  constexpr int TILE = GEMV_TILE_N;
  dim3 block(64, WARPS);
  dim3 grid((intermediate_dim + WARPS - 1) / WARPS, batch_size);
  size_t shmem = TILE * sizeof(float);

  mlp1_swiglu_bf16_kernel_batch<TILE, WARPS><<<grid, block, shmem>>>(
      gate_up_all, x, w_mlp1_layer_base, b_mlp1_layer_base, hidden_dim,
      intermediate_dim, experts_per_token, 
      expert_local_ids_0, n_active_0,
      expert_local_ids_1, n_active_1,
      swiglu_limit);
  HIP_CHECK(hipDeviceSynchronize());
}

template <int TILE_N, int WARPS_PER_BLOCK, int MAX_E = 4>
__global__ void mlp2_accum_bf16_kernel_batch(
    float *__restrict__ e_agg,                   // += combined
    const float *__restrict__ gate_up_all,       // [n_active, intermediate_dim]
    const __hip_bfloat16 *__restrict__ W_layer,  // base of layer: [E_dev, H, I]
    const __hip_bfloat16 *__restrict__ B_layer,  // base of layer: [E_dev, H]
    float4 expert_weights4_0,                    // packed weights (<=4)
    int4 expert_local_ids_0, int n_active_0,
    float4 expert_weights4_1,                    // packed weights (<=4)
    int4 expert_local_ids_1, int n_active_1,
    int intermediate_dim, int hidden_dim, int experts_per_token
) {
  extern __shared__ float sX[];  // size = MAX_E * TILE_N
  const int lane = threadIdx.x;  // 0..63
  const int warp = threadIdx.y;  // 0..WARPS_PER_BLOCK-1
  const int out_row = blockIdx.x * WARPS_PER_BLOCK + warp;

  float4 expert_weights4 = blockIdx.y == 0 ? expert_weights4_0 : expert_weights4_1;
  int4 expert_local_ids = blockIdx.y == 0 ? expert_local_ids_0 : expert_local_ids_1;
  int n_active = blockIdx.y == 0 ? n_active_0 : n_active_1;

  e_agg += blockIdx.y * hidden_dim;
  gate_up_all += blockIdx.y * experts_per_token * intermediate_dim;

  // unpack weights to registers
  float wts[MAX_E] = {expert_weights4.x, expert_weights4.y, expert_weights4.z,
                      expert_weights4.w};
  float acc[MAX_E];
#pragma unroll
  for (int e = 0; e < MAX_E; ++e) acc[e] = 0.f;

  for (int tile = 0; tile < intermediate_dim; tile += TILE_N) {
    const int tlen = min(TILE_N, intermediate_dim - tile);

#pragma unroll
    for (int ee = 0; ee < MAX_E; ++ee) {
      if (ee >= n_active) break;
      const float *src =
          gate_up_all + (size_t)ee * (size_t)intermediate_dim + tile;
      float *dst = sX + ee * TILE_N;
      for (int t = warp * warpSize * 4 + lane * 4; t < tlen;
           t += WARPS_PER_BLOCK * warpSize * 4) {
        if (t + 3 < tlen) {
          reinterpret_cast<float4 &>(dst[t]) =
              *reinterpret_cast<const float4 *>(&src[t]);
        } else {
          for (int k = 0; k < 4 && t + k < tlen; ++k) dst[t + k] = src[t + k];
        }
      }
    }
    __syncthreads();

    if (out_row < hidden_dim) {
      const int it8 = tlen >> 3;
#pragma unroll 1
      for (int ee = 0; ee < n_active; ++ee) {
        const int local_id = ((const int *)&expert_local_ids)[ee];
        if (local_id < 0) continue;

        const size_t expert_base =
            (size_t)local_id * (size_t)hidden_dim * (size_t)intermediate_dim;
        const __hip_bfloat16 *wrow =
            W_layer + expert_base + (size_t)out_row * (size_t)intermediate_dim +
            tile;

        const uint4 *w8 = reinterpret_cast<const uint4 *>(wrow);
        const float4 *x4 = reinterpret_cast<const float4 *>(sX + ee * TILE_N);

        float part = 0.f;
#pragma unroll 4
        for (int k8 = lane; k8 < it8; k8 += warpSize) {
          uint4 wb = w8[k8];
          unsigned short w01 = (unsigned short)(wb.x & 0xFFFF);
          unsigned short w02 = (unsigned short)(wb.x >> 16);
          unsigned short w11 = (unsigned short)(wb.y & 0xFFFF);
          unsigned short w12 = (unsigned short)(wb.y >> 16);
          unsigned short w21 = (unsigned short)(wb.z & 0xFFFF);
          unsigned short w22 = (unsigned short)(wb.z >> 16);
          unsigned short w31 = (unsigned short)(wb.w & 0xFFFF);
          unsigned short w32 = (unsigned short)(wb.w >> 16);

          float4 xb0 = x4[(k8 << 1) + 0];
          float4 xb1 = x4[(k8 << 1) + 1];

          part = fmaf(bf16bits_to_f32(w01), xb0.x, part);
          part = fmaf(bf16bits_to_f32(w02), xb0.y, part);
          part = fmaf(bf16bits_to_f32(w11), xb0.z, part);
          part = fmaf(bf16bits_to_f32(w12), xb0.w, part);
          part = fmaf(bf16bits_to_f32(w21), xb1.x, part);
          part = fmaf(bf16bits_to_f32(w22), xb1.y, part);
          part = fmaf(bf16bits_to_f32(w31), xb1.z, part);
          part = fmaf(bf16bits_to_f32(w32), xb1.w, part);
        }
        for (int k = (it8 << 3) + lane; k < tlen; k += warpSize) {
          __bf16_bits_u u;
          u.b = wrow[k];
          part += __bfloat162float(u.b) * (sX[ee * TILE_N + k]);
        }
        acc[ee] += part;
      }  // end loop experts
    }
    __syncthreads();
  }  // end tile loop

  if (out_row < hidden_dim) {
#pragma unroll
    for (int ee = 0; ee < MAX_E; ++ee)
      if (ee < n_active) acc[ee] = warp_sum_f32(acc[ee]);

    if (threadIdx.x == 0) {
      float sum = 0.f;
#pragma unroll
      for (int ee = 0; ee < MAX_E; ++ee) {
        if (ee >= n_active) break;
        const int local_id = ((const int *)&expert_local_ids)[ee];
        if (local_id < 0) continue;
        float bias = 0.f;
        if (B_layer) {
          bias = __bfloat162float(
              B_layer[(size_t)local_id * (size_t)hidden_dim + out_row]);
        }
        sum += wts[ee] * (acc[ee] + bias);
      }
      e_agg[out_row] += sum;  // write once per row
    }
  }
}

static inline void getp_mlp2_accum_bf16_batch(
    float *e_agg, const float *gate_up_all, const __hip_bfloat16 *w2_layer_base,
    const __hip_bfloat16 *b2_layer_base, int experts_per_token,
    int4 expert_local_ids_0, int n_active_0, float4 expert_weights4_0,
    int4 expert_local_ids_1, int n_active_1, float4 expert_weights4_1,
    int intermediate_dim, int hidden_dim, int batch_size) {
  PROFILE_FUNCTION();
  constexpr int WARPS = GEMV_WARPS_PER_BLOCK;
  constexpr int TILE = GEMV_TILE_N;
  dim3 block(64, WARPS);
  dim3 grid((hidden_dim + WARPS - 1) / WARPS, batch_size);
  size_t shmem = (size_t)TILE * 4 * sizeof(float);  // 4 expert * TILE_N
  mlp2_accum_bf16_kernel_batch<TILE, WARPS>
      <<<grid, block, shmem>>>(e_agg, gate_up_all, w2_layer_base, b2_layer_base,
                               expert_weights4_0, 
                               expert_local_ids_0, n_active_0,
                               expert_weights4_1, 
                               expert_local_ids_1, n_active_1,
                               intermediate_dim, hidden_dim, experts_per_token);
  // HIP_CHECK(hipDeviceSynchronize());
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
  // blockIdx.z equal to batch index
  // shift x to the corresponding batch
  x += blockIdx.z * n_heads * head_dim;
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
void getp_apply_rotary_emb(float *x, float *cos, float *sin, int n_heads, int head_dim, int batch_size) {
  PROFILE_FUNCTION();
  dim3 blockDim(16, 16);
  dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x, 
               (n_heads + blockDim.y - 1) / blockDim.y,
               batch_size);
  apply_rotary_emb_kernel<<<gridDim, blockDim>>>(x, cos, sin, n_heads, head_dim);
  // HIP_CHECK(hipDeviceSynchronize());
}

__global__ void multihead_attention_kernel(float *key_cache, float *value_cache, float *query,
                                           float *mask, float *attn_sinks, float *attn,
                                           int l, int head_dim, int n_attn_heads, int n_kv_heads,
                                           int pos, int att_lda, int mask_lda, int apply_mask, int batch_size) {
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  int bid = blockIdx.z; // batch index
  attn += bid * n_attn_heads * att_lda;
  if (h >= n_attn_heads || t > pos + 1) return;
  if (t == pos + 1) {
    if (t < att_lda) attn[h * att_lda + t] = attn_sinks[l * n_attn_heads + h];
    return;
  }
  const int kv_dim = head_dim * n_kv_heads;
  const int kv_mul = n_attn_heads / n_kv_heads;
  const float *q = query + bid * n_attn_heads * head_dim + h * head_dim;
  const float *k = key_cache + t * batch_size * kv_dim + bid * kv_dim + (h / kv_mul) * head_dim;
  float score = 0;
  for (int i = 0; i < head_dim; ++i) score += q[i] * k[i];
  score = score / sqrtf((float)head_dim);
  if (apply_mask && t <= pos) score += mask[pos * mask_lda + t];
  attn[h * att_lda + t] = score;
}
void getp_multihead_attention(float *key_cache, float *value_cache, float *query, float *mask,
                              float *attn_sinks, float *attn, int l, int head_dim, int n_attn_heads,
                              int n_kv_heads, int pos, int att_lda, int mask_lda, int sliding_window, int batch_size) {
  PROFILE_FUNCTION();
  int apply_mask = (sliding_window > 0 && (l % 2 == 0) ? 1 : 0);
  dim3 blockDim(16, 16);
  dim3 gridDim((pos + 2 + blockDim.x - 1) / blockDim.x, 
               (n_attn_heads + blockDim.y - 1) / blockDim.y,
               batch_size);
  multihead_attention_kernel<<<gridDim, blockDim>>>(key_cache, value_cache, query, mask, attn_sinks, attn,
                                                    l, head_dim, n_attn_heads, n_kv_heads, pos, att_lda, mask_lda, 
                                                    apply_mask, batch_size);
  // HIP_CHECK(hipDeviceSynchronize());
}

template <int TILE_T = 128>
__global__ void weighted_sum_tiled_kernel(float *__restrict__ tb,
                                          const float *__restrict__ value_cache,
                                          const float *__restrict__ att,
                                          int seq_len, int n_attn_heads,
                                          int n_kv_heads, int pos,
                                          int head_dim, int batch_size) {
  const int h = blockIdx.y;
  const int i0 = blockIdx.x * blockDim.x;
  const int i = i0 + threadIdx.x;
  const int b = blockIdx.z; // batch
  att += b * n_attn_heads * seq_len;
  tb += b * n_attn_heads * head_dim;
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
          value_cache + (size_t)t0 * batch_size * kv_dim + b * kv_dim + kv_h * head_dim + i;
      for (int tt = 0; tt < tlen; ++tt) {
        acc += s_att[tt] * (*vptr);
        vptr += batch_size * kv_dim;
      }
    }
    __syncthreads();
  }
  if (i < head_dim) tb[h * head_dim + i] = acc;
}

void getp_weighted_sum(float *tb, float *value_cache, float *att, int seq_len,
                       int n_attn_heads, int n_kv_heads, int pos, int head_dim, int batch_size) {
  PROFILE_FUNCTION();
  const int BLK_X = 32;
  dim3 block(BLK_X);
  dim3 grid((head_dim + BLK_X - 1) / BLK_X, n_attn_heads, batch_size);
  const size_t shmem = (size_t)MIN(128, pos + 1) * sizeof(float);
  weighted_sum_tiled_kernel<128><<<grid, block, shmem>>>(tb, value_cache, att, seq_len, n_attn_heads, n_kv_heads, pos, head_dim, batch_size);
  // HIP_CHECK(hipDeviceSynchronize());
}

__global__ void softmax_kernel(float *A, int M, int N, int lda) {
  extern __shared__ float smem[];
  float *max_ptr = &smem[blockDim.x];
  float *sum_ptr = &smem[blockDim.x + 1];
  // blockIdx.y equal to batch index
  // shift to the corresponding batch
  A += blockIdx.y * M * lda + blockIdx.x * lda;
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
void getp_softmax(float *A, int M, int N, int lda, int batch_size) {
  PROFILE_FUNCTION();
  dim3 blockDim(1024);
  dim3 gridDim(M, batch_size);
  softmax_kernel<<<gridDim, blockDim, sizeof(float) * (blockDim.x + 2)>>>(A, M, N, lda);
  // HIP_CHECK(hipDeviceSynchronize());
}

__global__ void vecadd_kernel(float *x, float *y, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // blockIdx.y equal to batch index
  // shift x, y to the corresponding batch
  x += blockIdx.y * size;
  y += blockIdx.y * size;
  if (i >= size) return;
  x[i] += y[i];
}
void getp_vecadd(float *x, float *y, int size, int batch_size) {
  PROFILE_FUNCTION();
  dim3 blockDim(1024);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x, batch_size);
  vecadd_kernel<<<gridDim, blockDim>>>(x, y, size);
  HIP_CHECK(hipDeviceSynchronize());
}

__global__ void topk_kernel(float *router_score, float *topk_values,
                            int *topk_indices, int n_experts,
                            int experts_per_token) {
  int tid = threadIdx.x;
  router_score += blockIdx.x * n_experts;
  topk_values  += blockIdx.x * experts_per_token;
  topk_indices += blockIdx.x * experts_per_token;
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

void getp_topk(float *topk_values, int *topk_indices, float *router_score,
               int n_experts, int experts_per_token, int batch_size) {
  PROFILE_FUNCTION();
  dim3 blockDim(experts_per_token);
  dim3 gridDim(batch_size);
  topk_kernel<<<gridDim, blockDim>>>(router_score, topk_values, topk_indices,
                                     n_experts, experts_per_token);
  HIP_CHECK(hipDeviceSynchronize());
}

float *getp_forward(Transformer *transformer,
                    DeviceTransformer **dev_transformeres, int token[], int pos) {
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

  float *topk_v = 
    reinterpret_cast<float *>(malloc(sizeof(float) * BATCH_SIZE * p->experts_per_token));
  int *topk_i = 
    reinterpret_cast<int *>(malloc(sizeof(int) * BATCH_SIZE * p->experts_per_token));

  HIP_CHECK(hipSetDevice(0));
  for (int b = 0; b < BATCH_SIZE; ++b) {
    float *content_row = w->token_embedding_table + token[b] * hidden_dim;
    HIP_CHECK(hipMemcpy(dev_x + b * hidden_dim, content_row, hidden_dim * sizeof(*x), hipMemcpyHostToDevice));
  }

  float *dev_cos_vals, *dev_sin_vals;
  HIP_CHECK(hipMalloc(&dev_cos_vals, sizeof(float) * (head_dim / 2)));
  HIP_CHECK(hipMalloc(&dev_sin_vals, sizeof(float) * (head_dim / 2)));
  float ntk_beta = 32.0f, ntk_alpha = 1.0f;
  getp_compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                       p->initial_context_length, ntk_beta, ntk_alpha,
                       dev_cos_vals, dev_sin_vals);

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    HIP_CHECK(hipSetDevice(0));
    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_attn_w + 1ll * l * hidden_dim, BATCH_SIZE, hidden_dim);
    int loff = l * p->seq_len * kv_dim;
    dev_s->k = dev_s->key_cache + loff + pos * BATCH_SIZE * kv_dim;
    dev_s->v = dev_s->value_cache + loff + pos * BATCH_SIZE * kv_dim;

    __hip_bfloat16 *dev_w_qkv =
        dev_w->w_qkv_bf16 +
        1ll * l * hidden_dim *
            (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    __hip_bfloat16 *dev_b_qkv =
        dev_w->b_qkv_bf16 +
        1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    getp_matmul_qkv_fused_bf16(dev_s->q, dev_s->k, dev_s->v, dev_s->t,
                               dev_w_qkv, dev_b_qkv, hidden_dim, head_dim,
                               p->n_attn_heads, p->n_kv_heads, BATCH_SIZE);

    getp_apply_rotary_emb(dev_s->q, dev_cos_vals, dev_sin_vals, p->n_attn_heads,
                          head_dim, BATCH_SIZE);
    getp_apply_rotary_emb(dev_s->k, dev_cos_vals, dev_sin_vals, p->n_kv_heads,
                          head_dim, BATCH_SIZE);

    int att_lda = p->seq_len + 1;
    int mask_lda = p->seq_len;
    getp_multihead_attention(dev_s->key_cache + loff, dev_s->value_cache + loff, dev_s->q,
                             dev_s->mask, dev_w->attn_sinks, dev_s->att, l, head_dim,
                             p->n_attn_heads, p->n_kv_heads, pos, att_lda, mask_lda, p->sliding_window, BATCH_SIZE);
    int Nsoft = MIN(pos + 2, att_lda);
    getp_softmax(dev_s->att, p->n_attn_heads, Nsoft, att_lda, BATCH_SIZE);
    getp_weighted_sum(dev_s->tb, dev_s->value_cache + loff, dev_s->att, att_lda,
                      p->n_attn_heads, p->n_kv_heads, pos, head_dim, BATCH_SIZE);

    __hip_bfloat16 *dev_w_o =
        dev_w->w_o_bf16 + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    __hip_bfloat16 *dev_b_o = dev_w->b_o_bf16 + 1ll * l * hidden_dim;
    getp_matmul<__hip_bfloat16>(dev_s->tb2, dev_s->tb, dev_w_o, dev_b_o,
                                head_dim * p->n_attn_heads, hidden_dim, BATCH_SIZE);

    getp_vecadd(dev_x, dev_s->tb2, hidden_dim, BATCH_SIZE);

    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim, BATCH_SIZE,
                 hidden_dim);

    __hip_bfloat16 *dev_w_router =
        dev_w->w_router_bf16 + 1ll * l * hidden_dim * n_experts;
    __hip_bfloat16 *dev_b_router = dev_w->b_router_bf16 + 1ll * l * n_experts;
    getp_matmul<__hip_bfloat16>(dev_s->router_score, dev_s->t, dev_w_router,
                                dev_b_router, hidden_dim, n_experts, BATCH_SIZE);

    HIP_CHECK(hipMemset(dev_s->e_agg, 0, hidden_dim * sizeof(float)));
    
    getp_topk(dev_s->topk_v, dev_s->topk_i, dev_s->router_score, n_experts,
                p->experts_per_token, BATCH_SIZE);
    getp_softmax(dev_s->topk_v, 1, p->experts_per_token, p->experts_per_token, BATCH_SIZE);

    HIP_CHECK(hipMemcpy(topk_v, dev_s->topk_v,
                        BATCH_SIZE * p->experts_per_token * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(topk_i, dev_s->topk_i,
                        BATCH_SIZE * p->experts_per_token * sizeof(int),
                        hipMemcpyDeviceToHost));

    for (int device_id = 0; device_id < n_devices; ++device_id) {
      HIP_CHECK(hipSetDevice(device_id));

      int experts_per_device = n_experts / n_devices;
      int expert_start = device_id * experts_per_device;
      int expert_end = expert_start + experts_per_device;
      
      DeviceTransformerWeights *curr_dev_w =
            &dev_transformeres[device_id]->weights;
      RunState *curr_dev_s = &dev_transformeres[device_id]->state;

      if (device_id != 0)
          HIP_CHECK(hipMemcpyPeer(curr_dev_s->t, device_id, dev_s->t, 0,
                                  sizeof(float) * BATCH_SIZE * hidden_dim));
      HIP_CHECK(hipMemset(curr_dev_s->e_agg, 0, BATCH_SIZE * hidden_dim * sizeof(float)));

      int local_ids[BATCH_SIZE * 4] = {-1, -1, -1, -1, 
                                       -1, -1, -1, -1};
      float local_wts[BATCH_SIZE * 4] = {0, 0, 0, 0,
                                         0, 0, 0, 0};
      int n_local[BATCH_SIZE];
      for (int b = 0; b < BATCH_SIZE; ++b) {
        n_local[b] = 0;
        for (int idx = 0; idx < p->experts_per_token; ++idx) {
          int e_global = topk_i[b * p->experts_per_token + idx];
          if (e_global >= expert_start && e_global < expert_end) {
            local_ids[b * 4 + n_local[b]] = e_global - expert_start;  // local index
            local_wts[b * 4 + n_local[b]] = topk_v[b * p->experts_per_token + idx];
            ++n_local[b];
          }
        }
      }

      if (n_local[0] + n_local[1] > 0) {
        // base pointer của layer hiện tại
        __hip_bfloat16 *w1_base =
            curr_dev_w->w_mlp1 +
            1ll * l * experts_per_device * 2 * p->intermediate_dim * hidden_dim;
        __hip_bfloat16 *b1_base =
            curr_dev_w->b_mlp1 +
            1ll * l * experts_per_device * 2 * p->intermediate_dim;

        int4 ids0 = {local_ids[0], local_ids[1], local_ids[2], local_ids[3]};
        int4 ids1 = {local_ids[4], local_ids[5], local_ids[6], local_ids[7]};
        // gate_up_all buffer: [n_local, intermediate_dim]
        getp_mlp1_swiglu_bf16_batch(curr_dev_s->gate_up, curr_dev_s->t, w1_base,
                                    b1_base, hidden_dim, 
                                    p->intermediate_dim, p->experts_per_token, 
                                    ids0, n_local[0],
                                    ids1, n_local[1],
                                    p->swiglu_limit, BATCH_SIZE);

        __hip_bfloat16 *w2_layer_base =
            curr_dev_w->w_mlp2 +
            1ll * l * experts_per_device * hidden_dim * p->intermediate_dim;
        __hip_bfloat16 *b2_layer_base =
            curr_dev_w->b_mlp2 + 1ll * l * experts_per_device * hidden_dim;
        float4 wpack0 = {0.f, 0.f, 0.f, 0.f};
        if (n_local[0] > 0) wpack0.x = local_wts[0];
        if (n_local[0] > 1) wpack0.y = local_wts[1];
        if (n_local[0] > 2) wpack0.z = local_wts[2];
        if (n_local[0] > 3) wpack0.w = local_wts[3];
        float4 wpack1 = {0.f, 0.f, 0.f, 0.f};
        if (n_local[1] > 0) wpack1.x = local_wts[4];
        if (n_local[1] > 1) wpack1.y = local_wts[5];
        if (n_local[1] > 2) wpack1.z = local_wts[6];
        if (n_local[1] > 3) wpack1.w = local_wts[7];

        getp_mlp2_accum_bf16_batch(curr_dev_s->e_agg, curr_dev_s->gate_up,
                                    w2_layer_base, b2_layer_base, p->experts_per_token, 
                                    ids0, n_local[0], wpack0,
                                    ids1, n_local[1], wpack1,
                                    p->intermediate_dim, hidden_dim, BATCH_SIZE);
      }
      if (device_id != 0) {
        HIP_CHECK(hipMemcpyPeer(dev_s->tb2, 0, curr_dev_s->e_agg, device_id,
                                sizeof(float) * BATCH_SIZE * hidden_dim));
        HIP_CHECK(hipSetDevice(0));
        getp_vecadd(dev_s->e_agg, dev_s->tb2, hidden_dim, BATCH_SIZE);
      }
    }

    HIP_CHECK(hipSetDevice(0));
    getp_vecadd(dev_x, dev_s->e_agg, hidden_dim, BATCH_SIZE);
  }


  HIP_CHECK(hipFree(dev_cos_vals));
  HIP_CHECK(hipFree(dev_sin_vals));
  free(topk_v);
  free(topk_i);

  getp_rmsnorm(dev_x, dev_x, dev_w->rms_out_w, BATCH_SIZE, hidden_dim);
  getp_matmul<__hip_bfloat16>(dev_s->logits, dev_x, dev_w->out_bf16,
                              (__hip_bfloat16 *)NULL, hidden_dim,
                              p->vocab_size, BATCH_SIZE);
  float *logits_result = reinterpret_cast<float *>(malloc(sizeof(float) * BATCH_SIZE * p->vocab_size));

  HIP_CHECK(hipMemcpy(logits_result, dev_s->logits, sizeof(float) * BATCH_SIZE * p->vocab_size,
                      hipMemcpyDeviceToHost));
  // HIP_CHECK(hipDeviceSynchronize());
  return logits_result;
}
