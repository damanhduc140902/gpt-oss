#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <malloc.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "collectives.hpp"
#include "getp_state_ext.hpp"
#include "getp_transformer.cpp"
#include "getp_transformer.hpp"
#include "profiler.hpp"

#ifndef GEMV_TILE_N
#define GEMV_TILE_N 512
#endif
#ifndef GEMV_WARPS_PER_BLOCK
#define GEMV_WARPS_PER_BLOCK 8
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
  rmsnorm_kernel<<<gridDim, blockDim, sizeof(float) * (blockDim.x + 1)>>>(
      o, x, weight, dim);
  //HIP_CHECK(hipDeviceSynchronize());
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

__global__ void moe_count_local_kernel(const int *local_ids, const int *n_local,
                                       int B, int K, int E, int *counts) {
  int b = blockIdx.x;
  if (b >= B) return;
  int nl = n_local[b];
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int lid = (k < nl) ? local_ids[(size_t)b * K + k] : -1;
    if (lid >= 0 && lid < E) atomicAdd(counts + lid, 1);
  }
}

__global__ void exclusive_scan_small_kernel(const int *in, int *out, int E) {
  extern __shared__ int smi[];
  int tid = threadIdx.x;
  int v = (tid < E) ? in[tid] : 0;
  smi[tid] = v;
  __syncthreads();
  for (int ofs = 1; ofs < blockDim.x; ofs <<= 1) {
    int t = (tid >= ofs) ? smi[tid - ofs] : 0;
    __syncthreads();
    smi[tid] += t;
    __syncthreads();
  }
  if (tid < E) out[tid] = (tid == 0) ? 0 : smi[tid - 1];
  if (tid == 0) out[E] = smi[E - 1];
}

__global__ void moe_fill_local_pos_kernel(const int *local_ids,
                                          const float *local_wts,
                                          const int *n_local, int B, int K,
                                          int E, const int *offsets,
                                          int *counters, int *tok_idx_out,
                                          float *w_out, int *pair_pos) {
  int b = blockIdx.x;
  if (b >= B) return;
  int nl = n_local[b];
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int lid = (k < nl) ? local_ids[(size_t)b * K + k] : -1;
    if (lid < 0 || lid >= E) {
      if (k < K) pair_pos[(size_t)b * K + k] = -1;
      continue;
    }
    int idx = atomicAdd(counters + lid, 1);
    int pos = offsets[lid] + idx;
    tok_idx_out[pos] = b;
    w_out[pos] = local_wts[(size_t)b * K + k];
    pair_pos[(size_t)b * K + k] = pos;
  }
}

static inline int build_moe_buckets_local_pos(const int *local_ids,
                                              const float *local_wts,
                                              const int *n_local, int B, int K,
                                              int E, int *e_counts,
                                              int *e_offsets, int *tok_idx_out,
                                              float *w_out, int *pair_pos) {
  HIP_CHECK(hipMemset(e_counts, 0, sizeof(int) * E));
  moe_count_local_kernel<<<dim3(B), dim3(128)>>>(local_ids, n_local, B, K, E,
                                                 e_counts);
  exclusive_scan_small_kernel<<<dim3(1), dim3(256), sizeof(int) * 256>>>(
      e_counts, e_offsets, E);
  std::vector<int> h_off(E + 1);
  HIP_CHECK(hipMemcpy(h_off.data(), e_offsets, sizeof(int) * (E + 1),
                      hipMemcpyDeviceToHost));
  int cap_pairs = 0;
  for (int i = 0; i < E; ++i)
    cap_pairs = MAX(cap_pairs, h_off[i + 1] - h_off[i]);
  HIP_CHECK(hipMemset(e_counts, 0, sizeof(int) * E));
  moe_fill_local_pos_kernel<<<dim3(B), dim3(128)>>>(
      local_ids, local_wts, n_local, B, K, E, e_offsets, e_counts, tok_idx_out,
      w_out, pair_pos);
  return cap_pairs;
}

















template <int TILE_N, int WARPS_PER_BLOCK, int TB>
__global__ void mlp1_swiglu_bf16_bucketed_kernel(
    float *__restrict__ gate_up_all, const float *__restrict__ x,
    const __hip_bfloat16 *__restrict__ W1,
    const __hip_bfloat16 *__restrict__ B1, const int *__restrict__ offsets,
    const int *__restrict__ tok_idx, int H, int I, int E, int cap_pairs,
    float swiglu_limit) {
  extern __shared__ float sX[];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int j = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (j >= I) return;
  const int e = blockIdx.y % E;
  const int blk = blockIdx.y / E;
  const int base_all = blk * TB;
  const int start = offsets[e];
  const int end = offsets[e + 1];
  const int n = end - start;
  const int base = start + base_all;
  if (base >= end) return;
  const int tcnt = min(TB, n - base_all);
  float accg[TB], accu[TB];
  for (int t = 0; t < TB; ++t) {
    accg[t] = 0.f;
    accu[t] = 0.f;
  }
  for (int tile = 0; tile < H; tile += TILE_N) {
    const int tlen = min(TILE_N, H - tile);
    for (int tb = 0; tb < tcnt; ++tb) {
      const int b = tok_idx[start + base_all + tb];
      float *dst = sX + tb * TILE_N;
      const float *src = x + (size_t)b * H + tile;
      for (int t = warp * warpSize * 4 + lane * 4; t < tlen;
           t += WARPS_PER_BLOCK * warpSize * 4) {
        if (t + 3 < tlen)
          reinterpret_cast<float4 &>(dst[t]) =
              *reinterpret_cast<const float4 *>(&src[t]);
        else {
          for (int k = 0; k < 4 && t + k < tlen; ++k) dst[t + k] = src[t + k];
        }
      }
    }
    __syncthreads();
    const size_t row_stride = (size_t)H;
    const size_t eb = (size_t)e * (size_t)(2 * I) * (size_t)H;
    const __hip_bfloat16 *wg =
        W1 + eb + (size_t)(2 * j + 0) * row_stride + tile;
    const __hip_bfloat16 *wu =
        W1 + eb + (size_t)(2 * j + 1) * row_stride + tile;
    const uint4 *wg8 = reinterpret_cast<const uint4 *>(wg);
    const uint4 *wu8 = reinterpret_cast<const uint4 *>(wu);
    const int it8 = tlen >> 3;
#pragma unroll 4
    for (int k8 = lane; k8 < it8; k8 += warpSize) {
      uint4 ag = wg8[k8], au = wu8[k8];
      unsigned short g01 = (unsigned short)(ag.x & 0xFFFF),
                     g02 = (unsigned short)(ag.x >> 16);
      unsigned short g11 = (unsigned short)(ag.y & 0xFFFF),
                     g12 = (unsigned short)(ag.y >> 16);
      unsigned short g21 = (unsigned short)(ag.z & 0xFFFF),
                     g22 = (unsigned short)(ag.z >> 16);
      unsigned short g31 = (unsigned short)(ag.w & 0xFFFF),
                     g32 = (unsigned short)(ag.w >> 16);
      unsigned short u01 = (unsigned short)(au.x & 0xFFFF),
                     u02 = (unsigned short)(au.x >> 16);
      unsigned short u11 = (unsigned short)(au.y & 0xFFFF),
                     u12 = (unsigned short)(au.y >> 16);
      unsigned short u21 = (unsigned short)(au.z & 0xFFFF),
                     u22 = (unsigned short)(au.z >> 16);
      unsigned short u31 = (unsigned short)(au.w & 0xFFFF),
                     u32 = (unsigned short)(au.w >> 16);
      float4 xb0, xb1;
      for (int tb = 0; tb < tcnt; ++tb) {
        const float4 *x4 = reinterpret_cast<const float4 *>(sX + tb * TILE_N);
        xb0 = x4[(k8 << 1) + 0];
        xb1 = x4[(k8 << 1) + 1];
        float g = accg[tb], u = accu[tb];
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
        accg[tb] = g;
        accu[tb] = u;
      }
    }
    for (int k = (it8 << 3) + lane; k < tlen; k += warpSize) {
      for (int tb = 0; tb < tcnt; ++tb) {
        const float *xs = sX + tb * TILE_N;
        accg[tb] += __bfloat162float(wg[k]) * xs[k];
        accu[tb] += __bfloat162float(wu[k]) * xs[k];
      }
    }
    __syncthreads();
  }
  for (int tb = 0; tb < TB; ++tb) {
    if (tb >= tcnt) break;
    float g = accg[tb], u = accu[tb];
    for (int off = warpSize >> 1; off > 0; off >>= 1) {
      g += __shfl_down(g, off);
      u += __shfl_down(u, off);
    }
    if (lane == 0) {
      const size_t b_off = (size_t)e * (size_t)(2 * I);
      float bg = B1 ? __bfloat162float(B1[b_off + (size_t)(2 * j + 0)]) : 0.f;
      float bu = B1 ? __bfloat162float(B1[b_off + (size_t)(2 * j + 1)]) : 0.f;
      g += bg;
      u += bu;
      if (g > swiglu_limit) g = swiglu_limit;
      if (u > swiglu_limit) u = swiglu_limit;
      if (u < -swiglu_limit) u = -swiglu_limit;
      const float a = 1.702f;
      float s = 1.f / (1.f + expf(-a * g));
      float val = (g * s) * (u + 1.f);
      int idx = start + base_all + tb;
      gate_up_all[(size_t)idx * (size_t)I + j] = val;
    }
  }
}
static inline void launch_mlp1_swiglu_bf16_bucketed(
    float *gate_up_all, const float *x, const __hip_bfloat16 *w1_layer,
    const __hip_bfloat16 *b1_layer, const int *offsets, const int *tok_idx,
    int H, int I, int E, int cap_pairs, float swiglu_limit) {
      PROFILE_FUNCTION();
  constexpr int WARPS = GEMV_WARPS_PER_BLOCK;
  constexpr int TILE = GEMV_TILE_N;
  constexpr int TB = 8;
  dim3 blk(64, WARPS);
  int by = (cap_pairs + TB - 1) / TB;
  dim3 grd((I + WARPS - 1) / WARPS, E * by);
  size_t shmem = (size_t)TILE * TB * sizeof(float);
  mlp1_swiglu_bf16_bucketed_kernel<TILE, WARPS, TB>
      <<<grd, blk, shmem>>>(gate_up_all, x, w1_layer, b1_layer, offsets,
                            tok_idx, H, I, E, cap_pairs, swiglu_limit);
  //HIP_CHECK(hipDeviceSynchronize());
}
template <int TILE_N, int WARPS_PER_BLOCK, int TB>
__global__ void mlp2_partial_bf16_bucketed_kernel(
    float *__restrict__ z_partial, const float *__restrict__ gate_up_all,
    const __hip_bfloat16 *__restrict__ W2,
    const __hip_bfloat16 *__restrict__ B2,
    const float *__restrict__ w_by_bucket, const int *__restrict__ offsets,
    const int *__restrict__ tok_idx, int I, int H, int E, int cap_pairs) {
  extern __shared__ float sX[];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int row = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (row >= H) return;
  const int e = blockIdx.y % E;
  const int blk = blockIdx.y / E;
  const int base_all = blk * TB;
  const int start = offsets[e];
  const int end = offsets[e + 1];
  const int n = end - start;
  const int base = start + base_all;
  if (base >= end) return;
  const int tcnt = min(TB, n - base_all);
  float acc[TB];
  for (int t = 0; t < TB; ++t) acc[t] = 0.f;
  for (int tile = 0; tile < I; tile += TILE_N) {
    const int tlen = min(TILE_N, I - tile);
    for (int tb = 0; tb < tcnt; ++tb) {
      const float *src =
          gate_up_all + (size_t)(start + base_all + tb) * (size_t)I + tile;
      float *dst = sX + tb * TILE_N;
      for (int t = warp * warpSize * 4 + lane * 4; t < tlen;
           t += WARPS_PER_BLOCK * warpSize * 4) {
        if (t + 3 < tlen)
          reinterpret_cast<float4 &>(dst[t]) =
              *reinterpret_cast<const float4 *>(&src[t]);
        else {
          for (int k = 0; k < 4 && t + k < tlen; ++k) dst[t + k] = src[t + k];
        }
      }
    }
    __syncthreads();
    const size_t eb = (size_t)e * (size_t)H * (size_t)I;
    const __hip_bfloat16 *wrow = W2 + eb + (size_t)row * (size_t)I + tile;
    const uint4 *w8 = reinterpret_cast<const uint4 *>(wrow);
    const int it8 = tlen >> 3;
#pragma unroll 4
    for (int k8 = lane; k8 < it8; k8 += warpSize) {
      uint4 wb = w8[k8];
      unsigned short w01 = (unsigned short)(wb.x & 0xFFFF),
                     w02 = (unsigned short)(wb.x >> 16);
      unsigned short w11 = (unsigned short)(wb.y & 0xFFFF),
                     w12 = (unsigned short)(wb.y >> 16);
      unsigned short w21 = (unsigned short)(wb.z & 0xFFFF),
                     w22 = (unsigned short)(wb.z >> 16);
      unsigned short w31 = (unsigned short)(wb.w & 0xFFFF),
                     w32 = (unsigned short)(wb.w >> 16);
      for (int tb = 0; tb < tcnt; ++tb) {
        const float4 *x4 = reinterpret_cast<const float4 *>(sX + tb * TILE_N);
        float4 xb0 = x4[(k8 << 1) + 0], xb1 = x4[(k8 << 1) + 1];
        float a = acc[tb];
        a = fmaf(bf16bits_to_f32(w01), xb0.x, a);
        a = fmaf(bf16bits_to_f32(w02), xb0.y, a);
        a = fmaf(bf16bits_to_f32(w11), xb0.z, a);
        a = fmaf(bf16bits_to_f32(w12), xb0.w, a);
        a = fmaf(bf16bits_to_f32(w21), xb1.x, a);
        a = fmaf(bf16bits_to_f32(w22), xb1.y, a);
        a = fmaf(bf16bits_to_f32(w31), xb1.z, a);
        a = fmaf(bf16bits_to_f32(w32), xb1.w, a);
        acc[tb] = a;
      }
    }
    for (int k = (it8 << 3) + lane; k < tlen; k += warpSize) {
      for (int tb = 0; tb < tcnt; ++tb)
        acc[tb] += __bfloat162float(wrow[k]) * (sX[tb * TILE_N + k]);
    }
    __syncthreads();
  }
  for (int tb = 0; tb < TB; ++tb) {
    if (tb < tcnt)
      for (int off = warpSize >> 1; off > 0; off >>= 1)
        acc[tb] += __shfl_down(acc[tb], off);
  }
  if (lane == 0) {
    for (int tb = 0; tb < tcnt; ++tb) {
      float bias = B2 ? __bfloat162float(B2[(size_t)e * (size_t)H + row]) : 0.f;
      float val = acc[tb] + bias;
      float w = w_by_bucket[start + base_all + tb];
      int pos = start + base_all + tb;
      z_partial[(size_t)pos * (size_t)H + row] = w * val;
    }
  }
}

static inline void launch_mlp2_partial_bf16_bucketed(
    float *z_partial, const float *gate_up_all, const __hip_bfloat16 *w2_layer,
    const __hip_bfloat16 *b2_layer, const float *w_by_bucket,
    const int *offsets, const int *tok_idx, int I, int H, int E,
    int cap_pairs) {
      PROFILE_FUNCTION();
  constexpr int WARPS = GEMV_WARPS_PER_BLOCK;
  constexpr int TILE = GEMV_TILE_N;
  constexpr int TB = 8;
  dim3 blk(64, WARPS);
  int by = (cap_pairs + TB - 1) / TB;
  dim3 grd((H + WARPS - 1) / WARPS, E * by);
  size_t shmem = (size_t)TILE * TB * sizeof(float);
  mlp2_partial_bf16_bucketed_kernel<TILE, WARPS, TB>
      <<<grd, blk, shmem>>>(z_partial, gate_up_all, w2_layer, b2_layer,
                            w_by_bucket, offsets, tok_idx, I, H, E, cap_pairs);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void moe_gather_pairs_kernel(float *__restrict__ e_agg,
                                        const float *__restrict__ z_partial,
                                        const int *__restrict__ pair_pos, int H,
                                        int K, int B) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (row >= H || b >= B) return;
  int base = b * K;
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    int pos = pair_pos[base + k];
    if (k >= K) break;
    if (pos >= 0) s += z_partial[(size_t)pos * (size_t)H + row];
  }
  e_agg[(size_t)b * (size_t)H + row] += s;
}
static inline void moe_gather_pairs(float *e_agg, const float *z_partial,
                                    const int *pair_pos, int H, int K, int B) {
  PROFILE_FUNCTION();
  dim3 blk(256), grd((H + blk.x - 1) / blk.x, B);
  moe_gather_pairs_kernel<<<grd, blk>>>(e_agg, z_partial, pair_pos, H, K, B);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void matmul_qkv_fused_kernel(
    float *__restrict__ q_out, float *__restrict__ k_out,
    float *__restrict__ v_out, const float *__restrict__ x,
    const __hip_bfloat16 *__restrict__ W, const __hip_bfloat16 *__restrict__ B,
    int n, int q_len, int k_len, int v_len, int batch_size) {
  constexpr int BLOCK_SIZE = 256;
  // Matrix
  const int M = batch_size;
  const int N = q_len + k_len + v_len;
  const int K = n;
  // Block tile
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 8;
  // Wave tile
  constexpr int WM = 16;
  constexpr int WN = 16;

  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 4;
  using mfloat4 =
      __attribute__((__vector_size__(MFMA_K * sizeof(float)))) float;

  const int waveIdx = threadIdx.x / warpSize;
  const int xInBlockTile = waveIdx % (BM / WM);
  const int yInBlockTile = waveIdx / (BM / WM);
  const int laneIdx = threadIdx.x % warpSize;
  const int xInMFMA = laneIdx % (MFMA_M);
  const int yInMFMA = laneIdx / (MFMA_M);

  const int loadXIdx_x = threadIdx.x % (BK);
  const int loadXIdx_y = threadIdx.x / (BK);
  const int loadWIdx_x = threadIdx.x % (BK);
  const int loadWIdx_y = threadIdx.x / (BK);
  constexpr int strideX = BLOCK_SIZE / (BK);
  constexpr int strideW = BLOCK_SIZE / (BK);

  assert(blockDim.x == BLOCK_SIZE);

  __shared__ float sx[BK][BM];
  __shared__ float sw[BK][BN];

  mfloat4 dmn = {0};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (int offset = 0; offset < BM; offset += strideX) {
      int index_x = bkIdx + loadXIdx_x;
      int index_y = BM * blockIdx.y + loadXIdx_y + offset;
      sx[index_x % BK][index_y % BM] =
          index_x < K && index_y < M ? x[index_y * K + index_x] : 0;
    }
    for (int offset = 0; offset < BN; offset += strideW) {
      int index_x = bkIdx + loadWIdx_x;
      int index_y = BN * blockIdx.x + loadWIdx_y + offset;
      sw[index_x % BK][index_y % BN] =
          index_x < K && index_y < N
              ? __bfloat162float(W[index_y * K + index_x])
              : 0;
    }
    __syncthreads();
    for (int k = 0; k < BK; k += MFMA_K) {
      float amk = sx[k + yInMFMA][yInBlockTile * WM + xInMFMA];
      float bkn = sw[k + yInMFMA][xInBlockTile * WN + xInMFMA];
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    const int xInD = laneIdx % MFMA_N;
    const int yInD = MFMA_K * (laneIdx / MFMA_N) + i;
    const int xInOutput = blockIdx.x * BN + xInBlockTile * WN + xInD;
    const int yInOutput = blockIdx.y * BM + yInBlockTile * WM + yInD;
    float result = dmn[i];
    if (yInOutput < M && xInOutput < N) {
      if (B) result += __bfloat162float(B[xInOutput]);
      float *dst;
      int outIdx;
      if (xInOutput < q_len) {
        dst = q_out + yInOutput * q_len;
        outIdx = xInOutput;
      } else if (xInOutput < q_len + k_len) {
        dst = k_out + yInOutput * k_len;
        outIdx = xInOutput - q_len;
      } else {
        dst = v_out + yInOutput * v_len;
        outIdx = xInOutput - q_len - k_len;
      }
      dst[outIdx] = result;
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

  const int M = batch_size;
  const int N = d_total;
  const int K = n;

  const int BN = 32;
  const int BM = 32;

  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  matmul_qkv_fused_kernel<<<grid, block>>>(q, k, v, x, w_qkv_bf16, b_qkv_bf16,
                                           n, q_len, k_len, v_len, batch_size);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void __launch_bounds__(256)
    matmul_kernel(float *xout, float *x, __hip_bfloat16 *w, __hip_bfloat16 *b,
                  int M, int N, int K) {
  constexpr int BLOCK_SIZE = 256;
  // Block tile
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  // Wave tile
  constexpr int WM = 16;
  constexpr int WN = 16;

  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 4;
  using mfloat4 =
      __attribute__((__vector_size__(MFMA_K * sizeof(float)))) float;

  const int waveIdx = threadIdx.x / warpSize;
  const int xInBlockTile = waveIdx % (BM / WM);
  const int yInBlockTile = waveIdx / (BM / WM);
  const int laneIdx = threadIdx.x % warpSize;
  const int xInMFMA = laneIdx % (MFMA_M);
  const int yInMFMA = laneIdx / (MFMA_M);

  const int loadXIdx_x = threadIdx.x % (BK / 4);
  const int loadXIdx_y = threadIdx.x / (BK / 4);
  const int loadWIdx_x = threadIdx.x % (BK / 8);
  const int loadWIdx_y = threadIdx.x / (BK / 8);
  constexpr int strideX = BLOCK_SIZE / (BK / 4);
  constexpr int strideW = BLOCK_SIZE / (BK / 8);

  assert(blockDim.x == BLOCK_SIZE);

  __shared__ float sx[BK][BM];
  __shared__ float sw[BK][BN];

  mfloat4 dmn = {0};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (int offset = 0; offset < BM; offset += strideX) {
      if (loadXIdx_y + offset >= BM) break;
      int index_x = bkIdx + 4 * loadXIdx_x;
      int index_y = BM * blockIdx.y + loadXIdx_y + offset;
      float4 tmp = index_x < K && index_y < M
                       ? *reinterpret_cast<float4 *>(&x[index_y * K + index_x])
                       : make_float4(0, 0, 0, 0);
      sx[4 * loadXIdx_x + 0][loadXIdx_y + offset] = tmp.x;
      sx[4 * loadXIdx_x + 1][loadXIdx_y + offset] = tmp.y;
      sx[4 * loadXIdx_x + 2][loadXIdx_y + offset] = tmp.z;
      sx[4 * loadXIdx_x + 3][loadXIdx_y + offset] = tmp.w;
    }
    for (int offset = 0; offset < BN; offset += strideW) {
      if (loadWIdx_y + offset >= BN) break;
      int index_x = bkIdx + 8 * loadWIdx_x;
      int index_y = BN * blockIdx.x + loadWIdx_y + offset;
      uint4 tmp = index_x < K && index_y < N
                      ? *reinterpret_cast<uint4 *>(&w[index_y * K + index_x])
                      : make_uint4(0, 0, 0, 0);
      sw[8 * loadWIdx_x + 0][loadWIdx_y + offset] =
          bf16bits_to_f32((unsigned short)(tmp.x & 0xFFFF));
      sw[8 * loadWIdx_x + 1][loadWIdx_y + offset] =
          bf16bits_to_f32((unsigned short)(tmp.x >> 16));
      sw[8 * loadWIdx_x + 2][loadWIdx_y + offset] =
          bf16bits_to_f32((unsigned short)(tmp.y & 0xFFFF));
      sw[8 * loadWIdx_x + 3][loadWIdx_y + offset] =
          bf16bits_to_f32((unsigned short)(tmp.y >> 16));
      sw[8 * loadWIdx_x + 4][loadWIdx_y + offset] =
          bf16bits_to_f32((unsigned short)(tmp.z & 0xFFFF));
      sw[8 * loadWIdx_x + 5][loadWIdx_y + offset] =
          bf16bits_to_f32((unsigned short)(tmp.z >> 16));
      sw[8 * loadWIdx_x + 6][loadWIdx_y + offset] =
          bf16bits_to_f32((unsigned short)(tmp.w & 0xFFFF));
      sw[8 * loadWIdx_x + 7][loadWIdx_y + offset] =
          bf16bits_to_f32((unsigned short)(tmp.w >> 16));
    }
    __syncthreads();
    for (int k = 0; k < BK; k += MFMA_K) {
      float amk = sx[k + yInMFMA][yInBlockTile * WM + xInMFMA];
      float bkn = sw[k + yInMFMA][xInBlockTile * WN + xInMFMA];
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    int xInD = laneIdx % MFMA_N;
    int yInD = MFMA_K * (laneIdx / MFMA_N) + i;
    int xInOutput = blockIdx.x * BN + xInBlockTile * WN + xInD;
    int yInOutput = blockIdx.y * BM + yInBlockTile * WM + yInD;
    float result = dmn[i];
    if (yInOutput < M && xInOutput < N) {
      if (b) result += __bfloat162float(b[xInOutput]);
      xout[yInOutput * N + xInOutput] = result;
    }
  }
}
template <typename T>
void getp_matmul(float *xout, float *x, T *w, T *b, int n, int d,
                 int batch_size) {
  PROFILE_FUNCTION();
  const int M = batch_size;
  const int N = d;
  const int K = n;

  const int BM = 32;
  const int BN = 32;

  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  matmul_kernel<<<grid, block>>>(xout, x, w, b, M, N, K);
  //HIP_CHECK(hipDeviceSynchronize());
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
                          float *sin_out, float *inv_freq) {
  PROFILE_FUNCTION();
  int d_half = head_dim / 2;
  float concentration =
      scaling_factor > 1.0f ? 0.1f * logf(scaling_factor) + 1.0f : 1.0f;
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
    compute_cos_sin_kernel<<<gridDim, blockDim>>>(cos_out, sin_out, inv_freq,
                                                  concentration, pos, d_half);
  }
  //HIP_CHECK(hipDeviceSynchronize());
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
void getp_apply_rotary_emb(float *x, float *cos, float *sin, int n_heads,
                           int head_dim, int batch_size) {
  PROFILE_FUNCTION();
  dim3 blockDim(16, 16);
  dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x,
               (n_heads + blockDim.y - 1) / blockDim.y, batch_size);
  apply_rotary_emb_kernel<<<gridDim, blockDim>>>(x, cos, sin, n_heads,
                                                 head_dim);
  //HIP_CHECK(hipDeviceSynchronize());
}

#ifndef GETP_ROUTER_TOPK_MAXK
#define GETP_ROUTER_TOPK_MAXK 4
#endif

__global__ void router_topk_softmax_batch_kernel(
    const float* __restrict__ router_score,
    int n_experts,
    int experts_per_token,
    float* __restrict__ topk_v_out,
    int* __restrict__ topk_i_out) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int K = experts_per_token;

  router_score += (size_t)b * n_experts;
  topk_v_out += (size_t)b * K;
  topk_i_out += (size_t)b * K;

  extern __shared__ unsigned char smem_raw[];
  float* scores = reinterpret_cast<float*>(smem_raw);
  float* valbuf = scores + n_experts;
  int*   idxbuf = reinterpret_cast<int*>(valbuf + blockDim.x);
  float* topv   = reinterpret_cast<float*>(idxbuf + blockDim.x);
  int*   topi   = reinterpret_cast<int*>(topv + GETP_ROUTER_TOPK_MAXK);

  for (int i = tid; i < n_experts; i += blockDim.x) scores[i] = router_score[i];
  __syncthreads();

  for (int sel = 0; sel < K; ++sel) {
    float best = -INFINITY;
    int   besti = n_experts;
    for (int i = tid; i < n_experts; i += blockDim.x) {
      float v = scores[i];
      float thr = 1e-6f * fmaxf(fabsf(v), fabsf(best));
      if (v > best + thr || (fabsf(v - best) <= thr && i < besti)) {
        best = v; besti = i;
      }
    }
    valbuf[tid] = best;
    idxbuf[tid] = besti;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        float vr = valbuf[tid + stride], vl = valbuf[tid];
        int   ir = idxbuf[tid + stride], il = idxbuf[tid];
        float thr = 1e-6f * fmaxf(fabsf(vr), fabsf(vl));
        bool take_r = (vr > vl + thr) || (fabsf(vr - vl) <= thr && ir < il);
        if (take_r) { valbuf[tid] = vr; idxbuf[tid] = ir; }
      }
      __syncthreads();
    }
    if (tid == 0) {
      topv[sel] = valbuf[0];
      topi[sel] = idxbuf[0];
      scores[topi[sel]] = -INFINITY;
    }
    __syncthreads();
  }

  if (tid == 0) {
    float m = topv[0];
    for (int i = 1; i < K; ++i) if (topv[i] > m) m = topv[i];
    float e[GETP_ROUTER_TOPK_MAXK]; float s = 0.f;
    for (int i = 0; i < K; ++i) { e[i] = expf(topv[i] - m); s += e[i]; }
    float invs = 1.f / s;
    for (int i = 0; i < K; ++i) { topk_v_out[i] = e[i] * invs; topk_i_out[i] = topi[i]; }
  }
}


static inline void getp_router_topk_softmax_batch(
    const float *router_score, int n_experts, int experts_per_token,
    float *topk_v_out, int *topk_i_out, int batch_size) {
  PROFILE_FUNCTION();
  const int BLK = 1024;
  const dim3 block(BLK);
  const dim3 grid(batch_size);
  const size_t shmem = (size_t)n_experts * sizeof(float) +
                       (size_t)BLK * (sizeof(float) + sizeof(int)) +
                       GETP_ROUTER_TOPK_MAXK * (sizeof(float) + sizeof(int));
  router_topk_softmax_batch_kernel<<<grid, block, shmem>>>(
      router_score, n_experts, experts_per_token, topk_v_out, topk_i_out);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void map_global_to_local_batch_kernel(
    const int *__restrict__ topk_i,    // [B,K]
    const float *__restrict__ topk_v,  // [B,K]
    int *__restrict__ local_ids,       // [B,K]
    float *__restrict__ local_wts,     // [B,K]
    int *__restrict__ n_local,         // [B]
    int K, int expert_start, int expert_end, int B) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  int base = b * K, cnt = 0;
  for (int i = 0; i < K; ++i) {
    int eg = topk_i[base + i];
    float w = topk_v[base + i];
    if (eg >= expert_start && eg < expert_end) {
      local_ids[base + cnt] = eg - expert_start;
      local_wts[base + cnt] = w;
      ++cnt;
    }
  }
  for (int i = cnt; i < K; ++i) {
    local_ids[base + i] = -1;
    local_wts[base + i] = 0.f;
  }
  n_local[b] = cnt;
}

static inline void getp_map_global_to_local_batch(
    const int *topk_i, const float *topk_v, int *local_ids, float *local_wts,
    int *n_local, int K, int expert_start, int expert_end, int B) {
  PROFILE_FUNCTION();
  const int BLK = 128, GRD = (B + BLK - 1) / BLK;
  map_global_to_local_batch_kernel<<<GRD, BLK>>>(topk_i, topk_v, local_ids,
                                                 local_wts, n_local, K,
                                                 expert_start, expert_end, B);
  //HIP_CHECK(hipDeviceSynchronize());
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
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void flash_attn_decode_kernel(
    float *__restrict__ tb, const float *__restrict__ key_cache,
    const float *__restrict__ value_cache, const float *__restrict__ q,
    const float *__restrict__ attn_sinks, int head_dim, int n_attn_heads,
    int n_kv_heads, int pos, int seq_len, int sliding_window, int apply_mask,
    int batch_size, int kv_dim, int kv_mul, float inv_sqrt_d) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int i = threadIdx.x;

  __shared__ float sbuf[2];

  // Pointers
  const float *qbh =
      q + (size_t)b * n_attn_heads * head_dim + (size_t)h * head_dim;
  float *obh = tb + (size_t)b * n_attn_heads * head_dim + (size_t)h * head_dim;

  const int kv_h = h / kv_mul;

  float qi = (i < head_dim) ? qbh[i] : 0.0f;
  float out_i = 0.0f;

  float m = -INFINITY;
  float l = 0.0f;

  int t_start = 0;
  if (apply_mask && sliding_window > 0) {
    t_start = MAX(0, pos - sliding_window + 1);
  }

  for (int t = t_start; t <= pos; ++t) {
    float k_i = 0.f;
    if (i < head_dim) {
      const float *kptr = key_cache + (size_t)t * batch_size * kv_dim +
                          (size_t)b * kv_dim + (size_t)kv_h * head_dim + i;
      k_i = *kptr;
    }
    float part = qi * k_i;
    float sum = warp_sum_f32(part);

    if (threadIdx.x == 0) {
      float s = sum * inv_sqrt_d;
      if (apply_mask && sliding_window > 0) {
        if ((pos - t) >= sliding_window) s = -INFINITY;
      }
      float m_new = fmaxf(m, s);
      float alpha = __expf(m - m_new);
      float e = __expf(s - m_new);
      l = l * alpha + e;
      m = m_new;
      sbuf[0] = e;
      sbuf[1] = alpha;
    }
    __syncthreads();

    float e = sbuf[0];
    float alpha = sbuf[1];

    if (i < head_dim) {
      const float *vptr = value_cache + (size_t)t * batch_size * kv_dim +
                          (size_t)b * kv_dim + (size_t)kv_h * head_dim + i;
      out_i = alpha * out_i + e * (*vptr);
    }
    __syncthreads();
  }

  // Attention sink
  if (threadIdx.x == 0) {
    float s_sink = attn_sinks[h];
    float m_new = fmaxf(m, s_sink);
    float alpha = __expf(m - m_new);
    float e = __expf(s_sink - m_new);
    l = l * alpha + e;
    m = m_new;
    sbuf[0] = alpha;
    sbuf[1] = l;
  }
  __syncthreads();

  float alpha_sink = sbuf[0];
  float l_final = sbuf[1];

  if (i < head_dim) {
    out_i = (alpha_sink * out_i) / l_final;
    obh[i] = out_i;
  }
}
static inline void getp_flash_attn_decode(
    float *tb, const float *key_cache_layer, const float *value_cache_layer,
    const float *q, const float *attn_sinks_layer,  // attn_sinks + l*n_heads
    int head_dim, int n_attn_heads, int n_kv_heads, int pos, int seq_len,
    int sliding_window, int layer_id, int batch_size) {
  PROFILE_FUNCTION();
  const int kv_dim = head_dim * n_kv_heads;
  const int kv_mul = n_attn_heads / n_kv_heads;
  const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
  const int apply_mask = (sliding_window > 0 && ((layer_id & 1) == 0)) ? 1 : 0;

  dim3 block(64);
  dim3 grid(n_attn_heads, batch_size);
  flash_attn_decode_kernel<<<grid, block>>>(
      tb, key_cache_layer, value_cache_layer, q, attn_sinks_layer, head_dim,
      n_attn_heads, n_kv_heads, pos, seq_len, sliding_window, apply_mask,
      batch_size, kv_dim, kv_mul, inv_sqrt_d);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void gather_embedding_kernel(float *__restrict__ x,
                                        const float *__restrict__ table,
                                        const int *__restrict__ tok, int H) {
  int b = blockIdx.x;
  int tid = threadIdx.x;
  int id = tok[b];
  const float *src = table + (size_t)id * H;
  float *dst = x + (size_t)b * H;
  for (int j = tid; j < H; j += blockDim.x) dst[j] = src[j];
}
static inline void getp_gather_embedding(float *x, const float *table,
                                         const int *tok, int H, int B) {
  PROFILE_FUNCTION();
  dim3 block(256), grid(B);
  gather_embedding_kernel<<<grid, block>>>(x, table, tok, H);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void argmax_rows_kernel(const float *__restrict__ logits, int V,
                                   int *__restrict__ out) {
  __shared__ float smax[1024];
  __shared__ int sidx[1024];
  int b = blockIdx.x;
  int tid = threadIdx.x;
  const float *row = logits + (size_t)b * V;
  float mv = -INFINITY;
  int mi = 0;
  for (int i = tid; i < V; i += blockDim.x) {
    float v = row[i];
    if (v > mv) {
      mv = v;
      mi = i;
    }
  }
  smax[tid] = mv;
  sidx[tid] = mi;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (smax[tid + s] > smax[tid]) {
        smax[tid] = smax[tid + s];
        sidx[tid] = sidx[tid + s];
      }
    }
    __syncthreads();
  }
  if (tid == 0) out[b] = sidx[0];
}
static inline void getp_argmax_rows(const float *logits, int V, int *out,
                                    int B) {
  PROFILE_FUNCTION();
  dim3 grid(B), block(1024);
  argmax_rows_kernel<<<grid, block>>>(logits, V, out);
  //HIP_CHECK(hipDeviceSynchronize());
}

int *getp_forward(Transformer * /*transformer*/,
                  DeviceTransformer **dev_transformeres, GPUWorker *worker,
                  int token[], int pos) {
  PROFILE_FUNCTION();

  int device_index = worker->device_index;
  RunStateExt *ext = ext_get(device_index);

  Config *p = &dev_transformeres[device_index]->config;
  DeviceTransformerWeights *dev_w = &dev_transformeres[device_index]->weights;
  RunState *dev_s = &dev_transformeres[device_index]->state;

  float *dev_x = dev_s->x;
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int n_experts = p->n_experts;

  HIP_CHECK(hipSetDevice(device_index));

  HIP_CHECK(hipMemcpy(dev_s->topk_i, token, sizeof(int) * BATCH_SIZE,
                      hipMemcpyHostToDevice));
  getp_gather_embedding(dev_x, dev_w->token_embedding_table, dev_s->topk_i,
                        hidden_dim, BATCH_SIZE);

  float *dev_cos_vals = dev_s->mlp1_out;
  float *dev_sin_vals = dev_s->gate;
  float *dev_inv_freq = dev_s->up;
  float ntk_beta = 32.0f, ntk_alpha = 1.0f;
  getp_compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                       p->initial_context_length, ntk_beta, ntk_alpha,
                       dev_cos_vals, dev_sin_vals, dev_inv_freq);

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_attn_w + 1ll * l * hidden_dim,
                 BATCH_SIZE, hidden_dim);

    int loff = l * p->seq_len * BATCH_SIZE * kv_dim;
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

    {
      int loff2 = l * p->seq_len * BATCH_SIZE * (p->head_dim * p->n_kv_heads);
      float *k_layer = dev_s->key_cache + loff2;
      float *v_layer = dev_s->value_cache + loff2;
      float *q_ptr = dev_s->q;
      const float *attn_sinks_layer =
          dev_w->attn_sinks + (size_t)l * p->n_attn_heads;

      getp_flash_attn_decode(dev_s->tb, k_layer, v_layer, q_ptr,
                             attn_sinks_layer, head_dim, p->n_attn_heads,
                             p->n_kv_heads, pos, p->seq_len, p->sliding_window,
                             (int)l, BATCH_SIZE);
    }

    __hip_bfloat16 *dev_w_o =
        dev_w->w_o_bf16 + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    __hip_bfloat16 *dev_b_o = dev_w->b_o_bf16 + 1ll * l * hidden_dim;
    getp_matmul<__hip_bfloat16>(dev_s->tb2, dev_s->tb, dev_w_o, dev_b_o,
                                head_dim * p->n_attn_heads, hidden_dim,
                                BATCH_SIZE);

    getp_vecadd(dev_x, dev_s->tb2, hidden_dim, BATCH_SIZE);

    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim,
                 BATCH_SIZE, hidden_dim);

    __hip_bfloat16 *dev_w_router =
        dev_w->w_router_bf16 + 1ll * l * hidden_dim * n_experts;
    __hip_bfloat16 *dev_b_router = dev_w->b_router_bf16 + 1ll * l * n_experts;
    getp_matmul<__hip_bfloat16>(dev_s->router_score, dev_s->t, dev_w_router,
                                dev_b_router, hidden_dim, n_experts,
                                BATCH_SIZE);

    getp_router_topk_softmax_batch(dev_s->router_score, n_experts,
                                   p->experts_per_token, dev_s->topk_v,
                                   dev_s->topk_i, BATCH_SIZE);

    getp_map_global_to_local_batch(dev_s->topk_i, dev_s->topk_v, ext->local_ids,
                                   ext->local_wts, ext->n_local,
                                   p->experts_per_token, worker->expert_start,
                                   worker->expert_end, BATCH_SIZE);

    int experts_per_device = worker->expert_end - worker->expert_start;

    int cap_pairs = build_moe_buckets_local_pos(
        ext->local_ids, ext->local_wts, ext->n_local, BATCH_SIZE,
        p->experts_per_token, experts_per_device, ext->e_counts, ext->e_offsets,
        ext->e_dev, ext->w_dev, ext->pair_pos);

    HIP_CHECK(hipMemset(dev_s->e_agg, 0,
                        (size_t)BATCH_SIZE * hidden_dim * sizeof(float)));

    __hip_bfloat16 *w1_base = dev_w->w_mlp1 + 1ll * l * experts_per_device * 2 *
                                                  p->intermediate_dim *
                                                  hidden_dim;
    __hip_bfloat16 *b1_base =
        dev_w->b_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim;

    launch_mlp1_swiglu_bf16_bucketed(dev_s->gate_up, dev_s->t, w1_base, b1_base,
                                     ext->e_offsets, ext->e_dev, hidden_dim,
                                     p->intermediate_dim, experts_per_device,
                                     cap_pairs, p->swiglu_limit);

    __hip_bfloat16 *w2_base = dev_w->w_mlp2 + 1ll * l * experts_per_device *
                                                  hidden_dim *
                                                  p->intermediate_dim;
    __hip_bfloat16 *b2_base =
        dev_w->b_mlp2 + 1ll * l * experts_per_device * hidden_dim;

    launch_mlp2_partial_bf16_bucketed(
        ext->z_partial, dev_s->gate_up, w2_base, b2_base, ext->w_dev,
        ext->e_offsets, ext->e_dev, p->intermediate_dim, hidden_dim,
        experts_per_device, cap_pairs);

    moe_gather_pairs(dev_s->e_agg, ext->z_partial, ext->pair_pos, hidden_dim,
                     p->experts_per_token, BATCH_SIZE);

    getp_vecadd(dev_x, dev_s->e_agg, hidden_dim, BATCH_SIZE);
  }

  getp_rmsnorm(dev_x, dev_x, dev_w->rms_out_w, BATCH_SIZE, hidden_dim);
  getp_matmul<__hip_bfloat16>(dev_s->logits, dev_x, dev_w->out_bf16,
                              (__hip_bfloat16 *)NULL, hidden_dim, p->vocab_size,
                              BATCH_SIZE);

  getp_argmax_rows(dev_s->logits, p->vocab_size, dev_s->topk_i, BATCH_SIZE);

  int *next_host = (int *)malloc(sizeof(int) * BATCH_SIZE);
  HIP_CHECK(hipMemcpy(next_host, dev_s->topk_i, sizeof(int) * BATCH_SIZE,
                      hipMemcpyDeviceToHost));
  //HIP_CHECK(hipDeviceSynchronize());
  return next_host;
}