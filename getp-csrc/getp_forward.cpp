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
#ifndef GETP_ROUTER_TOPK_MAXK
#define GETP_ROUTER_TOPK_MAXK 4
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

__global__ void __launch_bounds__(256) mlp1_swiglu_bf16_bucketed_mfma32_kernel(
    float *__restrict__ gate_up_all, const float *__restrict__ x,
    const __hip_bfloat16 *__restrict__ W1, const __hip_bfloat16 *__restrict__ B1,
    const int *__restrict__ offsets, const int *__restrict__ tok_idx,
    int H, int I, int E, int cap_pairs, float swiglu_limit) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int WM = 16, WN = 16;
  constexpr int MFMA_M = 16, MFMA_N = 16, MFMA_K = 4;
  using mfloat4 = __attribute__((__vector_size__(MFMA_K * sizeof(float)))) float;

  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int xTile = wave % (BM / WM);
  const int yTile = wave / (BM / WM);
  const int xMF = lane % MFMA_M;
  const int yMF = lane / MFMA_M;

  const int e = blockIdx.y % E;
  const int blk = blockIdx.y / E;
  const int start = offsets[e];
  const int end = offsets[e + 1];
  const int n = end - start;
  const int base_all = blk * BM;
  const int base = start + base_all;
  if (base >= end) return;
  const int tcnt = min(BM, n - base_all);

  const size_t w1_e = (size_t)e * (size_t)(2 * I) * (size_t)H;
  const size_t b1_e = (size_t)e * (size_t)(2 * I);

  __shared__ float sx[BK][BM];
  __shared__ float swg[BK][BN];
  __shared__ float swu[BK][BN];

  mfloat4 dgn = {0, 0, 0, 0};
  mfloat4 dun = {0, 0, 0, 0};

  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  const int strideX = BLOCK_SIZE / (BK / 4);
  const int strideW = BLOCK_SIZE / (BK / 8);

  for (int bk = 0; bk < H; bk += BK) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk = bk + 4 * loadX_x;
      const int r = loadX_y + off;
      const int pos = base + r;
      if (r < tcnt && kk + 3 < H) {
        const int b = tok_idx[pos];
        float4 v = *reinterpret_cast<const float4 *>(x + (size_t)b * H + kk);
        sx[4 * loadX_x + 0][r] = v.x;
        sx[4 * loadX_x + 1][r] = v.y;
        sx[4 * loadX_x + 2][r] = v.z;
        sx[4 * loadX_x + 3][r] = v.w;
      } else if (r < tcnt && kk < H) {
        const int b = tok_idx[pos];
        const float *p = x + (size_t)b * H + kk;
        sx[4 * loadX_x + 0][r] = (kk + 0 < H) ? p[0] : 0.f;
        sx[4 * loadX_x + 1][r] = (kk + 1 < H) ? p[1] : 0.f;
        sx[4 * loadX_x + 2][r] = (kk + 2 < H) ? p[2] : 0.f;
        sx[4 * loadX_x + 3][r] = (kk + 3 < H) ? p[3] : 0.f;
      }
    }

    for (int off = 0; off < BN; off += strideW) {
      if (loadW_y + off >= BN) break;
      const int kk = bk + 8 * loadW_x;
      const int c = blockIdx.x * BN + loadW_y + off;

      uint4 wg = make_uint4(0, 0, 0, 0);
      uint4 wu = make_uint4(0, 0, 0, 0);
      if (c < I) {
        if (kk + 7 < H) {
          wg = *reinterpret_cast<const uint4 *>(W1 + w1_e + (size_t)(2 * c + 0) * H + kk);
          wu = *reinterpret_cast<const uint4 *>(W1 + w1_e + (size_t)(2 * c + 1) * H + kk);
        } else if (kk < H) {
          const __hip_bfloat16 *pg = W1 + w1_e + (size_t)(2 * c + 0) * H + kk;
          const __hip_bfloat16 *pu = W1 + w1_e + (size_t)(2 * c + 1) * H + kk;
          unsigned u0=0,u1=0,u2=0,u3=0,v0=0,v1=0,v2=0,v3=0;
          if (kk + 0 < H) { u0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[0]); v0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[0]); }
          if (kk + 1 < H) { u0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[1]) << 16; v0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[1]) << 16; }
          if (kk + 2 < H) { u1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[2]); v1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[2]); }
          if (kk + 3 < H) { u1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[3]) << 16; v1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[3]) << 16; }
          if (kk + 4 < H) { u2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[4]); v2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[4]); }
          if (kk + 5 < H) { u2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[5]) << 16; v2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[5]) << 16; }
          if (kk + 6 < H) { u3 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[6]); v3 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[6]); }
          wg = make_uint4(u0, u1, u2, u3);
          wu = make_uint4(v0, v1, v2, v3);
        }
      }

      if (c < I) {
        swg[8 * loadW_x + 0][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.x & 0xFFFF));
        swg[8 * loadW_x + 1][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.x >> 16));
        swg[8 * loadW_x + 2][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.y & 0xFFFF));
        swg[8 * loadW_x + 3][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.y >> 16));
        swg[8 * loadW_x + 4][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.z & 0xFFFF));
        swg[8 * loadW_x + 5][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.z >> 16));
        swg[8 * loadW_x + 6][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.w & 0xFFFF));
        swg[8 * loadW_x + 7][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.w >> 16));

        swu[8 * loadW_x + 0][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.x & 0xFFFF));
        swu[8 * loadW_x + 1][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.x >> 16));
        swu[8 * loadW_x + 2][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.y & 0xFFFF));
        swu[8 * loadW_x + 3][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.y >> 16));
        swu[8 * loadW_x + 4][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.z & 0xFFFF));
        swu[8 * loadW_x + 5][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.z >> 16));
        swu[8 * loadW_x + 6][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.w & 0xFFFF));
        swu[8 * loadW_x + 7][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.w >> 16));
      }
    }

    __syncthreads();

    for (int k = 0; k < BK; k += MFMA_K) {
      float amk = sx[k + yMF][yTile * WM + xMF];
      float bgk = swg[k + yMF][xTile * WN + xMF];
      float buk = swu[k + yMF][xTile * WN + xMF];
      dgn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bgk, dgn, 0, 0, 0);
      dun = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, buk, dun, 0, 0, 0);
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    const int xD = lane % MFMA_N;
    const int yD = MFMA_K * (lane / MFMA_N) + i;
    const int col = blockIdx.x * BN + xTile * WN + xD;
    const int row = yTile * WM + yD;
    if (row < tcnt && col < I) {
      const int pos = base + row;
      const float bg = B1 ? __bfloat162float(B1[b1_e + (size_t)(2 * col + 0)]) : 0.f;
      const float bu = B1 ? __bfloat162float(B1[b1_e + (size_t)(2 * col + 1)]) : 0.f;
      float g = dgn[i] + bg;
      float u = dun[i] + bu;
      if (g > swiglu_limit) g = swiglu_limit;
      if (u > swiglu_limit) u = swiglu_limit;
      if (u < -swiglu_limit) u = -swiglu_limit;
      const float a = 1.702f;
      float s = 1.f / (1.f + expf(-a * g));
      gate_up_all[(size_t)pos * (size_t)I + col] = (g * s) * (u + 1.f);
    }
  }
}

__global__ void __launch_bounds__(128) mlp1_swiglu_bf16_bucketed_mfma16_kernel(
    float *__restrict__ gate_up_all, const float *__restrict__ x,
    const __hip_bfloat16 *__restrict__ W1, const __hip_bfloat16 *__restrict__ B1,
    const int *__restrict__ offsets, const int *__restrict__ tok_idx,
    int H, int I, int E, int cap_pairs, float swiglu_limit) {
  constexpr int BLOCK_SIZE = 128;
  constexpr int BM = 16, BN = 32, BK = 32;
  constexpr int WM = 16, WN = 16;
  constexpr int MFMA_M = 16, MFMA_N = 16, MFMA_K = 4;
  using mfloat4 = __attribute__((__vector_size__(MFMA_K * sizeof(float)))) float;

  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;

  const int mTiles = BM / WM;           // 1
  const int nTiles = BN / WN;           // 2
  const int mTile  = wave % mTiles;     // 0
  const int nTile  = wave / mTiles;     // 0 or 1

  const int xMF = lane % MFMA_M;
  const int yMF = lane / MFMA_M;

  const int e   = blockIdx.y % E;
  const int blk = blockIdx.y / E;
  const int start = offsets[e];
  const int end   = offsets[e + 1];
  const int n     = end - start;
  const int base_all = blk * BM;
  const int base = start + base_all;
  if (base >= end) return;
  const int tcnt = min(BM, n - base_all);

  const size_t w1_e = (size_t)e * (size_t)(2 * I) * (size_t)H;
  const size_t b1_e = (size_t)e * (size_t)(2 * I);

  __shared__ float sx[BK][BM];
  __shared__ float swg[BK][BN];
  __shared__ float swu[BK][BN];

  mfloat4 dgn = {0,0,0,0};
  mfloat4 dun = {0,0,0,0};

  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  const int strideX = BLOCK_SIZE / (BK / 4);
  const int strideW = BLOCK_SIZE / (BK / 8);

  for (int bk = 0; bk < H; bk += BK) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk = bk + 4 * loadX_x;
      const int r  = loadX_y + off;
      const int pos = base + r;
      if (r < tcnt && kk + 3 < H) {
        const int b = tok_idx[pos];
        float4 v = *reinterpret_cast<const float4 *>(x + (size_t)b * H + kk);
        sx[4 * loadX_x + 0][r] = v.x;
        sx[4 * loadX_x + 1][r] = v.y;
        sx[4 * loadX_x + 2][r] = v.z;
        sx[4 * loadX_x + 3][r] = v.w;
      } else if (r < tcnt && kk < H) {
        const int b = tok_idx[pos];
        const float *p = x + (size_t)b * H + kk;
        sx[4 * loadX_x + 0][r] = (kk + 0 < H) ? p[0] : 0.f;
        sx[4 * loadX_x + 1][r] = (kk + 1 < H) ? p[1] : 0.f;
        sx[4 * loadX_x + 2][r] = (kk + 2 < H) ? p[2] : 0.f;
        sx[4 * loadX_x + 3][r] = (kk + 3 < H) ? p[3] : 0.f;
      }
    }

    for (int off = 0; off < BN; off += strideW) {
      if (loadW_y + off >= BN) break;
      const int kk = bk + 8 * loadW_x;
      const int c  = blockIdx.x * BN + loadW_y + off;

      uint4 wg = make_uint4(0,0,0,0);
      uint4 wu = make_uint4(0,0,0,0);
      if (c < I) {
        if (kk + 7 < H) {
          wg = *reinterpret_cast<const uint4 *>(W1 + w1_e + (size_t)(2 * c + 0) * H + kk);
          wu = *reinterpret_cast<const uint4 *>(W1 + w1_e + (size_t)(2 * c + 1) * H + kk);
        } else if (kk < H) {
          const __hip_bfloat16 *pg = W1 + w1_e + (size_t)(2 * c + 0) * H + kk;
          const __hip_bfloat16 *pu = W1 + w1_e + (size_t)(2 * c + 1) * H + kk;
          unsigned u0=0,u1=0,u2=0,u3=0,v0=0,v1=0,v2=0,v3=0;
          if (kk + 0 < H) { u0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[0]); v0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[0]); }
          if (kk + 1 < H) { u0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[1]) << 16; v0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[1]) << 16; }
          if (kk + 2 < H) { u1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[2]); v1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[2]); }
          if (kk + 3 < H) { u1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[3]) << 16; v1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[3]) << 16; }
          if (kk + 4 < H) { u2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[4]); v2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[4]); }
          if (kk + 5 < H) { u2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[5]) << 16; v2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[5]) << 16; }
          if (kk + 6 < H) { u3 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[6]); v3 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[6]); }
          wg = make_uint4(u0,u1,u2,u3);
          wu = make_uint4(v0,v1,v2,v3);
        }
      }

      if (c < I) {
        swg[8 * loadW_x + 0][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.x & 0xFFFF));
        swg[8 * loadW_x + 1][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.x >> 16));
        swg[8 * loadW_x + 2][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.y & 0xFFFF));
        swg[8 * loadW_x + 3][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.y >> 16));
        swg[8 * loadW_x + 4][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.z & 0xFFFF));
        swg[8 * loadW_x + 5][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.z >> 16));
        swg[8 * loadW_x + 6][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.w & 0xFFFF));
        swg[8 * loadW_x + 7][loadW_y + off] = bf16bits_to_f32((unsigned short)(wg.w >> 16));

        swu[8 * loadW_x + 0][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.x & 0xFFFF));
        swu[8 * loadW_x + 1][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.x >> 16));
        swu[8 * loadW_x + 2][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.y & 0xFFFF));
        swu[8 * loadW_x + 3][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.y >> 16));
        swu[8 * loadW_x + 4][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.z & 0xFFFF));
        swu[8 * loadW_x + 5][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.z >> 16));
        swu[8 * loadW_x + 6][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.w & 0xFFFF));
        swu[8 * loadW_x + 7][loadW_y + off] = bf16bits_to_f32((unsigned short)(wu.w >> 16));
      }
    }

    __syncthreads();

    for (int k = 0; k < BK; k += MFMA_K) {
      const int rRow = mTile * WM + xMF;
      float amk = (rRow < tcnt) ? sx[k + yMF][rRow] : 0.f;
      float bgk = swg[k + yMF][nTile * WN + xMF];
      float buk = swu[k + yMF][nTile * WN + xMF];
      dgn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bgk, dgn, 0, 0, 0);
      dun = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, buk, dun, 0, 0, 0);
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    const int xD = lane % MFMA_N;
    const int yD = MFMA_K * (lane / MFMA_N) + i;
    const int col = blockIdx.x * BN + nTile * WN + xD;
    const int row = mTile * WM + yD;
    if (row < tcnt && col < I) {
      const int pos = base + row;
      const float bg = B1 ? __bfloat162float(B1[b1_e + (size_t)(2 * col + 0)]) : 0.f;
      const float bu = B1 ? __bfloat162float(B1[b1_e + (size_t)(2 * col + 1)]) : 0.f;
      float g = dgn[i] + bg;
      float u = dun[i] + bu;
      if (g > swiglu_limit) g = swiglu_limit;
      if (u > swiglu_limit) u = swiglu_limit;
      if (u < -swiglu_limit) u = -swiglu_limit;
      const float a = 1.702f;
      float s = 1.f / (1.f + expf(-a * g));
      gate_up_all[(size_t)pos * (size_t)I + col] = (g * s) * (u + 1.f);
    }
  }
}
static inline void launch_mlp1_swiglu_bf16_bucketed(
    float *gate_up_all, const float *x, const __hip_bfloat16 *w1_layer,
    const __hip_bfloat16 *b1_layer, const int *offsets, const int *tok_idx,
    int H, int I, int E, int cap_pairs, float swiglu_limit) {
  PROFILE_FUNCTION();
  if (cap_pairs <= 16) {
    constexpr int BN = 32, BM = 16;
    dim3 block(128);
    int by = (cap_pairs + BM - 1) / BM;
    dim3 grid((I + BN - 1) / BN, E * by);
    mlp1_swiglu_bf16_bucketed_mfma16_kernel<<<grid, block>>>(
        gate_up_all, x, w1_layer, b1_layer, offsets, tok_idx,
        H, I, E, cap_pairs, swiglu_limit);
  } else {
    constexpr int BN = 32, BM = 32;
    dim3 block(256);
    int by = (cap_pairs + BM - 1) / BM;
    dim3 grid((I + BN - 1) / BN, E * by);
    mlp1_swiglu_bf16_bucketed_mfma32_kernel<<<grid, block>>>(
        gate_up_all, x, w1_layer, b1_layer, offsets, tok_idx,
        H, I, E, cap_pairs, swiglu_limit);
  }
}


__global__ void __launch_bounds__(256) mlp2_partial_bf16_bucketed_mfma_kernel(
    float *__restrict__ z_partial, const float *__restrict__ gate_up_all,
    const __hip_bfloat16 *__restrict__ W2,
    const __hip_bfloat16 *__restrict__ B2,
    const float *__restrict__ w_by_bucket, const int *__restrict__ offsets,
    const int *__restrict__ tok_idx, int I, int H, int E, int cap_pairs) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 4;
  using mfloat4 =
      __attribute__((__vector_size__(MFMA_K * sizeof(float)))) float;

  const int laneIdx = threadIdx.x % warpSize;
  const int waveIdx = threadIdx.x / warpSize;
  const int xInBlockTile = waveIdx % (BM / WM);
  const int yInBlockTile = waveIdx / (BM / WM);
  const int xInMFMA = laneIdx % MFMA_M;
  const int yInMFMA = laneIdx / MFMA_M;

  const int e = blockIdx.y % E;
  const int blk = blockIdx.y / E;

  const int start = offsets[e];
  const int end = offsets[e + 1];
  const int n = end - start;

  const int base_all = blk * BM;
  const int base = start + base_all;
  if (base >= end) return;

  const int tcnt = min(BM, n - base_all);
  const size_t w2_e = (size_t)e * (size_t)H * (size_t)I;
  const size_t b2_e = (size_t)e * (size_t)H;

  __shared__ float sx[BK][BM];
  __shared__ float sw[BK][BN];

  for (int z = threadIdx.x; z < BK * BM; z += BLOCK_SIZE)
    ((float *)sx)[z] = 0.f;
  for (int z = threadIdx.x; z < BK * BN; z += BLOCK_SIZE)
    ((float *)sw)[z] = 0.f;
  __syncthreads();

  mfloat4 dmn = {0, 0, 0, 0};

  const int loadXIdx_x = threadIdx.x % (BK / 4);
  const int loadXIdx_y = threadIdx.x / (BK / 4);
  const int loadWIdx_x = threadIdx.x % (BK / 8);
  const int loadWIdx_y = threadIdx.x / (BK / 8);
  const int strideX = BLOCK_SIZE / (BK / 4);
  const int strideW = BLOCK_SIZE / (BK / 8);

  for (int bk = 0; bk < I; bk += BK) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadXIdx_y + off >= BM) break;
      const int kk = bk + 4 * loadXIdx_x;
      const int r = loadXIdx_y + off;
      const int pos = base + r;
      if (r < tcnt && kk + 3 < I) {
        float4 tmp = *reinterpret_cast<const float4 *>(
            gate_up_all + (size_t)pos * (size_t)I + kk);
        sx[4 * loadXIdx_x + 0][r] = tmp.x;
        sx[4 * loadXIdx_x + 1][r] = tmp.y;
        sx[4 * loadXIdx_x + 2][r] = tmp.z;
        sx[4 * loadXIdx_x + 3][r] = tmp.w;
      } else if (r < tcnt && kk < I) {
        const float *src = gate_up_all + (size_t)pos * (size_t)I + kk;
        sx[4 * loadXIdx_x + 0][r] = (kk + 0 < I) ? src[0] : 0.f;
        sx[4 * loadXIdx_x + 1][r] = (kk + 1 < I) ? src[1] : 0.f;
        sx[4 * loadXIdx_x + 2][r] = (kk + 2 < I) ? src[2] : 0.f;
        sx[4 * loadXIdx_x + 3][r] = (kk + 3 < I) ? src[3] : 0.f;
      }
    }

    for (int off = 0; off < BN; off += strideW) {
      if (loadWIdx_y + off >= BN) break;
      const int kk = bk + 8 * loadWIdx_x;
      const int c = blockIdx.x * BN + loadWIdx_y + off;

      uint4 wb = make_uint4(0, 0, 0, 0);
      if (c < H) {
        if (kk + 7 < I) {
          wb = *reinterpret_cast<const uint4 *>(W2 + w2_e +
                                                (size_t)c * (size_t)I + kk);
        } else if (kk < I) {
          const __hip_bfloat16 *src = W2 + w2_e + (size_t)c * (size_t)I + kk;
          unsigned int u0 = 0, u1 = 0, u2 = 0, u3 = 0;
          if (kk + 0 < I)
            u0 |=
                ((unsigned)(reinterpret_cast<const unsigned short *>(src)[0]));
          if (kk + 1 < I)
            u0 |= ((unsigned)(reinterpret_cast<const unsigned short *>(src)[1]))
                  << 16;
          if (kk + 2 < I)
            u1 |=
                ((unsigned)(reinterpret_cast<const unsigned short *>(src)[2]));
          if (kk + 3 < I)
            u1 |= ((unsigned)(reinterpret_cast<const unsigned short *>(src)[3]))
                  << 16;
          if (kk + 4 < I)
            u2 |=
                ((unsigned)(reinterpret_cast<const unsigned short *>(src)[4]));
          if (kk + 5 < I)
            u2 |= ((unsigned)(reinterpret_cast<const unsigned short *>(src)[5]))
                  << 16;
          if (kk + 6 < I)
            u3 |=
                ((unsigned)(reinterpret_cast<const unsigned short *>(src)[6]));
          wb = make_uint4(u0, u1, u2, u3);
        }
      }

      if (c < H) {
        sw[8 * loadWIdx_x + 0][loadWIdx_y + off] =
            bf16bits_to_f32((unsigned short)(wb.x & 0xFFFF));
        sw[8 * loadWIdx_x + 1][loadWIdx_y + off] =
            bf16bits_to_f32((unsigned short)(wb.x >> 16));
        sw[8 * loadWIdx_x + 2][loadWIdx_y + off] =
            bf16bits_to_f32((unsigned short)(wb.y & 0xFFFF));
        sw[8 * loadWIdx_x + 3][loadWIdx_y + off] =
            bf16bits_to_f32((unsigned short)(wb.y >> 16));
        sw[8 * loadWIdx_x + 4][loadWIdx_y + off] =
            bf16bits_to_f32((unsigned short)(wb.z & 0xFFFF));
        sw[8 * loadWIdx_x + 5][loadWIdx_y + off] =
            bf16bits_to_f32((unsigned short)(wb.z >> 16));
        sw[8 * loadWIdx_x + 6][loadWIdx_y + off] =
            bf16bits_to_f32((unsigned short)(wb.w & 0xFFFF));
        sw[8 * loadWIdx_x + 7][loadWIdx_y + off] =
            bf16bits_to_f32((unsigned short)(wb.w >> 16));
      }
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
    const int col = blockIdx.x * BN + xInBlockTile * WN + xInD;
    const int row = yInBlockTile * WM + yInD;
    if (row < tcnt && col < H) {
      const int pos = base + row;
      const float bias = B2 ? __bfloat162float(B2[b2_e + col]) : 0.f;
      const float w = w_by_bucket[pos];
      z_partial[(size_t)pos * (size_t)H + col] = (dmn[i] + bias) * w;
    }
  }
}

__global__ void __launch_bounds__(128) mlp2_partial_bf16_bucketed_mfma16_kernel(
    float *__restrict__ z_partial, const float *__restrict__ gate_up_all,
    const __hip_bfloat16 *__restrict__ W2, const __hip_bfloat16 *__restrict__ B2,
    const float *__restrict__ w_by_bucket, const int *__restrict__ offsets,
    const int *__restrict__ tok_idx, int I, int H, int E, int cap_pairs) {
  constexpr int BLOCK_SIZE = 128;
  constexpr int BM = 16, BN = 32, BK = 32;
  constexpr int WM = 16, WN = 16;
  constexpr int MFMA_M = 16, MFMA_N = 16, MFMA_K = 4;
  using mfloat4 = __attribute__((__vector_size__(MFMA_K * sizeof(float)))) float;

  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int mTiles = BM / WM;
  const int nTiles = BN / WN;
  const int mTile  = wave % mTiles;
  const int nTile  = wave / mTiles;
  const int xMF = lane % MFMA_M;
  const int yMF = lane / MFMA_M;

  const int e = blockIdx.y % E;
  const int blk = blockIdx.y / E;

  const int start = offsets[e];
  const int end   = offsets[e + 1];
  const int n     = end - start;

  const int base_all = blk * BM;
  const int base = start + base_all;
  if (base >= end) return;

  const int tcnt = min(BM, n - base_all);
  const size_t w2_e = (size_t)e * (size_t)H * (size_t)I;
  const size_t b2_e = (size_t)e * (size_t)H;

  __shared__ float sx[BK][BM];
  __shared__ float sw[BK][BN];

  mfloat4 dmn = {0, 0, 0, 0};

  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  const int strideX = BLOCK_SIZE / (BK / 4);
  const int strideW = BLOCK_SIZE / (BK / 8);

  for (int bk = 0; bk < I; bk += BK) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk = bk + 4 * loadX_x;
      const int r  = loadX_y + off;
      const int pos = base + r;
      if (r < tcnt && kk + 3 < I) {
        float4 v = *reinterpret_cast<const float4 *>(gate_up_all + (size_t)pos * (size_t)I + kk);
        sx[4 * loadX_x + 0][r] = v.x;
        sx[4 * loadX_x + 1][r] = v.y;
        sx[4 * loadX_x + 2][r] = v.z;
        sx[4 * loadX_x + 3][r] = v.w;
      } else if (r < tcnt && kk < I) {
        const float *p = gate_up_all + (size_t)pos * (size_t)I + kk;
        sx[4 * loadX_x + 0][r] = (kk + 0 < I) ? p[0] : 0.f;
        sx[4 * loadX_x + 1][r] = (kk + 1 < I) ? p[1] : 0.f;
        sx[4 * loadX_x + 2][r] = (kk + 2 < I) ? p[2] : 0.f;
        sx[4 * loadX_x + 3][r] = (kk + 3 < I) ? p[3] : 0.f;
      }
    }

    for (int off = 0; off < BN; off += strideW) {
      if (loadW_y + off >= BN) break;
      const int kk = bk + 8 * loadW_x;
      const int c  = blockIdx.x * BN + loadW_y + off;

      uint4 wb = make_uint4(0, 0, 0, 0);
      if (c < H) {
        if (kk + 7 < I) {
          wb = *reinterpret_cast<const uint4 *>(W2 + w2_e + (size_t)c * (size_t)I + kk);
        } else if (kk < I) {
          const __hip_bfloat16 *src = W2 + w2_e + (size_t)c * (size_t)I + kk;
          unsigned u0 = 0, u1 = 0, u2 = 0, u3 = 0;
          if (kk + 0 < I) u0 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[0]);
          if (kk + 1 < I) u0 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[1]) << 16;
          if (kk + 2 < I) u1 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[2]);
          if (kk + 3 < I) u1 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[3]) << 16;
          if (kk + 4 < I) u2 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[4]);
          if (kk + 5 < I) u2 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[5]) << 16;
          if (kk + 6 < I) u3 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[6]);
          wb = make_uint4(u0, u1, u2, u3);
        }
      }

      if (c < H) {
        sw[8 * loadW_x + 0][loadW_y + off] = bf16bits_to_f32((unsigned short)(wb.x & 0xFFFF));
        sw[8 * loadW_x + 1][loadW_y + off] = bf16bits_to_f32((unsigned short)(wb.x >> 16));
        sw[8 * loadW_x + 2][loadW_y + off] = bf16bits_to_f32((unsigned short)(wb.y & 0xFFFF));
        sw[8 * loadW_x + 3][loadW_y + off] = bf16bits_to_f32((unsigned short)(wb.y >> 16));
        sw[8 * loadW_x + 4][loadW_y + off] = bf16bits_to_f32((unsigned short)(wb.z & 0xFFFF));
        sw[8 * loadW_x + 5][loadW_y + off] = bf16bits_to_f32((unsigned short)(wb.z >> 16));
        sw[8 * loadW_x + 6][loadW_y + off] = bf16bits_to_f32((unsigned short)(wb.w & 0xFFFF));
        sw[8 * loadW_x + 7][loadW_y + off] = bf16bits_to_f32((unsigned short)(wb.w >> 16));
      } else {
        sw[8 * loadW_x + 0][loadW_y + off] = 0.f;
        sw[8 * loadW_x + 1][loadW_y + off] = 0.f;
        sw[8 * loadW_x + 2][loadW_y + off] = 0.f;
        sw[8 * loadW_x + 3][loadW_y + off] = 0.f;
        sw[8 * loadW_x + 4][loadW_y + off] = 0.f;
        sw[8 * loadW_x + 5][loadW_y + off] = 0.f;
        sw[8 * loadW_x + 6][loadW_y + off] = 0.f;
        sw[8 * loadW_x + 7][loadW_y + off] = 0.f;
      }
    }

    __syncthreads();

    for (int k = 0; k < BK; k += MFMA_K) {
      const int rRow = mTile * WM + xMF;
      float amk = (rRow < tcnt) ? sx[k + yMF][rRow] : 0.f;
      float bkn = sw[k + yMF][nTile * WN + xMF];
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    const int xD = lane % MFMA_N;
    const int yD = MFMA_K * (lane / MFMA_N) + i;
    const int col = blockIdx.x * BN + nTile * WN + xD;
    const int row = mTile * WM + yD;
    if (row < tcnt && col < H) {
      const int pos = base + row;
      const float bias = B2 ? __bfloat162float(B2[b2_e + col]) : 0.f;
      const float w = w_by_bucket[pos];
      z_partial[(size_t)pos * (size_t)H + col] = (dmn[i] + bias) * w;
    }
  }
}

static inline void launch_mlp2_partial_bf16_bucketed_mfma(
    float *z_partial, const float *gate_up_all, const __hip_bfloat16 *w2_layer,
    const __hip_bfloat16 *b2_layer, const float *w_by_bucket,
    const int *offsets, const int *tok_idx, int I, int H, int E,
    int cap_pairs) {
  if (cap_pairs <= 16) {
    constexpr int BN = 32, BM = 16;
    dim3 block(128);
    int by = (cap_pairs + BM - 1) / BM;
    dim3 grid((H + BN - 1) / BN, E * by);
    mlp2_partial_bf16_bucketed_mfma16_kernel<<<grid, block>>>(
        z_partial, gate_up_all, w2_layer, b2_layer, w_by_bucket,
        offsets, tok_idx, I, H, E, cap_pairs);
  } else {
    constexpr int BN = 32, BM = 32;
    dim3 block(256);
    int by = (cap_pairs + BM - 1) / BM;
    dim3 grid((H + BN - 1) / BN, E * by);
    mlp2_partial_bf16_bucketed_mfma_kernel<<<grid, block>>>(
        z_partial, gate_up_all, w2_layer, b2_layer, w_by_bucket,
        offsets, tok_idx, I, H, E, cap_pairs);
  }
}


// static inline void launch_mlp2_partial_bf16_bucketed_mfma(
//     float *z_partial, const float *gate_up_all, const __hip_bfloat16 *w2_layer,
//     const __hip_bfloat16 *b2_layer, const float *w_by_bucket,
//     const int *offsets, const int *tok_idx, int I, int H, int E,
//     int cap_pairs) {
//       PROFILE_FUNCTION();
//   constexpr int BN = 32;
//   constexpr int BM = 32;
//   dim3 block(256);
//   int by = (cap_pairs + BM - 1) / BM;
//   dim3 grid((H + BN - 1) / BN, E * by);
//   mlp2_partial_bf16_bucketed_mfma_kernel<<<grid, block>>>(
//       z_partial, gate_up_all, w2_layer, b2_layer, w_by_bucket, offsets, tok_idx,
//       I, H, E, cap_pairs);
//       //HIP_CHECK(hipDeviceSynchronize());
// }



__global__ void moe_gather_pairs_kernel(float *__restrict__ e_agg,
                                        const float *__restrict__ z_partial,
                                        const int *__restrict__ pair_pos, int H,
                                        int K, int B) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  if (row >= H || b >= B) return;
  int base = b * K;
  float s = 0.f;
  for (int k = 0; k < K; ++k) {
    int pos = pair_pos[base + k];
    if (pos >= 0) s += z_partial[(size_t)pos * (size_t)H + row];
  }
  e_agg[(size_t)b * (size_t)H + row] += s;
}

static inline void moe_gather_pairs(float *e_agg, const float *z_partial,
                                    const int *pair_pos, int H, int K, int B) {
  dim3 blk(256), grd((H + blk.x - 1) / blk.x, B);
  moe_gather_pairs_kernel<<<grd, blk>>>(e_agg, z_partial, pair_pos, H, K, B);
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
matmul_kernel_nosplit(float* __restrict__ xout,
                      const float* __restrict__ x,
                      const __hip_bfloat16* __restrict__ w,
                      const __hip_bfloat16* __restrict__ b,
                      int M, int N, int K) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int WM = 16, WN = 16;
  constexpr int MFMA_M = 16, MFMA_N = 16, MFMA_K = 4;
  using mfloat4 = __attribute__((__vector_size__(MFMA_K * sizeof(float)))) float;

  const int waveIdx = threadIdx.x / warpSize;
  const int xTile = waveIdx % (BM / WM);
  const int yTile = waveIdx / (BM / WM);
  const int lane   = threadIdx.x % warpSize;
  const int xMF    = lane % MFMA_M;
  const int yMF    = lane / MFMA_M;

  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  constexpr int strideX = BLOCK_SIZE / (BK / 4);
  constexpr int strideW = BLOCK_SIZE / (BK / 8);

  extern __shared__ float sm[];
  float* sx = sm;                      // [BK][BM]
  float* sw = sm + (size_t)BK * BM;    // [BK][BN]

  mfloat4 dmn = {0,0,0,0};

  for (int bk = 0; bk < K; bk += BK) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk = bk + 4 * loadX_x;
      const int row = blockIdx.y * BM + loadX_y + off;
      float4 v = (row < M && kk + 3 < K)
                  ? *reinterpret_cast<const float4*>(x + (size_t)row * K + kk)
                  : make_float4(0,0,0,0);
      sx[(4*loadX_x+0)*BM + (loadX_y+off)] = v.x;
      sx[(4*loadX_x+1)*BM + (loadX_y+off)] = v.y;
      sx[(4*loadX_x+2)*BM + (loadX_y+off)] = v.z;
      sx[(4*loadX_x+3)*BM + (loadX_y+off)] = v.w;
    }
    for (int off = 0; off < BN; off += strideW) {
      if (loadW_y + off >= BN) break;
      const int kk  = bk + 8 * loadW_x;
      const int col = blockIdx.x * BN + loadW_y + off;
      uint4 t = (col < N && kk + 7 < K)
                 ? *reinterpret_cast<const uint4*>(w + (size_t)col * K + kk)
                 : make_uint4(0,0,0,0);
      float* dst = sw + (size_t)(8*loadW_x) * BN + (loadW_y + off);
      dst[0*BN] = bf16bits_to_f32((unsigned short)(t.x & 0xFFFF));
      dst[1*BN] = bf16bits_to_f32((unsigned short)(t.x >> 16));
      dst[2*BN] = bf16bits_to_f32((unsigned short)(t.y & 0xFFFF));
      dst[3*BN] = bf16bits_to_f32((unsigned short)(t.y >> 16));
      dst[4*BN] = bf16bits_to_f32((unsigned short)(t.z & 0xFFFF));
      dst[5*BN] = bf16bits_to_f32((unsigned short)(t.z >> 16));
      dst[6*BN] = bf16bits_to_f32((unsigned short)(t.w & 0xFFFF));
      dst[7*BN] = bf16bits_to_f32((unsigned short)(t.w >> 16));
    }
    __syncthreads();

    for (int k = 0; k < BK; k += MFMA_K) {
      float amk = sx[(k + yMF) * BM + (yTile * WM + xMF)];
      float bkn = sw[(k + yMF) * BN + (xTile * WN + xMF)];
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }
    __syncthreads();
  }

  for (int i = 0; i < 4; ++i) {
    const int xD  = lane % MFMA_N;
    const int yD  = MFMA_K * (lane / MFMA_N) + i;
    const int col = blockIdx.x * BN + xTile * WN + xD;
    const int row = blockIdx.y * BM + yTile * WM + yD;
    if (row < M && col < N) {
      float v = dmn[i];
      if (b) v += __bfloat162float(b[col]);
      xout[(size_t)row * N + col] = v;
    }
  }
}
__global__ void __launch_bounds__(256)
matmul_kernel_splitk_store(float* __restrict__ partial,
                           const float* __restrict__ x,
                           const __hip_bfloat16* __restrict__ w,
                           int M, int N, int K, int splits) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int WM = 16, WN = 16;
  constexpr int MFMA_M = 16, MFMA_N = 16, MFMA_K = 4;
  using mfloat4 = __attribute__((__vector_size__(MFMA_K * sizeof(float)))) float;

  const int waveIdx = threadIdx.x / warpSize;
  const int xTile = waveIdx % (BM / WM);
  const int yTile = waveIdx / (BM / WM);
  const int lane   = threadIdx.x % warpSize;
  const int xMF    = lane % MFMA_M;
  const int yMF    = lane / MFMA_M;

  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  constexpr int strideX = BLOCK_SIZE / (BK / 4);
  constexpr int strideW = BLOCK_SIZE / (BK / 8);

  extern __shared__ float sm[];
  float* sx = sm;                      // [BK][BM]
  float* sw = sm + (size_t)BK * BM;    // [BK][BN]

  mfloat4 dmn = {0,0,0,0};

  const int split   = blockIdx.z;
  const int kChunk  = (K + splits - 1) / splits;
  const int kBeg    = split * kChunk;
  const int kEnd    = min(K, kBeg + kChunk);

  for (int bk = kBeg; bk < kEnd; bk += BK) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk  = bk + 4 * loadX_x;
      const int row = blockIdx.y * BM + loadX_y + off;
      float4 v = (row < M && kk + 3 < kEnd)
                  ? *reinterpret_cast<const float4*>(x + (size_t)row * K + kk)
                  : make_float4(0,0,0,0);
      sx[(4*loadX_x+0)*BM + (loadX_y+off)] = v.x;
      sx[(4*loadX_x+1)*BM + (loadX_y+off)] = v.y;
      sx[(4*loadX_x+2)*BM + (loadX_y+off)] = v.z;
      sx[(4*loadX_x+3)*BM + (loadX_y+off)] = v.w;
    }
    for (int off = 0; off < BN; off += strideW) {
      if (loadW_y + off >= BN) break;
      const int kk  = bk + 8 * loadW_x;
      const int col = blockIdx.x * BN + loadW_y + off;
      uint4 t = (col < N && kk < kEnd)
                 ? *reinterpret_cast<const uint4*>(w + (size_t)col * K + kk)
                 : make_uint4(0,0,0,0);
      float* dst = sw + (size_t)(8*loadW_x) * BN + (loadW_y + off);
      dst[0*BN] = bf16bits_to_f32((unsigned short)(t.x & 0xFFFF));
      dst[1*BN] = bf16bits_to_f32((unsigned short)(t.x >> 16));
      dst[2*BN] = bf16bits_to_f32((unsigned short)(t.y & 0xFFFF));
      dst[3*BN] = bf16bits_to_f32((unsigned short)(t.y >> 16));
      dst[4*BN] = bf16bits_to_f32((unsigned short)(t.z & 0xFFFF));
      dst[5*BN] = bf16bits_to_f32((unsigned short)(t.z >> 16));
      dst[6*BN] = bf16bits_to_f32((unsigned short)(t.w & 0xFFFF));
      dst[7*BN] = bf16bits_to_f32((unsigned short)(t.w >> 16));
    }
    __syncthreads();

    for (int k = 0; k < BK; k += MFMA_K) {
      float amk = sx[(k + yMF) * BM + (yTile * WM + xMF)];
      float bkn = sw[(k + yMF) * BN + (xTile * WN + xMF)];
      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
    }
    __syncthreads();
  }

  const size_t slice_stride = (size_t)M * N;
  const size_t base_out = (size_t)split * slice_stride;

  for (int i = 0; i < 4; ++i) {
    const int xD  = lane % MFMA_N;
    const int yD  = MFMA_K * (lane / MFMA_N) + i;
    const int col = blockIdx.x * BN + xTile * WN + xD;
    const int row = blockIdx.y * BM + yTile * WM + yD;
    if (row < M && col < N) {
      partial[base_out + (size_t)row * N + col] = dmn[i];
    }
  }
}
__global__ void reduce_splitk_with_bias(float* __restrict__ out,
                                        const float* __restrict__ partial,
                                        const __hip_bfloat16* __restrict__ b,
                                        int M, int N, int splits) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M * N;
  if (idx >= total) return;
  float s = 0.f;
  const size_t stride = (size_t)total;
  for (int sp = 0; sp < splits; ++sp) s += partial[(size_t)sp * stride + idx];
  if (b) s += __bfloat162float(b[idx % N]);
  out[idx] = s;
}
template <typename T>
void getp_matmul(float* xout, float* x, T* w, T* b, int n, int d, int batch_size) {
  PROFILE_FUNCTION();
  const int M = batch_size, N = d, K = n;

  constexpr int BM = 32, BN = 32, BK = 32;
  dim3 block(256);
  const int gx = (N + BN - 1) / BN;
  const int gy = (M + BM - 1) / BM;
  const int grid_xy = max(1, gx) * max(1, gy);

  int dev = 0, cu = 104;
  HIP_CHECK(hipGetDevice(&dev));
  hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev);
  if (cu <= 0) cu = 104;

  const int target_cta = cu * 4;
  int splits = (grid_xy >= target_cta) ? 1 : (target_cta + grid_xy - 1) / grid_xy;
  splits = min(8, max(1, splits));
  const int max_splits_by_K = max(1, (K + BK - 1) / BK);
  splits = min(splits, max_splits_by_K);

  const size_t shmem = (size_t)BK * (BM + BN) * sizeof(float);

  if (splits == 1) {
    dim3 grid(gx, gy);
    matmul_kernel_nosplit<<<grid, block, shmem>>>(xout, x,
        reinterpret_cast<const __hip_bfloat16*>(w),
        reinterpret_cast<const __hip_bfloat16*>(b),
        M, N, K);
    return;
  }

  float* partial = nullptr;
  HIP_CHECK(hipMalloc(&partial, (size_t)splits * (size_t)M * (size_t)N * sizeof(float)));

  dim3 grid(gx, gy, splits);
  matmul_kernel_splitk_store<<<grid, block, shmem>>>(
      partial, x, reinterpret_cast<const __hip_bfloat16*>(w),
      M, N, K, splits);

  const int total = M * N;
  dim3 rBlock(256);
  dim3 rGrid((total + rBlock.x - 1) / rBlock.x);
  reduce_splitk_with_bias<<<rGrid, rBlock>>>(xout, partial,
      reinterpret_cast<const __hip_bfloat16*>(b),
      M, N, splits);

  HIP_CHECK(hipFree(partial));
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

__global__ void router_topk_softmax_batch_kernel(
    const float *__restrict__ router_score, int n_experts,
    int experts_per_token, float *__restrict__ topk_v_out,
    int *__restrict__ topk_i_out) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int K = experts_per_token;
  const float eps = 1e-6f;

  router_score += (size_t)b * n_experts;
  topk_v_out += (size_t)b * K;
  topk_i_out += (size_t)b * K;

  extern __shared__ unsigned char smem_raw[];
  float *scores = reinterpret_cast<float *>(smem_raw);
  float *valbuf = scores + n_experts;
  int *idxbuf = reinterpret_cast<int *>(valbuf + blockDim.x);
  float *topv = reinterpret_cast<float *>(idxbuf + blockDim.x);
  int *topi = reinterpret_cast<int *>(topv + GETP_ROUTER_TOPK_MAXK);

  for (int i = tid; i < n_experts; i += blockDim.x) scores[i] = router_score[i];
  __syncthreads();

  for (int sel = 0; sel < K; ++sel) {
    float best = -INFINITY;
    int besti = n_experts;
    for (int i = tid; i < n_experts; i += blockDim.x) {
      float v = scores[i];
      float thr = eps * fmaxf(fabsf(v), fabsf(best));
      if (v > best + thr || (fabsf(v - best) <= thr && i < besti)) {
        best = v;
        besti = i;
      }
    }
    valbuf[tid] = best;
    idxbuf[tid] = besti;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (tid < s) {
        float vr = valbuf[tid + s], vl = valbuf[tid];
        int ir = idxbuf[tid + s], il = idxbuf[tid];
        float thr = eps * fmaxf(fabsf(vr), fabsf(vl));
        bool take_r = (vr > vl + thr) || (fabsf(vr - vl) <= thr && ir < il);
        if (take_r) {
          valbuf[tid] = vr;
          idxbuf[tid] = ir;
        }
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
    for (int i = 1; i < K; ++i)
      if (topv[i] > m) m = topv[i];
    float e[GETP_ROUTER_TOPK_MAXK];
    float s = 0.f;
    for (int i = 0; i < K; ++i) {
      e[i] = expf(topv[i] - m);
      s += e[i];
    }
    float invs = 1.f / s;
    for (int i = 0; i < K; ++i) {
      topk_v_out[i] = e[i] * invs;
      topk_i_out[i] = topi[i];
    }
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
  const float eps = 1e-6f;

  int b = blockIdx.x;
  int tid = threadIdx.x;
  const float *row = logits + (size_t)b * V;

  float mv = -INFINITY;
  int mi = V;
  for (int i = tid; i < V; i += blockDim.x) {
    float v = row[i];
    float thr = eps * fmaxf(fabsf(v), fabsf(mv));
    if (v > mv + thr || (fabsf(v - mv) <= thr && i < mi)) {
      mv = v;
      mi = i;
    }
  }
  smax[tid] = mv;
  sidx[tid] = mi;
  __syncthreads();

  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      float vr = smax[tid + s], vl = smax[tid];
      int ir = sidx[tid + s], il = sidx[tid];
      float thr = eps * fmaxf(fabsf(vr), fabsf(vl));
      bool take_r = (vr > vl + thr) || (fabsf(vr - vl) <= thr && ir < il);
      if (take_r) {
        smax[tid] = vr;
        sidx[tid] = ir;
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
                  DeviceTransformer **dev_transformers, GPUWorker *workers,
                  int token[], int pos) {
  PROFILE_FUNCTION();

  GPUWorker *main_worker = workers; // execute non-expert layers
  int device_index = main_worker->device_index;

  Config *p = &dev_transformers[device_index]->config;
  DeviceTransformerWeights *dev_w = &dev_transformers[device_index]->weights;
  RunState *dev_s = &dev_transformers[device_index]->state;

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

    HIP_CHECK(hipMemset(dev_s->e_agg, 0,
                        (size_t)BATCH_SIZE * hidden_dim * sizeof(float)));

    // TODO: Add async memcpy and kernel launch in MOE using stream
    for (int i = 0; i < EXPERT_PARALLELISM; ++i) {
      GPUWorker *expert_worker = workers + i;
      int experts_per_device = expert_worker->expert_end - expert_worker->expert_start;
      int expert_device_index = expert_worker->device_index;

      DeviceTransformerWeights *expert_dev_w = &dev_transformers[expert_device_index]->weights;
      RunState *expert_dev_s = &dev_transformers[expert_device_index]->state;
      RunStateExt *expert_ext = ext_get(expert_device_index);

      HIP_CHECK(hipSetDevice(expert_device_index));

      if (expert_device_index != device_index) {
        HIP_CHECK(hipMemcpyPeer(expert_dev_s->t, expert_device_index, dev_s->t, device_index, BATCH_SIZE * hidden_dim * sizeof(float)));
        HIP_CHECK(hipMemcpyPeer(expert_dev_s->topk_i, expert_device_index, dev_s->topk_i, device_index, BATCH_SIZE * p->experts_per_token * sizeof(int)));
        HIP_CHECK(hipMemcpyPeer(expert_dev_s->topk_v, expert_device_index, dev_s->topk_v, device_index, BATCH_SIZE * p->experts_per_token * sizeof(float)));
      }

      getp_map_global_to_local_batch(expert_dev_s->topk_i, expert_dev_s->topk_v, expert_ext->local_ids,
                                     expert_ext->local_wts, expert_ext->n_local,
                                     p->experts_per_token, expert_worker->expert_start,
                                     expert_worker->expert_end, BATCH_SIZE);
  
      int cap_pairs = build_moe_buckets_local_pos(
          expert_ext->local_ids, expert_ext->local_wts, expert_ext->n_local, BATCH_SIZE,
          p->experts_per_token, experts_per_device, expert_ext->e_counts, expert_ext->e_offsets,
          expert_ext->e_dev, expert_ext->w_dev, expert_ext->pair_pos);
  
      HIP_CHECK(hipMemset(expert_dev_s->e_agg, 0,
                          (size_t)BATCH_SIZE * hidden_dim * sizeof(float)));
  
      __hip_bfloat16 *w1_base = expert_dev_w->w_mlp1 + 1ll * l * experts_per_device * 2 *
                                                          p->intermediate_dim *
                                                          hidden_dim;
      __hip_bfloat16 *b1_base =
          expert_dev_w->b_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim;
  
      launch_mlp1_swiglu_bf16_bucketed(expert_dev_s->gate_up, expert_dev_s->t, w1_base, b1_base,
                                       expert_ext->e_offsets, expert_ext->e_dev, hidden_dim,
                                       p->intermediate_dim, experts_per_device,
                                       cap_pairs, p->swiglu_limit);
  
      __hip_bfloat16 *w2_base = expert_dev_w->w_mlp2 + 1ll * l * experts_per_device *
                                                          hidden_dim *
                                                          p->intermediate_dim;
      __hip_bfloat16 *b2_base =
          expert_dev_w->b_mlp2 + 1ll * l * experts_per_device * hidden_dim;
  
      launch_mlp2_partial_bf16_bucketed_mfma(
          expert_ext->z_partial, expert_dev_s->gate_up, w2_base, b2_base, expert_ext->w_dev,
          expert_ext->e_offsets, expert_ext->e_dev, p->intermediate_dim, hidden_dim,
          experts_per_device, cap_pairs);
  
      moe_gather_pairs(expert_dev_s->e_agg, expert_ext->z_partial, expert_ext->pair_pos, hidden_dim,
                       p->experts_per_token, BATCH_SIZE);

      if (expert_device_index != device_index) {
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpyPeer(dev_s->tb2, device_index, expert_dev_s->e_agg, expert_device_index, BATCH_SIZE * hidden_dim * sizeof(float)));
        HIP_CHECK(hipSetDevice(device_index));
        getp_vecadd(dev_s->e_agg, dev_s->tb2, hidden_dim, BATCH_SIZE);
        HIP_CHECK(hipDeviceSynchronize());
      }
    }

    HIP_CHECK(hipSetDevice(device_index));
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