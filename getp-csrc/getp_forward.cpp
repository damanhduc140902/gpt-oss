#include <hip/hip_runtime.h>
#include <malloc.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "collectives.hpp"
#include "getp_barrier.hpp"
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

#ifndef GETP_BN_AGG
#define GETP_BN_AGG 4
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
void getp_rmsnorm(float *o, float *x, float *weight, int batch_size, int dim,
                  hipStream_t stream) {
  PROFILE_FUNCTION();
  dim3 blockDim(1024);
  dim3 gridDim(1, batch_size);
  rmsnorm_kernel<<<gridDim, blockDim, sizeof(float) * (blockDim.x + 1),
                   stream>>>(o, x, weight, dim);
  //HIP_CHECK(hipDeviceSynchronize());
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

__device__ inline unsigned short f32_to_bf16bits(float v) {
  __bf16_bits_u t;
  t.b = __float2bfloat16(v);
  return t.u;
}

using f32x4 = __attribute__((vector_size(16))) float;
using bf16x4 = __attribute__((vector_size(8))) unsigned short;

__global__ void kv_store_pair_fp32_to_bf16_kernel(
    const float *__restrict__ k, const float *__restrict__ v,
    __hip_bfloat16 *__restrict__ kdst, __hip_bfloat16 *__restrict__ vdst,
    int elems) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int i4 = i << 2;
  if (i4 + 3 < elems) {
    float4 kf = reinterpret_cast<const float4 *>(k)[i];
    float4 vf = reinterpret_cast<const float4 *>(v)[i];
    uint2 ku, vu;
    ku.x = (unsigned)f32_to_bf16bits(kf.x) |
           ((unsigned)f32_to_bf16bits(kf.y) << 16);
    ku.y = (unsigned)f32_to_bf16bits(kf.z) |
           ((unsigned)f32_to_bf16bits(kf.w) << 16);
    vu.x = (unsigned)f32_to_bf16bits(vf.x) |
           ((unsigned)f32_to_bf16bits(vf.y) << 16);
    vu.y = (unsigned)f32_to_bf16bits(vf.z) |
           ((unsigned)f32_to_bf16bits(vf.w) << 16);
    reinterpret_cast<uint2 *>(kdst)[i] = ku;
    reinterpret_cast<uint2 *>(vdst)[i] = vu;
  } else {
    for (int t = 0; t < 4; ++t) {
      int idx = i4 + t;
      if (idx < elems) {
        reinterpret_cast<unsigned short *>(kdst)[idx] = f32_to_bf16bits(k[idx]);
        reinterpret_cast<unsigned short *>(vdst)[idx] = f32_to_bf16bits(v[idx]);
      }
    }
  }
}
static inline void kv_store_pair_fp32_to_bf16(__hip_bfloat16 *kdst,
                                              __hip_bfloat16 *vdst,
                                              const float *k, const float *v,
                                              int elems, hipStream_t stream) {
  PROFILE_FUNCTION();
                                                int threads = 256;
  int blocks = ((elems + 3) / 4 + threads - 1) / threads;
  kv_store_pair_fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
      k, v, kdst, vdst, elems);
  //HIP_CHECK(hipDeviceSynchronize());
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

__global__ void moe_blocks_per_expert_kernel(const int *offsets, int E, int BM,
                                             int *blk_counts) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  int n = offsets[e + 1] - offsets[e];
  blk_counts[e] = (n + BM - 1) / BM;
}

static inline int build_moe_block_schedule(const int *offsets, int E, int BM,
                                           int *blk_counts, int *blk_offsets,
                                           hipStream_t stream) {
                                            PROFILE_FUNCTION();
  dim3 b(256), g((E + b.x - 1) / b.x);
  moe_blocks_per_expert_kernel<<<g, b, 0, stream>>>(offsets, E, BM, blk_counts);
  exclusive_scan_small_kernel<<<dim3(1), dim3(256), sizeof(int) * 256,
                                stream>>>(blk_counts, blk_offsets, E);
  int total_blocks = 0;
  HIP_CHECK(hipMemcpyAsync(&total_blocks, blk_offsets + E, sizeof(int),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  //HIP_CHECK(hipDeviceSynchronize());
  return total_blocks;
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

static inline int build_moe_buckets_local_pos(
    const int *local_ids, const float *local_wts, const int *n_local, int B,
    int K, int E, int *e_counts, int *e_offsets, int *tok_idx_out, float *w_out,
    int *pair_pos, hipStream_t stream) {
      PROFILE_FUNCTION();
  HIP_CHECK(hipMemsetAsync(e_counts, 0, sizeof(int) * E, stream));
  moe_count_local_kernel<<<dim3(B), dim3(128), 0, stream>>>(local_ids, n_local,
                                                            B, K, E, e_counts);
  exclusive_scan_small_kernel<<<dim3(1), dim3(256), sizeof(int) * 256,
                                stream>>>(e_counts, e_offsets, E);
  std::vector<int> h_off(E + 1);
  HIP_CHECK(hipMemcpyAsync(h_off.data(), e_offsets, sizeof(int) * (E + 1),
                           hipMemcpyDeviceToHost, stream));
  // TODO: get rid of stream synchronize,
  // although the synchorization is not a big bottleneck
  HIP_CHECK(hipStreamSynchronize(stream));
  int cap_pairs = 0;
  for (int i = 0; i < E; ++i)
    cap_pairs = MAX(cap_pairs, h_off[i + 1] - h_off[i]);
  HIP_CHECK(hipMemsetAsync(e_counts, 0, sizeof(int) * E, stream));
  moe_fill_local_pos_kernel<<<dim3(B), dim3(128), 0, stream>>>(
      local_ids, local_wts, n_local, B, K, E, e_offsets, e_counts, tok_idx_out,
      w_out, pair_pos);
      //HIP_CHECK(hipDeviceSynchronize());
  return cap_pairs;
}

__device__ inline uint4 pack8_f32_to_bf16(float4 a, float4 b) {
  uint4 u;
  u.x = (unsigned)f32_to_bf16bits(a.x) | ((unsigned)f32_to_bf16bits(a.y) << 16);
  u.y = (unsigned)f32_to_bf16bits(a.z) | ((unsigned)f32_to_bf16bits(a.w) << 16);
  u.z = (unsigned)f32_to_bf16bits(b.x) | ((unsigned)f32_to_bf16bits(b.y) << 16);
  u.w = (unsigned)f32_to_bf16bits(b.z) | ((unsigned)f32_to_bf16bits(b.w) << 16);
  return u;
}

__global__ void moe_scatter_acts_to_expert_bf16_kernel(__hip_bfloat16 *a_in,
                                                       const float *x,
                                                       const int *tok_idx,
                                                       int total_pairs, int H) {
  int pos = blockIdx.x;
  if (pos >= total_pairs) return;
  int b = tok_idx[pos];
  const float *__restrict__ src = x + (size_t)b * H;
  __hip_bfloat16 *__restrict__ dst = a_in + (size_t)pos * H;

  int t = threadIdx.x;
  int step8 = blockDim.x * 8;
  int H8 = H & ~7;
  for (int kk8 = t * 8; kk8 < H8; kk8 += step8) {
    const float4 *p4 = reinterpret_cast<const float4 *>(src + kk8);
    float4 a = p4[0], b4 = p4[1];
    uint4 u = pack8_f32_to_bf16(a, b4);
    reinterpret_cast<uint4 *>(dst)[kk8 >> 3] = u;
  }
  for (int kk = H8 + t; kk < H; kk += blockDim.x)
    reinterpret_cast<unsigned short *>(dst)[kk] = f32_to_bf16bits(src[kk]);
}

static inline void moe_scatter_acts_to_expert_bf16(__hip_bfloat16 *a_in,
                                                   const float *x,
                                                   const int *tok_idx,
                                                   int total_pairs, int H,
                                                   hipStream_t stream) {
  PROFILE_FUNCTION();
  dim3 block(256), grid(total_pairs);
  moe_scatter_acts_to_expert_bf16_kernel<<<grid, block, 0, stream>>>(
      a_in, x, tok_idx, total_pairs, H);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void __launch_bounds__(256) mlp1_swiglu_bf16_bucketed_kernel_outbf16(
    __hip_bfloat16 *__restrict__ gate_up_bf16,
    const __hip_bfloat16 *__restrict__ A_in,
    const __hip_bfloat16 *__restrict__ W1,
    const __hip_bfloat16 *__restrict__ B1, const int *__restrict__ offsets,
    const int *__restrict__ blk_offs, int H, int I, int E, float swiglu_limit) {
  // constexpr int BM = 32, BN = 32, BK = 32, WM = 16, WN = 16;
  // const int BNt = BN * GETP_BN_AGG;
  // const int lane = threadIdx.x & 63;
  // const int wave = threadIdx.x >> 6;
  // const int xTile = wave % (BM / WM);
  // const int yTile = wave / (BM / WM);
  // const int xMF = lane & 15;
  // const int yMF = lane >> 4;
  constexpr int BM = 64, BN = 16, BK = 32, WM = 16, WN = 16;
const int BNt = BN * GETP_BN_AGG;
const int lane = threadIdx.x & 63;
const int wave = threadIdx.x >> 6;
const int splitM = BM / WM;
const int splitN = BN / WN;
const int xTile = wave % splitN;
const int yTile = wave / splitN;
const int xMF = lane & 15;
const int yMF = lane >> 4;


  int by = blockIdx.y;
  int lo = 0, hi = E;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    int off = blk_offs[mid];
    if (by < off)
      hi = mid;
    else
      lo = mid;
  }
  int e = lo;
  int blk = by - blk_offs[e];

  const int s = offsets[e], t = offsets[e + 1];
  const int n = t - s;
  const int base_all = blk * BM;
  if (base_all >= n) return;
  const int base = s + base_all;
  const int tcnt = min(BM, n - base_all);

  const size_t w1_e = (size_t)e * (size_t)(2 * I) * (size_t)H;
  const size_t b1_e = (size_t)e * (size_t)(2 * I);

  extern __shared__ unsigned short sm[];
  unsigned short *sx = sm;
  unsigned short *swg = sx + (size_t)BK * BM;
  unsigned short *swu = swg + (size_t)BK * BNt;

  f32x4 dg0 = {0, 0, 0, 0}, dg1 = {0, 0, 0, 0}, dg2 = {0, 0, 0, 0},
        dg3 = {0, 0, 0, 0};
  f32x4 du0 = {0, 0, 0, 0}, du1 = {0, 0, 0, 0}, du2 = {0, 0, 0, 0},
        du3 = {0, 0, 0, 0};

  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  constexpr int strideX = 256 / (BK / 4);
  constexpr int strideW = 256 / (BK / 8);
  const int colBase = blockIdx.x * BNt;

  for (int bk = 0; bk < H; bk += BK) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk = bk + 4 * loadX_x;
      const int r = loadX_y + off;
      const int pos = base + r;
      if (r < tcnt) {
        const __hip_bfloat16 *p = A_in + (size_t)pos * H + kk;
        if (kk + 3 < H) {
          uint2 v = *reinterpret_cast<const uint2 *>(p);
          sx[(4 * loadX_x + 0) * BM + r] = (unsigned short)(v.x & 0xFFFF);
          sx[(4 * loadX_x + 1) * BM + r] = (unsigned short)(v.x >> 16);
          sx[(4 * loadX_x + 2) * BM + r] = (unsigned short)(v.y & 0xFFFF);
          sx[(4 * loadX_x + 3) * BM + r] = (unsigned short)(v.y >> 16);
        } else {
          const unsigned short *q = reinterpret_cast<const unsigned short *>(p);
          sx[(4 * loadX_x + 0) * BM + r] = (kk + 0 < H) ? q[0] : 0;
          sx[(4 * loadX_x + 1) * BM + r] = (kk + 1 < H) ? q[1] : 0;
          sx[(4 * loadX_x + 2) * BM + r] = (kk + 2 < H) ? q[2] : 0;
          sx[(4 * loadX_x + 3) * BM + r] = (kk + 3 < H) ? q[3] : 0;
        }
      }
    }
    for (int off = 0; off < BNt; off += strideW) {
      if (loadW_y + off >= BNt) break;
      const int kk = bk + 8 * loadW_x;
      const int col = colBase + loadW_y + off;
      uint4 wg = make_uint4(0, 0, 0, 0), wu = make_uint4(0, 0, 0, 0);
      if (col < I) {
        if (kk + 7 < H) {
          wg = *reinterpret_cast<const uint4 *>(W1 + w1_e +
                                                (size_t)(2 * col + 0) * H + kk);
          wu = *reinterpret_cast<const uint4 *>(W1 + w1_e +
                                                (size_t)(2 * col + 1) * H + kk);
        } else if (kk < H) {
          const __hip_bfloat16 *pg = W1 + w1_e + (size_t)(2 * col + 0) * H + kk;
          const __hip_bfloat16 *pu = W1 + w1_e + (size_t)(2 * col + 1) * H + kk;
          unsigned u0 = 0, u1 = 0, u2 = 0, u3 = 0, v0 = 0, v1 = 0, v2 = 0,
                   v3 = 0;
          if (kk + 0 < H) {
            u0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[0]);
            v0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[0]);
          }
          if (kk + 1 < H) {
            u0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[1])
                  << 16;
            v0 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[1])
                  << 16;
          }
          if (kk + 2 < H) {
            u1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[2]);
            v1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[2]);
          }
          if (kk + 3 < H) {
            u1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[3])
                  << 16;
            v1 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[3])
                  << 16;
          }
          if (kk + 4 < H) {
            u2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[4]);
            v2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[4]);
          }
          if (kk + 5 < H) {
            u2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[5])
                  << 16;
            v2 |= (unsigned)(reinterpret_cast<const unsigned short *>(pu)[5])
                  << 16;
          }
          if (kk + 6 < H) {
            u3 |= (unsigned)(reinterpret_cast<const unsigned short *>(pg)[6]);
          }
          wg = make_uint4(u0, u1, u2, u3);
          wu = make_uint4(v0, v1, v2, v3);
        }
      }
      unsigned short *dg = swg + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
      unsigned short *du = swu + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
      if (col < I) {
        dg[0 * BNt] = (unsigned short)(wg.x & 0xFFFF);
        dg[1 * BNt] = (unsigned short)(wg.x >> 16);
        dg[2 * BNt] = (unsigned short)(wg.y & 0xFFFF);
        dg[3 * BNt] = (unsigned short)(wg.y >> 16);
        dg[4 * BNt] = (unsigned short)(wg.z & 0xFFFF);
        dg[5 * BNt] = (unsigned short)(wg.z >> 16);
        dg[6 * BNt] = (unsigned short)(wg.w & 0xFFFF);
        dg[7 * BNt] = (unsigned short)(wg.w >> 16);
        du[0 * BNt] = (unsigned short)(wu.x & 0xFFFF);
        du[1 * BNt] = (unsigned short)(wu.x >> 16);
        du[2 * BNt] = (unsigned short)(wu.y & 0xFFFF);
        du[3 * BNt] = (unsigned short)(wu.y >> 16);
        du[4 * BNt] = (unsigned short)(wu.z & 0xFFFF);
        du[5 * BNt] = (unsigned short)(wu.z >> 16);
        du[6 * BNt] = (unsigned short)(wu.w & 0xFFFF);
        du[7 * BNt] = (unsigned short)(wu.w >> 16);
      } else {
        for (int t = 0; t < 8; ++t) {
          dg[t * BNt] = 0;
          du[t * BNt] = 0;
        }
      }
    }
    __syncthreads();

    for (int k = 0; k < BK; k += 16) {
      const int basek = k + yMF * 4;
      const int row = yTile * WM + xMF;
      const int c = xTile * WN + xMF;
      bf16x4 av = {sx[(basek + 0) * BM + row], sx[(basek + 1) * BM + row],
                   sx[(basek + 2) * BM + row], sx[(basek + 3) * BM + row]};
      bf16x4 g0 = {swg[(basek + 0) * BNt + c + 0 * BN],
                   swg[(basek + 1) * BNt + c + 0 * BN],
                   swg[(basek + 2) * BNt + c + 0 * BN],
                   swg[(basek + 3) * BNt + c + 0 * BN]};
      bf16x4 g1 = {swg[(basek + 0) * BNt + c + 1 * BN],
                   swg[(basek + 1) * BNt + c + 1 * BN],
                   swg[(basek + 2) * BNt + c + 1 * BN],
                   swg[(basek + 3) * BNt + c + 1 * BN]};
      bf16x4 g2 = {swg[(basek + 0) * BNt + c + 2 * BN],
                   swg[(basek + 1) * BNt + c + 2 * BN],
                   swg[(basek + 2) * BNt + c + 2 * BN],
                   swg[(basek + 3) * BNt + c + 2 * BN]};
      bf16x4 g3 = {swg[(basek + 0) * BNt + c + 3 * BN],
                   swg[(basek + 1) * BNt + c + 3 * BN],
                   swg[(basek + 2) * BNt + c + 3 * BN],
                   swg[(basek + 3) * BNt + c + 3 * BN]};
      bf16x4 u0 = {swu[(basek + 0) * BNt + c + 0 * BN],
                   swu[(basek + 1) * BNt + c + 0 * BN],
                   swu[(basek + 2) * BNt + c + 0 * BN],
                   swu[(basek + 3) * BNt + c + 0 * BN]};
      bf16x4 u1 = {swu[(basek + 0) * BNt + c + 1 * BN],
                   swu[(basek + 1) * BNt + c + 1 * BN],
                   swu[(basek + 2) * BNt + c + 1 * BN],
                   swu[(basek + 3) * BNt + c + 1 * BN]};
      bf16x4 u2 = {swu[(basek + 0) * BNt + c + 2 * BN],
                   swu[(basek + 1) * BNt + c + 2 * BN],
                   swu[(basek + 2) * BNt + c + 2 * BN],
                   swu[(basek + 3) * BNt + c + 2 * BN]};
      bf16x4 u3 = {swu[(basek + 0) * BNt + c + 3 * BN],
                   swu[(basek + 1) * BNt + c + 3 * BN],
                   swu[(basek + 2) * BNt + c + 3 * BN],
                   swu[(basek + 3) * BNt + c + 3 * BN]};
      dg0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, g0, dg0, 0, 0, 0);
      dg1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, g1, dg1, 0, 0, 0);
      dg2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, g2, dg2, 0, 0, 0);
      dg3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, g3, dg3, 0, 0, 0);
      du0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, u0, du0, 0, 0, 0);
      du1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, u1, du1, 0, 0, 0);
      du2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, u2, du2, 0, 0, 0);
      du3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, u3, du3, 0, 0, 0);
    }
    __syncthreads();
  }

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  for (int i = 0; i < 4; ++i) {
    const int row = yTile * WM + (yD + i);
    if (row >= tcnt) continue;
    const int pos = base + row;
    auto store_group = [&](int col, float gval, float uval) {
      if (col >= I) return;
      const float bg =
          B1 ? __bfloat162float(B1[b1_e + (size_t)(2 * col + 0)]) : 0.f;
      const float bu =
          B1 ? __bfloat162float(B1[b1_e + (size_t)(2 * col + 1)]) : 0.f;
      float g = gval + bg, u = uval + bu;
      if (g > swiglu_limit) g = swiglu_limit;
      if (u > swiglu_limit) u = swiglu_limit;
      if (u < -swiglu_limit) u = -swiglu_limit;
      const float a = 1.702f;
      float s = 1.f / (1.f + __expf(-a * g));
      float out = (g * s) * (u + 1.f);
      reinterpret_cast<unsigned short *>(gate_up_bf16)[(size_t)pos * I + col] =
          f32_to_bf16bits(out);
    };
    const int colBase = blockIdx.x * BNt;
    const int c0 = colBase + xTile * WN + xD + 0 * BN;
    const int c1 = colBase + xTile * WN + xD + 1 * BN;
    const int c2 = colBase + xTile * WN + xD + 2 * BN;
    const int c3 = colBase + xTile * WN + xD + 3 * BN;
    store_group(c0, dg0[i], du0[i]);
    store_group(c1, dg1[i], du1[i]);
    store_group(c2, dg2[i], du2[i]);
    store_group(c3, dg3[i], du3[i]);
  }
}
static inline void launch_mlp1_swiglu_bf16_bucketed_outbf16(
    __hip_bfloat16 *gate_up_bf16, const __hip_bfloat16 *a_in,
    const __hip_bfloat16 *w1_layer, const __hip_bfloat16 *b1_layer,
    const int *offsets, const int *blk_offs, int H, int I, int E,
    int total_blocks, float swiglu_limit, hipStream_t stream) {
  PROFILE_FUNCTION();
  // constexpr int BM = 32, BN = 32, BK = 32;
  constexpr int BM = 64, BN = 16, BK = 32;

  const int BNt = BN * GETP_BN_AGG;
  dim3 block(256);
  dim3 grid((I + BNt - 1) / BNt, total_blocks);
  size_t shmem = (size_t)BK * ((size_t)BM + (size_t)BNt + (size_t)BNt) *
                 sizeof(unsigned short);
  mlp1_swiglu_bf16_bucketed_kernel_outbf16<<<grid, block, shmem, stream>>>(
      gate_up_bf16, a_in, w1_layer, b1_layer, offsets, blk_offs, H, I, E,
      swiglu_limit);
  //HIP_CHECK(hipDeviceSynchronize());
  
}

__global__ void __launch_bounds__(256)
    mlp2_partial_bf16_bucketed_splitk_kernel_inbf16(
        float *__restrict__ z_partial,
        const __hip_bfloat16 *__restrict__ gate_up_bf16,
        const __hip_bfloat16 *__restrict__ W2,
        const __hip_bfloat16 *__restrict__ B2,
        const float *__restrict__ w_by_bucket, const int *__restrict__ offsets,
        const int *__restrict__ blk_offs, const int *__restrict__ tok_idx,
        int I, int H, int E, int splits) {
  // constexpr int BM = 32, BN = 32, BK = 32, WM = 16, WN = 16;
  // const int BNt = BN * GETP_BN_AGG;
  // const int lane = threadIdx.x & 63;
  // const int wave = threadIdx.x >> 6;
  // const int xTile = wave % (BM / WM);
  // const int yTile = wave / (BM / WM);
  // const int xMF = lane & 15;
  // const int yMF = lane >> 4;
  constexpr int BM = 64, BN = 16, BK = 32, WM = 16, WN = 16;
const int BNt = BN * GETP_BN_AGG;
const int lane = threadIdx.x & 63;
const int wave = threadIdx.x >> 6;
const int splitM = BM / WM;
const int splitN = BN / WN;
const int xTile = wave % splitN;
const int yTile = wave / splitN;
const int xMF = lane & 15;
const int yMF = lane >> 4;


  extern __shared__ unsigned short smem_raw_mlp2[];
  unsigned short *sx = smem_raw_mlp2;
  unsigned short *sw = sx + (size_t)BK * BM;

  int by = blockIdx.y;
  int lo = 0, hi = E;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    int off = blk_offs[mid];
    if (by < off)
      hi = mid;
    else
      lo = mid;
  }
  int e = lo;
  int blk = by - blk_offs[e];

  const int s = offsets[e], t = offsets[e + 1];
  const int n = t - s;
  const int base_all = blk * BM;
  if (base_all >= n) return;
  const int base = s + base_all;
  const int tcnt = min(BM, n - base_all);

  const size_t w2_e = (size_t)e * (size_t)H * (size_t)I;
  const size_t b2_e = (size_t)e * (size_t)H;

  f32x4 d0 = {0, 0, 0, 0}, d1 = {0, 0, 0, 0}, d2 = {0, 0, 0, 0},
        d3 = {0, 0, 0, 0};

  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  constexpr int strideX = 256 / (BK / 4);
  constexpr int strideW = 256 / (BK / 8);

  const int split = blockIdx.z;
  const int kChunk = (I + splits - 1) / splits;
  const int kBeg = split * kChunk;
  const int kEnd = min(I, kBeg + kChunk);

  const int colBase = blockIdx.x * BNt;

  for (int bk = kBeg; bk < kEnd; bk += BK) {
    for (int off = 0; off; off += 0) {
      break;
    }
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk = bk + 4 * loadX_x;
      const int r = loadX_y + off;
      const int pos = base + r;
      if (r < tcnt) {
        const __hip_bfloat16 *src = gate_up_bf16 + (size_t)pos * (size_t)I + kk;
        if (kk + 3 < kEnd) {
          uint2 v = *reinterpret_cast<const uint2 *>(src);
          sx[(4 * loadX_x + 0) * BM + r] = (unsigned short)(v.x & 0xFFFF);
          sx[(4 * loadX_x + 1) * BM + r] = (unsigned short)(v.x >> 16);
          sx[(4 * loadX_x + 2) * BM + r] = (unsigned short)(v.y & 0xFFFF);
          sx[(4 * loadX_x + 3) * BM + r] = (unsigned short)(v.y >> 16);
        } else {
          const unsigned short *q =
              reinterpret_cast<const unsigned short *>(src);
          sx[(4 * loadX_x + 0) * BM + r] = (kk + 0 < kEnd) ? q[0] : 0;
          sx[(4 * loadX_x + 1) * BM + r] = (kk + 1 < kEnd) ? q[1] : 0;
          sx[(4 * loadX_x + 2) * BM + r] = (kk + 2 < kEnd) ? q[2] : 0;
          sx[(4 * loadX_x + 3) * BM + r] = (kk + 3 < kEnd) ? q[3] : 0;
        }
      }
    }
    for (int off = 0; off < BNt; off += strideW) {
      if (loadW_y + off >= BNt) break;
      const int kk = bk + 8 * loadW_x;
      const int col = colBase + loadW_y + off;
      uint4 wb = make_uint4(0, 0, 0, 0);
      if (col < H && kk < kEnd) {
        if (kk + 7 < kEnd) {
          wb = *reinterpret_cast<const uint4 *>(W2 + w2_e +
                                                (size_t)col * (size_t)I + kk);
        } else {
          const __hip_bfloat16 *src = W2 + w2_e + (size_t)col * (size_t)I + kk;
          unsigned u0 = 0, u1 = 0, u2 = 0, u3 = 0;
          if (kk + 0 < kEnd)
            u0 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[0]);
          if (kk + 1 < kEnd)
            u0 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[1])
                  << 16;
          if (kk + 2 < kEnd)
            u1 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[2]);
          if (kk + 3 < kEnd)
            u1 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[3])
                  << 16;
          if (kk + 4 < kEnd)
            u2 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[4]);
          if (kk + 5 < kEnd)
            u2 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[5])
                  << 16;
          if (kk + 6 < kEnd)
            u3 |= (unsigned)(reinterpret_cast<const unsigned short *>(src)[6]);
          wb = make_uint4(u0, u1, u2, u3);
        }
      }
      unsigned short *dst = sw + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
      if (col < H) {
        dst[0 * BNt] = (unsigned short)(wb.x & 0xFFFF);
        dst[1 * BNt] = (unsigned short)(wb.x >> 16);
        dst[2 * BNt] = (unsigned short)(wb.y & 0xFFFF);
        dst[3 * BNt] = (unsigned short)(wb.y >> 16);
        dst[4 * BNt] = (unsigned short)(wb.z & 0xFFFF);
        dst[5 * BNt] = (unsigned short)(wb.z >> 16);
        dst[6 * BNt] = (unsigned short)(wb.w & 0xFFFF);
        dst[7 * BNt] = (unsigned short)(wb.w >> 16);
      } else {
        for (int t = 0; t < 8; ++t) dst[t * BNt] = 0;
      }
    }
    __syncthreads();

    for (int k = 0; k < BK; k += 16) {
      const int basek = k + yMF * 4;
      const int row = yTile * WM + xMF;
      const int c = xTile * WN + xMF;
      bf16x4 av = {sx[(basek + 0) * BM + row], sx[(basek + 1) * BM + row],
                   sx[(basek + 2) * BM + row], sx[(basek + 3) * BM + row]};
      bf16x4 b0 = {sw[(basek + 0) * BNt + c + 0 * BN],
                   sw[(basek + 1) * BNt + c + 0 * BN],
                   sw[(basek + 2) * BNt + c + 0 * BN],
                   sw[(basek + 3) * BNt + c + 0 * BN]};
      bf16x4 b1 = {sw[(basek + 0) * BNt + c + 1 * BN],
                   sw[(basek + 1) * BNt + c + 1 * BN],
                   sw[(basek + 2) * BNt + c + 1 * BN],
                   sw[(basek + 3) * BNt + c + 1 * BN]};
      bf16x4 b2 = {sw[(basek + 0) * BNt + c + 2 * BN],
                   sw[(basek + 1) * BNt + c + 2 * BN],
                   sw[(basek + 2) * BNt + c + 2 * BN],
                   sw[(basek + 3) * BNt + c + 2 * BN]};
      bf16x4 b3 = {sw[(basek + 0) * BNt + c + 3 * BN],
                   sw[(basek + 1) * BNt + c + 3 * BN],
                   sw[(basek + 2) * BNt + c + 3 * BN],
                   sw[(basek + 3) * BNt + c + 3 * BN]};
      d0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b0, d0, 0, 0, 0);
      d1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b1, d1, 0, 0, 0);
      d2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b2, d2, 0, 0, 0);
      d3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b3, d3, 0, 0, 0);
    }
    __syncthreads();
  }

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  for (int i = 0; i < 4; ++i) {
    const int row = yTile * WM + (yD + i);
    if (row >= tcnt) continue;
    const int pos = base + row;
    const float w = w_by_bucket[pos];
    auto store_group = [&](int col, float val) {
      if (col >= H) return;
      const float bia = B2 ? __bfloat162float(B2[b2_e + col]) : 0.f;
      float *dst = z_partial + (size_t)pos * (size_t)H + col;
      if (splits == 1)
        *dst = (val + bia) * w;
      else {
        atomicAdd(dst, val * w);
        if (blockIdx.z == 0) atomicAdd(dst, bia * w);
      }
    };
    const int colBase = blockIdx.x * BNt;
    const int c0 = colBase + xTile * WN + xD + 0 * BN;
    const int c1 = colBase + xTile * WN + xD + 1 * BN;
    const int c2 = colBase + xTile * WN + xD + 2 * BN;
    const int c3 = colBase + xTile * WN + xD + 3 * BN;
    store_group(c0, d0[i]);
    store_group(c1, d1[i]);
    store_group(c2, d2[i]);
    store_group(c3, d3[i]);
  }
}
static inline void launch_mlp2_partial_bf16_bucketed_frombf16(
    float *z_partial, const __hip_bfloat16 *gate_up_bf16,
    const __hip_bfloat16 *w2_layer, const __hip_bfloat16 *b2_layer,
    const float *w_by_bucket, const int *offsets, const int *blk_offs,
    const int *tok_idx, int I, int H, int E, int total_blocks,
    hipStream_t stream) {
  PROFILE_FUNCTION();
  // constexpr int BN = 32, BM = 32, BK = 32;
  constexpr int BN = 16, BM = 64, BK = 32;

  const int BNt = BN * GETP_BN_AGG;
  const int gx = (H + BNt - 1) / BNt;
  const int gy = total_blocks;
  const int grid_xy = max(1, gx) * max(1, gy);

  int dev = 0, cu = 104;
  HIP_CHECK(hipGetDevice(&dev));
  HIP_CHECK(
      hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev));
  if (cu <= 0) cu = 104;

  const int target_cta = cu * 4;
  int splits =
      (grid_xy >= target_cta) ? 1 : (target_cta + grid_xy - 1) / grid_xy;
  splits = min(8, max(1, splits));
  const int max_splits_by_K = max(1, (I + BK - 1) / BK);
  splits = min(splits, max_splits_by_K);

  if (splits > 1) {
    int total_pairs = 0;
    HIP_CHECK(hipMemcpyAsync(&total_pairs, offsets + E, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemsetAsync(
        z_partial, 0, (size_t)total_pairs * (size_t)H * sizeof(float), stream));
  }

  dim3 block(256);
  dim3 grid(gx, gy, splits);
  size_t shmem =
      (size_t)BK * ((size_t)BM + (size_t)BNt) * sizeof(unsigned short);

  mlp2_partial_bf16_bucketed_splitk_kernel_inbf16<<<grid, block, shmem,
                                                    stream>>>(
      z_partial, gate_up_bf16, w2_layer, b2_layer, w_by_bucket, offsets,
      blk_offs, tok_idx, I, H, E, splits);
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
  for (int k = 0; k < K; ++k) {
    int pos = pair_pos[base + k];
    if (pos >= 0) s += z_partial[(size_t)pos * (size_t)H + row];
  }
  e_agg[(size_t)b * (size_t)H + row] += s;
}

static inline void moe_gather_pairs(float *e_agg, const float *z_partial,
                                    const int *pair_pos, int H, int K, int B,
                                    hipStream_t stream) {
  PROFILE_FUNCTION();
  dim3 blk(256), grd((H + blk.x - 1) / blk.x, B);
  moe_gather_pairs_kernel<<<grd, blk, 0, stream>>>(e_agg, z_partial, pair_pos,
                                                   H, K, B);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void __launch_bounds__(256)
new_matmul_qkv_fused_kernel(
  float *q_out, float *k_out,
  float *v_out, float *x,
  __hip_bfloat16 *w, __hip_bfloat16 *b,
  int n, int q_len, int k_len, int v_len, int batch_size
) {
  constexpr int BLOCK_SIZE = 256;

  const int M = batch_size;
  const int N = q_len + k_len + v_len;
  const int K = n;

  // MFMA config
  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;
  
  constexpr int GPRs_A  = 2;
  constexpr int GPRs_B  = 2;
  constexpr int GPRs_CD = 4;

  using CDfloat = __attribute__( (__vector_size__(GPRs_CD * sizeof(float)) )) float;
  using Afloat  = __attribute__( (__vector_size__(GPRs_A  * sizeof(float)) )) unsigned short;
  using Bfloat  = __attribute__( (__vector_size__(GPRs_B  * sizeof(float)) )) unsigned short;

  // Block tile
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  // Wave tile
  constexpr int WM = 64;
  constexpr int WN = 64;

  constexpr int nIterWaveM = WM / MFMA_M;
  constexpr int nIterWaveN = WN / MFMA_N;

  const int waveIdx = threadIdx.x / warpSize;
  const int xInBlockTile = waveIdx % (BM / WM);
  const int yInBlockTile = waveIdx / (BM / WM);
  const int laneIdx = threadIdx.x % warpSize;

  const int Ai = (laneIdx % 16);
  const int Ak = 4 * (laneIdx / 16);
  const int Bj = (laneIdx % 16);
  const int Bk = 4 * (laneIdx / 16);

  constexpr int nbLoadsX = BM * (BK / 4) / BLOCK_SIZE;
  constexpr int nbLoadsW = BN * (BK / 8) / BLOCK_SIZE;

  const int loadXIdx_y = threadIdx.x / (BK / 4);
  const int loadXIdx_x = threadIdx.x % (BK / 4);
  const int loadWIdx_y = threadIdx.x / (BK / 8);
  const int loadWIdx_x = threadIdx.x % (BK / 8);
  constexpr int strideX = BLOCK_SIZE / (BK / 4);
  constexpr int strideW = BLOCK_SIZE / (BK / 8);

  assert(blockDim.x == BLOCK_SIZE);

  __shared__ unsigned short xs[BK][BM];
  __shared__ unsigned short ws[BK][BN];

  Afloat x_local;
  Bfloat w_local;

  CDfloat acc_local[nIterWaveM * nIterWaveN] = {0};

  // Load x and w from global memory to shared memory
  for (int i = 0; i < nbLoadsX; ++i) {
    int offset = i * strideX;
    int index_x = 0 + 4 * loadXIdx_x;
    int index_y = BM * blockIdx.y + loadXIdx_y + offset;
    float4 tmp = index_x < K && index_y < M ?
      *reinterpret_cast<float4 *>(&x[index_y * K + index_x]) : make_float4(0.f, 0.f, 0.f, 0.f);
    xs[4 * loadXIdx_x + 0][loadXIdx_y + offset] = f32_to_bf16bits(tmp.x);
    xs[4 * loadXIdx_x + 1][loadXIdx_y + offset] = f32_to_bf16bits(tmp.y);
    xs[4 * loadXIdx_x + 2][loadXIdx_y + offset] = f32_to_bf16bits(tmp.z);
    xs[4 * loadXIdx_x + 3][loadXIdx_y + offset] = f32_to_bf16bits(tmp.w);
  }
  for (int i = 0; i < nbLoadsW; ++i) {
    int offset = i * strideW;
    int index_x = 0 + 8 * loadWIdx_x;
    int index_y = BN * blockIdx.x + loadWIdx_y + offset;
    uint4 tmp = index_x < K && index_y < N ?
      *reinterpret_cast<uint4 *>(&w[index_y * K + index_x]) : uint4(0, 0, 0, 0);
    ws[8 * loadWIdx_x + 0][loadWIdx_y + offset] = (unsigned short)(tmp.x & 0xFFFF);
    ws[8 * loadWIdx_x + 1][loadWIdx_y + offset] = (unsigned short)(tmp.x >> 16);
    ws[8 * loadWIdx_x + 2][loadWIdx_y + offset] = (unsigned short)(tmp.y & 0xFFFF);
    ws[8 * loadWIdx_x + 3][loadWIdx_y + offset] = (unsigned short)(tmp.y >> 16);
    ws[8 * loadWIdx_x + 4][loadWIdx_y + offset] = (unsigned short)(tmp.z & 0xFFFF);
    ws[8 * loadWIdx_x + 5][loadWIdx_y + offset] = (unsigned short)(tmp.z >> 16);
    ws[8 * loadWIdx_x + 6][loadWIdx_y + offset] = (unsigned short)(tmp.w & 0xFFFF);
    ws[8 * loadWIdx_x + 7][loadWIdx_y + offset] = (unsigned short)(tmp.w >> 16);
  }
  __syncthreads();
  
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Prefetch
    float4  regX[nbLoadsX];
    uint4   regW[nbLoadsW];
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int offset = i * strideX;
        int index_x = bkIdx + BK + 4 * loadXIdx_x;
        int index_y = BM * blockIdx.y + loadXIdx_y + offset;
        regX[i] = index_x < K && index_y < M ?
          *reinterpret_cast<float4 *>(&x[index_y * K + index_x]) : make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int offset = i * strideW;
        int index_x = bkIdx + BK + 8 * loadWIdx_x;
        int index_y = BN * blockIdx.x + loadWIdx_y + offset;
        regW[i] = index_x < K && index_y < N ?
          *reinterpret_cast<uint4 *>(&w[index_y * K + index_x]) : uint4(0, 0, 0, 0);
      }
    }

    // Computing
    for (int k = 0; k < BK; k += MFMA_K) {
      for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
        for (int iterWaveN = 0; iterWaveN < nIterWaveN; ++iterWaveN) {

          for (int i = 0; i < 4; ++i) {
            x_local[i] = xs[k + Ak + i][yInBlockTile * WM + MFMA_M * iterWaveM + Ai];
            w_local[i] = ws[k + Bk + i][xInBlockTile * WN + MFMA_N * iterWaveN + Bj];
          }

          CDfloat y_local = acc_local[iterWaveM * nIterWaveN + iterWaveN]; 
          y_local  = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(x_local, w_local, y_local, 0, 0, 0);
          acc_local[iterWaveM * nIterWaveN + iterWaveN] = y_local;
        }
      }
    }
    __syncthreads();
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int offset = i * strideX;
        xs[4 * loadXIdx_x + 0][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].x);
        xs[4 * loadXIdx_x + 1][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].y);
        xs[4 * loadXIdx_x + 2][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].z);
        xs[4 * loadXIdx_x + 3][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].w);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int offset = i * strideW;
        ws[8 * loadWIdx_x + 0][loadWIdx_y + offset] = (unsigned short)(regW[i].x & 0xFFFF);
        ws[8 * loadWIdx_x + 1][loadWIdx_y + offset] = (unsigned short)(regW[i].x >> 16);
        ws[8 * loadWIdx_x + 2][loadWIdx_y + offset] = (unsigned short)(regW[i].y & 0xFFFF);
        ws[8 * loadWIdx_x + 3][loadWIdx_y + offset] = (unsigned short)(regW[i].y >> 16);
        ws[8 * loadWIdx_x + 4][loadWIdx_y + offset] = (unsigned short)(regW[i].z & 0xFFFF);
        ws[8 * loadWIdx_x + 5][loadWIdx_y + offset] = (unsigned short)(regW[i].z >> 16);
        ws[8 * loadWIdx_x + 6][loadWIdx_y + offset] = (unsigned short)(regW[i].w & 0xFFFF);
        ws[8 * loadWIdx_x + 7][loadWIdx_y + offset] = (unsigned short)(regW[i].w >> 16);
      }
    }
    __syncthreads();
  }

  for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
    for (int iterWaveN = 0; iterWaveN < nIterWaveN; ++iterWaveN) {
      int acc_index = iterWaveM * nIterWaveN + iterWaveN;
      for (int i = 0; i < GPRs_CD; ++i) {
        int CDi = 4 * (laneIdx / 16) + (i % 4);
        int CDj = (laneIdx % 16);
        int global_x = blockIdx.x * BN + WN * xInBlockTile + iterWaveN * MFMA_N + CDj;
        int global_y = blockIdx.y * BM + WM * yInBlockTile + iterWaveM * MFMA_M + CDi;
        float result = acc_local[acc_index][i];
        if (global_x < N && global_y < M) {
          if (b) result += __bfloat162float(b[global_x]);
          float *dst;
          int outIdx;
          if (global_x < q_len) {
            dst = q_out + (size_t)global_y * q_len;
            outIdx = global_x;
          } else if (global_x < q_len + k_len) {
            dst = k_out + (size_t)global_y * k_len;
            outIdx = global_x - q_len;
          } else {
            dst = v_out + (size_t)global_y * v_len;
            outIdx = global_x - q_len - k_len;
          }
          dst[outIdx] = result;
        }
      }
    }
  }
}

static inline void getp_matmul_qkv_fused_bf16(
    float *q, float *k, float *v, float *x, __hip_bfloat16 *w_qkv_bf16,
    __hip_bfloat16 *b_qkv_bf16, int n, int head_dim, int n_attn_heads,
    int n_kv_heads, int batch_size, hipStream_t stream) {
  PROFILE_FUNCTION();
  const int q_len = head_dim * n_attn_heads;
  const int k_len = head_dim * n_kv_heads;
  const int v_len = head_dim * n_kv_heads;

  const int M = batch_size;
  const int N = q_len + k_len + v_len;
  const int K = n;

  const int BN = 128;
  const int BM = 128;
  const int BK = 16;

  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  new_matmul_qkv_fused_kernel<<<grid, block, 0, stream>>>(
      q, k, v, x, w_qkv_bf16, b_qkv_bf16, n, q_len, k_len, v_len, batch_size);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void __launch_bounds__(256)
    matmul_kernel_nosplit(float *__restrict__ xout, const float *__restrict__ x,
                          const __hip_bfloat16 *__restrict__ w,
                          const __hip_bfloat16 *__restrict__ b, int M, int N,
                          int K) {
  constexpr int BM = 32, BN = 32, BK = 32, WM = 16, WN = 16;
  const int BNt = BN * GETP_BN_AGG;
  const int wave = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int xTile = wave % (BM / WM);
  const int yTile = wave / (BM / WM);
  const int xMF = lane & 15;
  const int yMF = lane >> 4;
  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  constexpr int strideX = 256 / (BK / 4);
  constexpr int strideW = 256 / (BK / 8);

  extern __shared__ unsigned short sm[];
  unsigned short *sx0 = sm;
  unsigned short *sw0 = sx0 + (size_t)BK * BM;
  unsigned short *sx1 = sw0 + (size_t)BK * BNt;
  unsigned short *sw1 = sx1 + (size_t)BK * BM;

  f32x4 d0 = {0, 0, 0, 0}, d1 = {0, 0, 0, 0}, d2 = {0, 0, 0, 0},
        d3 = {0, 0, 0, 0};

  const int rowBase = blockIdx.y * BM;
  const int colBase = blockIdx.x * BNt;

  auto prefetch = [&](int bk, unsigned short *sx, unsigned short *sw) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk = bk + 4 * loadX_x;
      const int row = rowBase + loadX_y + off;
      if (row < M) {
        if (kk + 3 < K) {
          float4 v =
              *reinterpret_cast<const float4 *>(x + (size_t)row * K + kk);
          sx[(4 * loadX_x + 0) * BM + (loadX_y + off)] = f32_to_bf16bits(v.x);
          sx[(4 * loadX_x + 1) * BM + (loadX_y + off)] = f32_to_bf16bits(v.y);
          sx[(4 * loadX_x + 2) * BM + (loadX_y + off)] = f32_to_bf16bits(v.z);
          sx[(4 * loadX_x + 3) * BM + (loadX_y + off)] = f32_to_bf16bits(v.w);
        } else {
          for (int t = 0; t < 4; ++t) {
            int kx = kk + t;
            unsigned short val = 0;
            if (kx < K) val = f32_to_bf16bits(x[(size_t)row * K + kx]);
            sx[(4 * loadX_x + t) * BM + (loadX_y + off)] = val;
          }
        }
      }
    }
    for (int off = 0; off < BNt; off += strideW) {
      if (loadW_y + off >= BNt) break;
      const int kk = bk + 8 * loadW_x;
      const int col = colBase + loadW_y + off;
      unsigned short *dst = sw + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
      if (col < N) {
        if (kk + 7 < K) {
          uint4 t = *reinterpret_cast<const uint4 *>(w + (size_t)col * K + kk);
          dst[0 * BNt] = (unsigned short)(t.x & 0xFFFF);
          dst[1 * BNt] = (unsigned short)(t.x >> 16);
          dst[2 * BNt] = (unsigned short)(t.y & 0xFFFF);
          dst[3 * BNt] = (unsigned short)(t.y >> 16);
          dst[4 * BNt] = (unsigned short)(t.z & 0xFFFF);
          dst[5 * BNt] = (unsigned short)(t.z >> 16);
          dst[6 * BNt] = (unsigned short)(t.w & 0xFFFF);
          dst[7 * BNt] = (unsigned short)(t.w >> 16);
        } else {
          for (int t = 0; t < 8; ++t) {
            int kx = kk + t;
            unsigned short val = 0;
            if (kx < K)
              val = reinterpret_cast<const unsigned short *>(
                  w + (size_t)col * K + kx)[0];
            dst[t * BNt] = val;
          }
        }
      } else {
        for (int t = 0; t < 8; ++t) dst[t * BNt] = 0;
      }
    }
  };

  int bk = 0;
  if (K > 0) prefetch(0, sx0, sw0);
  __syncthreads();
  bool ping = true;

  for (; bk < K; bk += BK) {
    unsigned short *sx = ping ? sx0 : sx1;
    unsigned short *sw = ping ? sw0 : sw1;
    int nxt = bk + BK;
    if (nxt < K) prefetch(nxt, ping ? sx1 : sx0, ping ? sw1 : sw0);
    for (int k = 0; k < BK; k += 16) {
      const int basek = k + yMF * 4;
      const int row = yTile * WM + xMF;
      const int c = xTile * WN + xMF;
      bf16x4 av = {sx[(basek + 0) * BM + row], sx[(basek + 1) * BM + row],
                   sx[(basek + 2) * BM + row], sx[(basek + 3) * BM + row]};
      bf16x4 b0 = {sw[(basek + 0) * BNt + c + 0 * BN],
                   sw[(basek + 1) * BNt + c + 0 * BN],
                   sw[(basek + 2) * BNt + c + 0 * BN],
                   sw[(basek + 3) * BNt + c + 0 * BN]};
      bf16x4 b1 = {sw[(basek + 0) * BNt + c + 1 * BN],
                   sw[(basek + 1) * BNt + c + 1 * BN],
                   sw[(basek + 2) * BNt + c + 1 * BN],
                   sw[(basek + 3) * BNt + c + 1 * BN]};
      bf16x4 b2 = {sw[(basek + 0) * BNt + c + 2 * BN],
                   sw[(basek + 1) * BNt + c + 2 * BN],
                   sw[(basek + 2) * BNt + c + 2 * BN],
                   sw[(basek + 3) * BNt + c + 2 * BN]};
      bf16x4 b3 = {sw[(basek + 0) * BNt + c + 3 * BN],
                   sw[(basek + 1) * BNt + c + 3 * BN],
                   sw[(basek + 2) * BNt + c + 3 * BN],
                   sw[(basek + 3) * BNt + c + 3 * BN]};
      d0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b0, d0, 0, 0, 0);
      d1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b1, d1, 0, 0, 0);
      d2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b2, d2, 0, 0, 0);
      d3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b3, d3, 0, 0, 0);
    }
    __syncthreads();
    ping = !ping;
  }

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  for (int i = 0; i < 4; ++i) {
    const int row = rowBase + yTile * WM + (yD + i);
    if (row >= M) continue;
    int c0 = colBase + xTile * WN + xD + 0 * BN;
    int c1 = colBase + xTile * WN + xD + 1 * BN;
    int c2 = colBase + xTile * WN + xD + 2 * BN;
    int c3 = colBase + xTile * WN + xD + 3 * BN;
    if (c0 < N) {
      float v = d0[i];
      if (b) v += __bfloat162float(b[c0]);
      xout[(size_t)row * N + c0] = v;
    }
    if (c1 < N) {
      float v = d1[i];
      if (b) v += __bfloat162float(b[c1]);
      xout[(size_t)row * N + c1] = v;
    }
    if (c2 < N) {
      float v = d2[i];
      if (b) v += __bfloat162float(b[c2]);
      xout[(size_t)row * N + c2] = v;
    }
    if (c3 < N) {
      float v = d3[i];
      if (b) v += __bfloat162float(b[c3]);
      xout[(size_t)row * N + c3] = v;
    }
  }
}

__global__ void __launch_bounds__(256)
    matmul_kernel_splitk_store(float *__restrict__ partial,
                               const float *__restrict__ x,
                               const __hip_bfloat16 *__restrict__ w, int M,
                               int N, int K, int splits) {
  constexpr int BM = 32, BN = 32, BK = 32, WM = 16, WN = 16;
  const int BNt = BN * GETP_BN_AGG;

  const int wave = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int xTile = wave % (BM / WM);
  const int yTile = wave / (BM / WM);
  const int xMF = lane & 15;
  const int yMF = lane >> 4;

  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  constexpr int strideX = 256 / (BK / 4);
  constexpr int strideW = 256 / (BK / 8);

  extern __shared__ unsigned short sm[];
  unsigned short *sx0 = sm;
  unsigned short *sw0 = sx0 + (size_t)BK * BM;
  unsigned short *sx1 = sw0 + (size_t)BK * BNt;
  unsigned short *sw1 = sx1 + (size_t)BK * BM;

  f32x4 d0 = {0, 0, 0, 0}, d1 = {0, 0, 0, 0}, d2 = {0, 0, 0, 0},
        d3 = {0, 0, 0, 0};

  const int rowBase = blockIdx.y * BM;
  const int colBase = blockIdx.x * BNt;

  const int split = blockIdx.z;
  const int kChunk = (K + splits - 1) / splits;
  const int kBeg = split * kChunk;
  const int kEnd = min(K, kBeg + kChunk);

  auto prefetch = [&](int bk, unsigned short *sx, unsigned short *sw) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk = bk + 4 * loadX_x;
      const int row = rowBase + loadX_y + off;
      if (row < M) {
        if (kk + 3 < kEnd) {
          float4 v =
              *reinterpret_cast<const float4 *>(x + (size_t)row * K + kk);
          sx[(4 * loadX_x + 0) * BM + (loadX_y + off)] = f32_to_bf16bits(v.x);
          sx[(4 * loadX_x + 1) * BM + (loadX_y + off)] = f32_to_bf16bits(v.y);
          sx[(4 * loadX_x + 2) * BM + (loadX_y + off)] = f32_to_bf16bits(v.z);
          sx[(4 * loadX_x + 3) * BM + (loadX_y + off)] = f32_to_bf16bits(v.w);
        } else {
          const int kk0 = kk;
          unsigned short v0 = 0, v1 = 0, v2 = 0, v3 = 0;
          if (kk0 + 0 < kEnd)
            v0 = f32_to_bf16bits(x[(size_t)row * K + kk0 + 0]);
          if (kk0 + 1 < kEnd)
            v1 = f32_to_bf16bits(x[(size_t)row * K + kk0 + 1]);
          if (kk0 + 2 < kEnd)
            v2 = f32_to_bf16bits(x[(size_t)row * K + kk0 + 2]);
          if (kk0 + 3 < kEnd)
            v3 = f32_to_bf16bits(x[(size_t)row * K + kk0 + 3]);
          sx[(4 * loadX_x + 0) * BM + (loadX_y + off)] = v0;
          sx[(4 * loadX_x + 1) * BM + (loadX_y + off)] = v1;
          sx[(4 * loadX_x + 2) * BM + (loadX_y + off)] = v2;
          sx[(4 * loadX_x + 3) * BM + (loadX_y + off)] = v3;
        }
      }
    }
    for (int off = 0; off < BNt; off += strideW) {
      if (loadW_y + off >= BNt) break;
      const int kk = bk + 8 * loadW_x;
      const int col = colBase + loadW_y + off;
      unsigned short *dst = sw + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
      if (col < N && kk < kEnd) {
        if (kk + 7 < kEnd) {
          uint4 t = *reinterpret_cast<const uint4 *>(w + (size_t)col * K + kk);
          dst[0 * BNt] = (unsigned short)(t.x & 0xFFFF);
          dst[1 * BNt] = (unsigned short)(t.x >> 16);
          dst[2 * BNt] = (unsigned short)(t.y & 0xFFFF);
          dst[3 * BNt] = (unsigned short)(t.y >> 16);
          dst[4 * BNt] = (unsigned short)(t.z & 0xFFFF);
          dst[5 * BNt] = (unsigned short)(t.z >> 16);
          dst[6 * BNt] = (unsigned short)(t.w & 0xFFFF);
          dst[7 * BNt] = (unsigned short)(t.w >> 16);
        } else {
          for (int t = 0; t < 8; ++t) {
            int kx = kk + t;
            unsigned short val = 0;
            if (kx < kEnd)
              val = reinterpret_cast<const unsigned short *>(
                  w + (size_t)col * K + kx)[0];
            dst[t * BNt] = val;
          }
        }
      } else {
        for (int t = 0; t < 8; ++t) dst[t * BNt] = 0;
      }
    }
  };

  int bk = kBeg;
  if (bk < kEnd) prefetch(bk, sx0, sw0);
  __syncthreads();
  bool ping = true;

  for (; bk < kEnd; bk += BK) {
    unsigned short *sx = ping ? sx0 : sx1;
    unsigned short *sw = ping ? sw0 : sw1;
    int nxt = bk + BK;
    if (nxt < kEnd) prefetch(nxt, ping ? sx1 : sx0, ping ? sw1 : sw0);
    for (int k = 0; k < BK; k += 16) {
      const int basek = k + yMF * 4;
      const int row = yTile * WM + xMF;
      const int c = xTile * WN + xMF;
      bf16x4 av = {sx[(basek + 0) * BM + row], sx[(basek + 1) * BM + row],
                   sx[(basek + 2) * BM + row], sx[(basek + 3) * BM + row]};
      bf16x4 b0 = {sw[(basek + 0) * BNt + c + 0 * BN],
                   sw[(basek + 1) * BNt + c + 0 * BN],
                   sw[(basek + 2) * BNt + c + 0 * BN],
                   sw[(basek + 3) * BNt + c + 0 * BN]};
      bf16x4 b1 = {sw[(basek + 0) * BNt + c + 1 * BN],
                   sw[(basek + 1) * BNt + c + 1 * BN],
                   sw[(basek + 2) * BNt + c + 1 * BN],
                   sw[(basek + 3) * BNt + c + 1 * BN]};
      bf16x4 b2 = {sw[(basek + 0) * BNt + c + 2 * BN],
                   sw[(basek + 1) * BNt + c + 2 * BN],
                   sw[(basek + 2) * BNt + c + 2 * BN],
                   sw[(basek + 3) * BNt + c + 2 * BN]};
      bf16x4 b3 = {sw[(basek + 0) * BNt + c + 3 * BN],
                   sw[(basek + 1) * BNt + c + 3 * BN],
                   sw[(basek + 2) * BNt + c + 3 * BN],
                   sw[(basek + 3) * BNt + c + 3 * BN]};
      d0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b0, d0, 0, 0, 0);
      d1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b1, d1, 0, 0, 0);
      d2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b2, d2, 0, 0, 0);
      d3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, b3, d3, 0, 0, 0);
    }
    __syncthreads();
    ping = !ping;
  }

  const size_t slice_stride = (size_t)M * N;
  const size_t base_out = (size_t)split * slice_stride;

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  for (int i = 0; i < 4; ++i) {
    const int row = rowBase + yTile * WM + (yD + i);
    if (row >= M) continue;
    int c0 = colBase + xTile * WN + xD + 0 * BN;
    int c1 = colBase + xTile * WN + xD + 1 * BN;
    int c2 = colBase + xTile * WN + xD + 2 * BN;
    int c3 = colBase + xTile * WN + xD + 3 * BN;
    if (c0 < N) partial[base_out + (size_t)row * N + c0] = d0[i];
    if (c1 < N) partial[base_out + (size_t)row * N + c1] = d1[i];
    if (c2 < N) partial[base_out + (size_t)row * N + c2] = d2[i];
    if (c3 < N) partial[base_out + (size_t)row * N + c3] = d3[i];
  }
}

__global__ void reduce_splitk_with_bias(float *__restrict__ out,
                                        const float *__restrict__ partial,
                                        const __hip_bfloat16 *__restrict__ b,
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
void getp_matmul(float *xout, float *x, T *w, T *b, int n, int d,
                 int batch_size, hipStream_t stream) {
  PROFILE_FUNCTION();
  const int M = batch_size, N = d, K = n;
  constexpr int BM = 32, BN = 32, BK = 32;
  const int BNt = BN * GETP_BN_AGG;

  dim3 block(256);
  const int gx = (N + BNt - 1) / BNt;
  const int gy = (M + BM - 1) / BM;
  const int grid_xy = max(1, gx) * max(1, gy);

  int dev = 0, cu = 104;
  HIP_CHECK(hipGetDevice(&dev));
  HIP_CHECK(
      hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev));
  if (cu <= 0) cu = 104;

  const int target_cta = cu * 4;
  int splits =
      (grid_xy >= target_cta) ? 1 : (target_cta + grid_xy - 1) / grid_xy;
  splits = min(8, max(1, splits));
  const int max_splits_by_K = max(1, (K + BK - 1) / BK);
  splits = min(splits, max_splits_by_K);

  const size_t shmem = (size_t)2 * (size_t)BK * ((size_t)BM + (size_t)BNt) *
                       sizeof(unsigned short);

  if (splits == 1) {
    dim3 grid(gx, gy);
    matmul_kernel_nosplit<<<grid, block, shmem, stream>>>(
        xout, x, reinterpret_cast<const __hip_bfloat16 *>(w),
        reinterpret_cast<const __hip_bfloat16 *>(b), M, N, K);
    //HIP_CHECK(hipDeviceSynchronize());
    return;
  }

  float *partial = nullptr;
  HIP_CHECK(hipMallocAsync(
      &partial, (size_t)splits * (size_t)M * (size_t)N * sizeof(float),
      stream));

  dim3 grid(gx, gy, splits);
  matmul_kernel_splitk_store<<<grid, block, shmem, stream>>>(
      partial, x, reinterpret_cast<const __hip_bfloat16 *>(w), M, N, K, splits);

  const int total = M * N;
  dim3 rBlock(256);
  dim3 rGrid((total + rBlock.x - 1) / rBlock.x);
  reduce_splitk_with_bias<<<rGrid, rBlock, 0, stream>>>(
      xout, partial, reinterpret_cast<const __hip_bfloat16 *>(b), M, N, splits);

  HIP_CHECK(hipFreeAsync(partial, stream));
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
                          float *sin_out, float *inv_freq, hipStream_t stream) {
  PROFILE_FUNCTION();
  int d_half = head_dim / 2;
  float concentration =
      scaling_factor > 1.0f ? 0.1f * logf(scaling_factor) + 1.0f : 1.0f;
  {
    dim3 blockDim(256);
    dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x);
    compute_inv_freq_kernel<<<gridDim, blockDim, 0, stream>>>(
        base, head_dim, scaling_factor, initial_context_length, ntk_beta,
        ntk_alpha, inv_freq);
  }
  {
    dim3 blockDim(256);
    dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x);
    compute_cos_sin_kernel<<<gridDim, blockDim, 0, stream>>>(
        cos_out, sin_out, inv_freq, concentration, pos, d_half);
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
                           int head_dim, int batch_size, hipStream_t stream) {
  PROFILE_FUNCTION();
  dim3 blockDim(16, 16);
  dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x,
               (n_heads + blockDim.y - 1) / blockDim.y, batch_size);
  apply_rotary_emb_kernel<<<gridDim, blockDim, 0, stream>>>(x, cos, sin,
                                                            n_heads, head_dim);
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
    float *topk_v_out, int *topk_i_out, int batch_size, hipStream_t stream) {
  PROFILE_FUNCTION();
  const int BLK = 1024;
  const dim3 block(BLK);
  const dim3 grid(batch_size);
  const size_t shmem = (size_t)n_experts * sizeof(float) +
                       (size_t)BLK * (sizeof(float) + sizeof(int)) +
                       GETP_ROUTER_TOPK_MAXK * (sizeof(float) + sizeof(int));
  router_topk_softmax_batch_kernel<<<grid, block, shmem, stream>>>(
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
    int *n_local, int K, int expert_start, int expert_end, int B,
    hipStream_t stream) {
  PROFILE_FUNCTION();
  const int BLK = 128, GRD = (B + BLK - 1) / BLK;
  map_global_to_local_batch_kernel<<<GRD, BLK, 0, stream>>>(
      topk_i, topk_v, local_ids, local_wts, n_local, K, expert_start,
      expert_end, B);
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
void getp_vecadd(float *x, float *y, int size, int batch_size,
                 hipStream_t stream) {
  PROFILE_FUNCTION();
  dim3 blockDim(1024);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x, batch_size);
  vecadd_kernel<<<gridDim, blockDim, 0, stream>>>(x, y, size);
  //HIP_CHECK(hipDeviceSynchronize());
}

template <int TILE_T>
__global__ void __launch_bounds__(128)
flash_attn_decode_even_layer_bf16_matrix_core_kernel(
  float *__restrict__ tb, const __hip_bfloat16 *__restrict__ key_cache,
  const __hip_bfloat16 *__restrict__ value_cache, const float *__restrict__ q,
  const float *__restrict__ attn_sinks, int head_dim, int n_attn_heads,
  int n_kv_heads, int pos, int batch_size, int kv_dim, int kv_mul, 
  int sliding_window, int cache_tcap, float inv_sqrt_d
) {
  /* Ignore sliding window mechanism, only consider odd layer */
  constexpr int HEAD_DIM = 64;
  constexpr int BLOCK_SIZE = 128;

  constexpr int MFMA_M = 4;
  constexpr int MFMA_N = 4;
  constexpr int MFMA_K = 1;
  constexpr int MFMA_BLOCK = 16;
  constexpr int GPRs_CD = 4;

  using CDfloat = __attribute__( (__vector_size__(GPRs_CD * sizeof(float)) )) float;

  assert(head_dim == 64);
  assert(blockDim.x == BLOCK_SIZE);

  // Wave tile
  constexpr int WM = 4;
  constexpr int WN = TILE_T;

  constexpr int nIterWaveM = WM / MFMA_M;
  constexpr int nIterWaveN = WN / MFMA_N;

  const int waveIdx = threadIdx.x / warpSize;
  const int laneIdx = threadIdx.x % warpSize;

  const int yInBlockTile = waveIdx;

  const int Ai = (laneIdx % 4);
  const int Ak = 0;
  const int Ablock = (laneIdx / 4);
  const int Bj = (laneIdx % 4);
  const int Bk = 0;
  const int Bblock = (laneIdx / 4);

  const int nbLoadsKV = TILE_T * HEAD_DIM / BLOCK_SIZE;

  const int kv_h = blockIdx.x;
  const int b = blockIdx.y;

  const int loadKVIdx_y = threadIdx.x / (HEAD_DIM);
  const int loadKVIdx_x = threadIdx.x % (HEAD_DIM);
  constexpr int strideKV = BLOCK_SIZE / (HEAD_DIM);

  const int t_start = MAX(0, pos - sliding_window + 1);
  const int n_steps = pos + 1 - t_start;

  __shared__ __hip_bfloat16 ks[TILE_T][HEAD_DIM];
  __shared__ __hip_bfloat16 vs[TILE_T][HEAD_DIM];

  float out_values[MFMA_M] = {0.f};
  float m_values[MFMA_M] = {-INFINITY};
  float l_values[MFMA_M] = {0.f};

  const float *qptr =
      q + (size_t)b * n_attn_heads * head_dim;
  
  float q_values[HEAD_DIM / MFMA_BLOCK];
  for (int i = 0; i < HEAD_DIM / MFMA_BLOCK; ++i) {
    q_values[i] = qptr[(Ai + MFMA_M * waveIdx + kv_h * kv_mul) * head_dim + i * MFMA_BLOCK + Ablock];
  }

  // Load data to shared memory
  for (int i = 0; i < nbLoadsKV; ++i) {
    int offset = i * strideKV;
    int index_x = loadKVIdx_x;
    int index_y = t_start + loadKVIdx_y + offset;
    if (index_y < t_start + n_steps) {
      const __hip_bfloat16 *kptr =
          key_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
          (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
      const __hip_bfloat16 *vptr =
          value_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
          (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
      ks[loadKVIdx_y + offset][loadKVIdx_x] = kptr[0];
      vs[loadKVIdx_y + offset][loadKVIdx_x] = vptr[0];
    }
    else {
      ks[loadKVIdx_y + offset][loadKVIdx_x] = 0;
      vs[loadKVIdx_y + offset][loadKVIdx_x] = 0;
    }
  }
  __syncthreads();

  for (int btIdx = t_start; btIdx < t_start + n_steps; btIdx += TILE_T) {
    // Prefetch
    __hip_bfloat16 regK[nbLoadsKV];
    __hip_bfloat16 regV[nbLoadsKV];
    if (btIdx + TILE_T < t_start + n_steps) {
      for (int i = 0; i < nbLoadsKV; ++i) {
        int offset = i * strideKV;
        int index_x = loadKVIdx_x;
        int index_y = btIdx + TILE_T + loadKVIdx_y + offset;
        if (index_y < t_start + n_steps) {
          const __hip_bfloat16 *kptr =
              key_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
              (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
          const __hip_bfloat16 *vptr =
              value_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
              (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
          regK[i] = kptr[0];
          regV[i] = vptr[0];
        }
        else {
          regK[i] = regV[i] = 0;
        }
      }
    }

    // Computing
    for (int t = 0; t < TILE_T; t += MFMA_N) {
      CDfloat acc = {0};
      for (int dimIdx = 0; dimIdx < HEAD_DIM; dimIdx += MFMA_BLOCK) {
        float k_value = __bfloat162float(ks[t + Bj][dimIdx + Ablock]);
        acc = __builtin_amdgcn_mfma_f32_4x4x1f32(q_values[dimIdx / MFMA_BLOCK], k_value, acc, 0, 0, 0);
      }

      for (int width = warpSize / 2; width >= MFMA_N; width /= 2) {
        for (int i = 0; i < GPRs_CD; ++i) {
          acc[i] += __shfl_down(acc[i], width);
        }
      }

      for (int i = 0; i < MFMA_M; ++i) {
        for (int j = 0; j < MFMA_N; ++j) {
          if (btIdx + t + j >= t_start + n_steps) break;
          float attn_score = __shfl(acc[i], j);
          attn_score *= inv_sqrt_d;
          float m_new = fmaxf(m_values[i], attn_score);
          float alpha = __expf(m_values[i] - m_new);
          float e = __expf(attn_score - m_new);
          l_values[i] = l_values[i] * alpha + e;
          m_values[i] = m_new;

          out_values[i] = alpha * out_values[i] + e * __bfloat162float(vs[t + j][laneIdx]);
        }
      }
    }

    __syncthreads();

    if (btIdx + TILE_T < t_start + n_steps) {
      for (int i = 0; i < nbLoadsKV; ++i) {
        int offset = i * strideKV;
        int index_x = loadKVIdx_x;
        int index_y = btIdx + TILE_T + loadKVIdx_y + offset;
        ks[loadKVIdx_y + offset][loadKVIdx_x] = regK[i];
        vs[loadKVIdx_y + offset][loadKVIdx_x] = regV[i];
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < MFMA_M; ++i) {
    int h = kv_h * kv_mul + waveIdx * MFMA_M + i;

    float attn_score = attn_sinks[h];
    float m_new = fmaxf(m_values[i], attn_score);
    float alpha = __expf(m_values[i] - m_new);
    float e = __expf(attn_score - m_new);
    l_values[i] = l_values[i] * alpha + e;
    m_values[i] = m_new;
    out_values[i] = alpha * out_values[i] / l_values[i];

    float *obh =
        tb + (size_t)b * n_attn_heads * head_dim + 
             (size_t)h * head_dim;
    obh[laneIdx] = out_values[i];
  }

}

template <int TILE_T>
__global__ void __launch_bounds__(128)
flash_attn_decode_odd_layer_bf16_matrix_core_kernel(
  float *__restrict__ tb, const __hip_bfloat16 *__restrict__ key_cache,
  const __hip_bfloat16 *__restrict__ value_cache, const float *__restrict__ q,
  const float *__restrict__ attn_sinks, int head_dim, int n_attn_heads,
  int n_kv_heads, int pos, int batch_size, int kv_dim, int kv_mul, 
  float inv_sqrt_d
) {
  /* Ignore sliding window mechanism, only consider odd layer */
  constexpr int HEAD_DIM = 64;
  constexpr int BLOCK_SIZE = 128;

  constexpr int MFMA_M = 4;
  constexpr int MFMA_N = 4;
  constexpr int MFMA_K = 1;
  constexpr int MFMA_BLOCK = 16;
  constexpr int GPRs_CD = 4;

  using CDfloat = __attribute__( (__vector_size__(GPRs_CD * sizeof(float)) )) float;

  assert(head_dim == 64);
  assert(blockDim.x == BLOCK_SIZE);

  // Wave tile
  constexpr int WM = 4;
  constexpr int WN = TILE_T;

  constexpr int nIterWaveM = WM / MFMA_M;
  constexpr int nIterWaveN = WN / MFMA_N;

  const int waveIdx = threadIdx.x / warpSize;
  const int laneIdx = threadIdx.x % warpSize;

  const int yInBlockTile = waveIdx;

  const int Ai = (laneIdx % 4);
  const int Ak = 0;
  const int Ablock = (laneIdx / 4);
  const int Bj = (laneIdx % 4);
  const int Bk = 0;
  const int Bblock = (laneIdx / 4);

  const int nbLoadsKV = TILE_T * HEAD_DIM / BLOCK_SIZE;

  const int kv_h = blockIdx.x;
  const int b = blockIdx.y;

  const int loadKVIdx_y = threadIdx.x / (HEAD_DIM);
  const int loadKVIdx_x = threadIdx.x % (HEAD_DIM);
  constexpr int strideKV = BLOCK_SIZE / (HEAD_DIM);

  const int n_steps = pos + 1;

  __shared__ __hip_bfloat16 ks[TILE_T][HEAD_DIM];
  __shared__ __hip_bfloat16 vs[TILE_T][HEAD_DIM];

  float out_values[MFMA_M] = {0.f};
  float m_values[MFMA_M] = {-INFINITY};
  float l_values[MFMA_M] = {0.f};

  const float *qptr =
      q + (size_t)b * n_attn_heads * head_dim;
  
  float q_values[HEAD_DIM / MFMA_BLOCK];
  for (int i = 0; i < HEAD_DIM / MFMA_BLOCK; ++i) {
    q_values[i] = qptr[(Ai + MFMA_M * waveIdx + kv_h * kv_mul) * head_dim + i * MFMA_BLOCK + Ablock];
  }

  // Load data to shared memory
  for (int i = 0; i < nbLoadsKV; ++i) {
    int offset = i * strideKV;
    int index_x = loadKVIdx_x;
    int index_y = 0 + loadKVIdx_y + offset;
    if (index_y < n_steps) {
      const __hip_bfloat16 *kptr =
          key_cache + index_y * (size_t)batch_size * (size_t)kv_dim +
          (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
      const __hip_bfloat16 *vptr =
          value_cache + index_y * (size_t)batch_size * (size_t)kv_dim +
          (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
      ks[loadKVIdx_y + offset][loadKVIdx_x] = kptr[0];
      vs[loadKVIdx_y + offset][loadKVIdx_x] = vptr[0];
    }
    else {
      ks[loadKVIdx_y + offset][loadKVIdx_x] = 0;
      vs[loadKVIdx_y + offset][loadKVIdx_x] = 0;
    }
  }
  __syncthreads();

  for (int btIdx = 0; btIdx < n_steps; btIdx += TILE_T) {
    // Prefetch
    __hip_bfloat16 regK[nbLoadsKV];
    __hip_bfloat16 regV[nbLoadsKV];
    if (btIdx + TILE_T < n_steps) {
      for (int i = 0; i < nbLoadsKV; ++i) {
        int offset = i * strideKV;
        int index_x = loadKVIdx_x;
        int index_y = btIdx + TILE_T + loadKVIdx_y + offset;
        if (index_y < n_steps) {
          const __hip_bfloat16 *kptr =
              key_cache + index_y * (size_t)batch_size * (size_t)kv_dim +
              (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
          const __hip_bfloat16 *vptr =
              value_cache + index_y * (size_t)batch_size * (size_t)kv_dim +
              (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
          regK[i] = kptr[0];
          regV[i] = vptr[0];
        }
        else {
          regK[i] = regV[i] = 0;
        }
      }
    }

    // Computing
    for (int t = 0; t < TILE_T; t += MFMA_N) {
      CDfloat acc = {0};
      for (int dimIdx = 0; dimIdx < HEAD_DIM; dimIdx += MFMA_BLOCK) {
        float k_value = __bfloat162float(ks[t + Bj][dimIdx + Ablock]);
        acc = __builtin_amdgcn_mfma_f32_4x4x1f32(q_values[dimIdx / MFMA_BLOCK], k_value, acc, 0, 0, 0);
      }

      for (int width = warpSize / 2; width >= MFMA_N; width /= 2) {
        for (int i = 0; i < GPRs_CD; ++i) {
          acc[i] += __shfl_down(acc[i], width);
        }
      }

      for (int i = 0; i < MFMA_M; ++i) {
        for (int j = 0; j < MFMA_N; ++j) {
          if (btIdx + t + j >= n_steps) break;
          float attn_score = __shfl(acc[i], j);
          attn_score *= inv_sqrt_d;
          float m_new = fmaxf(m_values[i], attn_score);
          float alpha = __expf(m_values[i] - m_new);
          float e = __expf(attn_score - m_new);
          l_values[i] = l_values[i] * alpha + e;
          m_values[i] = m_new;

          out_values[i] = alpha * out_values[i] + e * __bfloat162float(vs[t + j][laneIdx]);
        }
      }
    }

    __syncthreads();

    if (btIdx + TILE_T < n_steps) {
      for (int i = 0; i < nbLoadsKV; ++i) {
        int offset = i * strideKV;
        int index_x = loadKVIdx_x;
        int index_y = btIdx + TILE_T + loadKVIdx_y + offset;
        ks[loadKVIdx_y + offset][loadKVIdx_x] = regK[i];
        vs[loadKVIdx_y + offset][loadKVIdx_x] = regV[i];
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < MFMA_M; ++i) {
    int h = kv_h * kv_mul + waveIdx * MFMA_M + i;

    float attn_score = attn_sinks[h];
    float m_new = fmaxf(m_values[i], attn_score);
    float alpha = __expf(m_values[i] - m_new);
    float e = __expf(attn_score - m_new);
    l_values[i] = l_values[i] * alpha + e;
    m_values[i] = m_new;
    out_values[i] = alpha * out_values[i] / l_values[i];

    float *obh =
        tb + (size_t)b * n_attn_heads * head_dim + 
             (size_t)h * head_dim;
    obh[laneIdx] = out_values[i];
  }

}
static inline void getp_flash_attn_decode_bf16(
    float *tb, const __hip_bfloat16 *key_cache_layer,
    const __hip_bfloat16 *value_cache_layer, const float *q,
    const float *attn_sinks_layer, int head_dim, int n_attn_heads,
    int n_kv_heads, int pos, int seq_len, int sliding_window, int layer_id,
    int batch_size, int cache_tcap, hipStream_t stream) {
  PROFILE_FUNCTION();
  const int kv_dim = head_dim * n_kv_heads;
  const int kv_mul = n_attn_heads / n_kv_heads;
  const float invsd = 1.0f / sqrtf((float)head_dim);
  const int apply_mask = (sliding_window > 0 && ((layer_id & 1) == 0)) ? 1 : 0;

  if (apply_mask) {
    dim3 block(128);
    dim3 grid(n_kv_heads, batch_size);
    flash_attn_decode_even_layer_bf16_matrix_core_kernel<16><<<grid, block>>>
      (tb, key_cache_layer, 
       value_cache_layer, q, 
       attn_sinks_layer, head_dim, n_attn_heads, 
       n_kv_heads, pos, batch_size, kv_dim, kv_mul, 
       sliding_window, cache_tcap, invsd);
  }
  else {
    dim3 block(128);
    dim3 grid(n_kv_heads, batch_size);
    flash_attn_decode_odd_layer_bf16_matrix_core_kernel<16><<<grid, block>>>
      (tb, key_cache_layer, 
      value_cache_layer, q, 
      attn_sinks_layer, head_dim, n_attn_heads, 
      n_kv_heads, pos, batch_size, kv_dim, kv_mul, 
      invsd);
    }

  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void gather_embedding_bf16_kernel(float *x,
                                             const __hip_bfloat16 *table,
                                             const int *tok, int H) {
  int b = blockIdx.x;
  int tid = threadIdx.x;
  int id = tok[b];
  const __hip_bfloat16 *src = table + (size_t)id * H;
  float *dst = x + (size_t)b * H;
  for (int j = tid; j < H; j += blockDim.x) dst[j] = __bfloat162float(src[j]);
}

static inline void getp_gather_embedding_bf16(float *x,
                                              const __hip_bfloat16 *table,
                                              const int *tok, int H, int B,
                                              hipStream_t stream) {
  PROFILE_FUNCTION();
  dim3 block(256), grid(B);
  gather_embedding_bf16_kernel<<<grid, block, 0, stream>>>(x, table, tok, H);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void __launch_bounds__(256)
new_matmul_kernel(float *xout, float *x, __hip_bfloat16 *w, __hip_bfloat16 *b, int M, int N, int K) {
  constexpr int BLOCK_SIZE = 256;

  // MFMA config
  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;
  
  constexpr int GPRs_A  = 2;
  constexpr int GPRs_B  = 2;
  constexpr int GPRs_CD = 4;

  using CDfloat = __attribute__( (__vector_size__(GPRs_CD * sizeof(float)) )) float;
  using Afloat  = __attribute__( (__vector_size__(GPRs_A  * sizeof(float)) )) unsigned short;
  using Bfloat  = __attribute__( (__vector_size__(GPRs_B  * sizeof(float)) )) unsigned short;

  // Block tile
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  // Wave tile
  constexpr int WM = 64;
  constexpr int WN = 64;

  constexpr int nIterWaveM = WM / MFMA_M;
  constexpr int nIterWaveN = WN / MFMA_N;

  const int waveIdx = threadIdx.x / warpSize;
  const int xInBlockTile = waveIdx % (BM / WM);
  const int yInBlockTile = waveIdx / (BM / WM);
  const int laneIdx = threadIdx.x % warpSize;

  const int Ai = (laneIdx % 16);
  const int Ak = 4 * (laneIdx / 16);
  const int Bj = (laneIdx % 16);
  const int Bk = 4 * (laneIdx / 16);

  constexpr int nbLoadsX = BM * (BK / 4) / BLOCK_SIZE;
  constexpr int nbLoadsW = BN * (BK / 8) / BLOCK_SIZE;

  const int loadXIdx_y = threadIdx.x / (BK / 4);
  const int loadXIdx_x = threadIdx.x % (BK / 4);
  const int loadWIdx_y = threadIdx.x / (BK / 8);
  const int loadWIdx_x = threadIdx.x % (BK / 8);
  constexpr int strideX = BLOCK_SIZE / (BK / 4);
  constexpr int strideW = BLOCK_SIZE / (BK / 8);

  assert(blockDim.x == BLOCK_SIZE);

  __shared__ unsigned short xs[BK][BM];
  __shared__ unsigned short ws[BK][BN];

  Afloat x_local;
  Bfloat w_local;

  CDfloat acc_local[nIterWaveM * nIterWaveN] = {0};

  // Load x and w from global memory to shared memory
  for (int i = 0; i < nbLoadsX; ++i) {
    int offset = i * strideX;
    int index_x = 0 + 4 * loadXIdx_x;
    int index_y = BM * blockIdx.y + loadXIdx_y + offset;
    float4 tmp = index_x < K && index_y < M ?
      *reinterpret_cast<float4 *>(&x[index_y * K + index_x]) : make_float4(0.f, 0.f, 0.f, 0.f);
    xs[4 * loadXIdx_x + 0][loadXIdx_y + offset] = f32_to_bf16bits(tmp.x);
    xs[4 * loadXIdx_x + 1][loadXIdx_y + offset] = f32_to_bf16bits(tmp.y);
    xs[4 * loadXIdx_x + 2][loadXIdx_y + offset] = f32_to_bf16bits(tmp.z);
    xs[4 * loadXIdx_x + 3][loadXIdx_y + offset] = f32_to_bf16bits(tmp.w);
  }
  for (int i = 0; i < nbLoadsW; ++i) {
    int offset = i * strideW;
    int index_x = 0 + 8 * loadWIdx_x;
    int index_y = BN * blockIdx.x + loadWIdx_y + offset;
    uint4 tmp = index_x < K && index_y < N ?
      *reinterpret_cast<uint4 *>(&w[index_y * K + index_x]) : uint4(0, 0, 0, 0);
    ws[8 * loadWIdx_x + 0][loadWIdx_y + offset] = (unsigned short)(tmp.x & 0xFFFF);
    ws[8 * loadWIdx_x + 1][loadWIdx_y + offset] = (unsigned short)(tmp.x >> 16);
    ws[8 * loadWIdx_x + 2][loadWIdx_y + offset] = (unsigned short)(tmp.y & 0xFFFF);
    ws[8 * loadWIdx_x + 3][loadWIdx_y + offset] = (unsigned short)(tmp.y >> 16);
    ws[8 * loadWIdx_x + 4][loadWIdx_y + offset] = (unsigned short)(tmp.z & 0xFFFF);
    ws[8 * loadWIdx_x + 5][loadWIdx_y + offset] = (unsigned short)(tmp.z >> 16);
    ws[8 * loadWIdx_x + 6][loadWIdx_y + offset] = (unsigned short)(tmp.w & 0xFFFF);
    ws[8 * loadWIdx_x + 7][loadWIdx_y + offset] = (unsigned short)(tmp.w >> 16);
  }
  __syncthreads();
  
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Prefetch
    float4  regX[nbLoadsX];
    uint4   regW[nbLoadsW];
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int offset = i * strideX;
        int index_x = bkIdx + BK + 4 * loadXIdx_x;
        int index_y = BM * blockIdx.y + loadXIdx_y + offset;
        regX[i] = index_x < K && index_y < M ?
          *reinterpret_cast<float4 *>(&x[index_y * K + index_x]) : make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int offset = i * strideW;
        int index_x = bkIdx + BK + 8 * loadWIdx_x;
        int index_y = BN * blockIdx.x + loadWIdx_y + offset;
        regW[i] = index_x < K && index_y < N ?
          *reinterpret_cast<uint4 *>(&w[index_y * K + index_x]) : uint4(0, 0, 0, 0);
      }
    }

    // Computing
    for (int k = 0; k < BK; k += MFMA_K) {
      for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
        for (int iterWaveN = 0; iterWaveN < nIterWaveN; ++iterWaveN) {

          for (int i = 0; i < 4; ++i) {
            x_local[i] = xs[k + Ak + i][yInBlockTile * WM + MFMA_M * iterWaveM + Ai];
            w_local[i] = ws[k + Bk + i][xInBlockTile * WN + MFMA_N * iterWaveN + Bj];
          }

          CDfloat y_local = acc_local[iterWaveM * nIterWaveN + iterWaveN]; 
          y_local  = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(x_local, w_local, y_local, 0, 0, 0);
          acc_local[iterWaveM * nIterWaveN + iterWaveN] = y_local;
        }
      }
    }
    __syncthreads();
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int offset = i * strideX;
        xs[4 * loadXIdx_x + 0][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].x);
        xs[4 * loadXIdx_x + 1][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].y);
        xs[4 * loadXIdx_x + 2][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].z);
        xs[4 * loadXIdx_x + 3][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].w);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int offset = i * strideW;
        ws[8 * loadWIdx_x + 0][loadWIdx_y + offset] = (unsigned short)(regW[i].x & 0xFFFF);
        ws[8 * loadWIdx_x + 1][loadWIdx_y + offset] = (unsigned short)(regW[i].x >> 16);
        ws[8 * loadWIdx_x + 2][loadWIdx_y + offset] = (unsigned short)(regW[i].y & 0xFFFF);
        ws[8 * loadWIdx_x + 3][loadWIdx_y + offset] = (unsigned short)(regW[i].y >> 16);
        ws[8 * loadWIdx_x + 4][loadWIdx_y + offset] = (unsigned short)(regW[i].z & 0xFFFF);
        ws[8 * loadWIdx_x + 5][loadWIdx_y + offset] = (unsigned short)(regW[i].z >> 16);
        ws[8 * loadWIdx_x + 6][loadWIdx_y + offset] = (unsigned short)(regW[i].w & 0xFFFF);
        ws[8 * loadWIdx_x + 7][loadWIdx_y + offset] = (unsigned short)(regW[i].w >> 16);
      }
    }
    __syncthreads();
  }

  for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
    for (int iterWaveN = 0; iterWaveN < nIterWaveN; ++iterWaveN) {
      int acc_index = iterWaveM * nIterWaveN + iterWaveN;
      for (int i = 0; i < GPRs_CD; ++i) {
        int CDi = 4 * (laneIdx / 16) + (i % 4);
        int CDj = (laneIdx % 16);
        int global_x = blockIdx.x * BN + WN * xInBlockTile + iterWaveN * MFMA_N + CDj;
        int global_y = blockIdx.y * BM + WM * yInBlockTile + iterWaveM * MFMA_M + CDi;
        float result = acc_local[acc_index][i];
        if (global_x < N && global_y < M) {
          if (b) result += __bfloat162float(b[global_x]);
          xout[global_y * N + global_x] = result;
        }
      }
    }
  }
}
void getp_new_matmul(
  float *xout, float *x, __hip_bfloat16 *w, __hip_bfloat16 *b, int n, int d,
                 int batch_size, hipStream_t stream
) {
  PROFILE_FUNCTION();
  const int M = batch_size, N = d, K = n;
  const int BN = 128;
  const int BM = 128;
  const int BK = 16;

  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  new_matmul_kernel<<<grid, block, 0, stream>>>
    (xout, x, w, b, M, N, K);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void matmul_attn_o_bf16_kernel(float *xout, const float *x, const __hip_bfloat16 *w, const __hip_bfloat16 *b, int M, int N, int K) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;
  constexpr int GPRs_A = 2;
  constexpr int GPRs_B = 2;
  constexpr int GPRs_CD = 4;
  using CDfloat = __attribute__((__vector_size__(GPRs_CD * sizeof(float)))) float;
  using Afloat  = __attribute__((__vector_size__(GPRs_A  * sizeof(float)))) unsigned short;
  using Bfloat  = __attribute__((__vector_size__(GPRs_B  * sizeof(float)))) unsigned short;
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int WM = 64;
  constexpr int WN = 64;
  constexpr int nIterWaveM = WM / MFMA_M;
  constexpr int nIterWaveN = WN / MFMA_N;
  const int waveIdx = threadIdx.x / warpSize;
  const int xInBlockTile = waveIdx % (BM / WM);
  const int yInBlockTile = waveIdx / (BM / WM);
  const int laneIdx = threadIdx.x % warpSize;
  const int Ai = (laneIdx % 16);
  const int Ak = 4 * (laneIdx / 16);
  const int Bj = (laneIdx % 16);
  const int Bk = 4 * (laneIdx / 16);
  constexpr int nbLoadsX = BM * (BK / 4) / BLOCK_SIZE;
  constexpr int nbLoadsW = BN * (BK / 8) / BLOCK_SIZE;
  const int loadXIdx_y = threadIdx.x / (BK / 4);
  const int loadXIdx_x = threadIdx.x % (BK / 4);
  const int loadWIdx_y = threadIdx.x / (BK / 8);
  const int loadWIdx_x = threadIdx.x % (BK / 8);
  constexpr int strideX = BLOCK_SIZE / (BK / 4);
  constexpr int strideW = BLOCK_SIZE / (BK / 8);
  __shared__ unsigned short xs[BK][BM];
  __shared__ unsigned short ws[BK][BN];
  Afloat x_local;
  Bfloat w_local;
  CDfloat acc_local[nIterWaveM * nIterWaveN] = {0};
  for (int i = 0; i < nbLoadsX; ++i) {
    int offset = i * strideX;
    int index_x = 0 + 4 * loadXIdx_x;
    int index_y = BM * blockIdx.y + loadXIdx_y + offset;
    float4 tmp = index_x < K && index_y < M ? *reinterpret_cast<const float4 *>(&x[(size_t)index_y * K + index_x]) : make_float4(0.f, 0.f, 0.f, 0.f);
    xs[4 * loadXIdx_x + 0][loadXIdx_y + offset] = f32_to_bf16bits(tmp.x);
    xs[4 * loadXIdx_x + 1][loadXIdx_y + offset] = f32_to_bf16bits(tmp.y);
    xs[4 * loadXIdx_x + 2][loadXIdx_y + offset] = f32_to_bf16bits(tmp.z);
    xs[4 * loadXIdx_x + 3][loadXIdx_y + offset] = f32_to_bf16bits(tmp.w);
  }
  for (int i = 0; i < nbLoadsW; ++i) {
    int offset = i * strideW;
    int index_x = 0 + 8 * loadWIdx_x;
    int index_y = BN * blockIdx.x + loadWIdx_y + offset;
    uint4 tmp = index_x < K && index_y < N ? *reinterpret_cast<const uint4 *>(&w[(size_t)index_y * K + index_x]) : make_uint4(0,0,0,0);
    ws[8 * loadWIdx_x + 0][loadWIdx_y + offset] = (unsigned short)(tmp.x & 0xFFFF);
    ws[8 * loadWIdx_x + 1][loadWIdx_y + offset] = (unsigned short)(tmp.x >> 16);
    ws[8 * loadWIdx_x + 2][loadWIdx_y + offset] = (unsigned short)(tmp.y & 0xFFFF);
    ws[8 * loadWIdx_x + 3][loadWIdx_y + offset] = (unsigned short)(tmp.y >> 16);
    ws[8 * loadWIdx_x + 4][loadWIdx_y + offset] = (unsigned short)(tmp.z & 0xFFFF);
    ws[8 * loadWIdx_x + 5][loadWIdx_y + offset] = (unsigned short)(tmp.z >> 16);
    ws[8 * loadWIdx_x + 6][loadWIdx_y + offset] = (unsigned short)(tmp.w & 0xFFFF);
    ws[8 * loadWIdx_x + 7][loadWIdx_y + offset] = (unsigned short)(tmp.w >> 16);
  }
  __syncthreads();
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    float4 regX[nbLoadsX];
    uint4  regW[nbLoadsW];
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int offset = i * strideX;
        int index_x = bkIdx + BK + 4 * loadXIdx_x;
        int index_y = BM * blockIdx.y + loadXIdx_y + offset;
        regX[i] = index_x < K && index_y < M ? *reinterpret_cast<const float4 *>(&x[(size_t)index_y * K + index_x]) : make_float4(0.f,0.f,0.f,0.f);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int offset = i * strideW;
        int index_x = bkIdx + BK + 8 * loadWIdx_x;
        int index_y = BN * blockIdx.x + loadWIdx_y + offset;
        regW[i] = index_x < K && index_y < N ? *reinterpret_cast<const uint4 *>(&w[(size_t)index_y * K + index_x]) : make_uint4(0,0,0,0);
      }
    }
    for (int k = 0; k < BK; k += MFMA_K) {
      for (int iterM = 0; iterM < nIterWaveM; ++iterM) {
        for (int iterN = 0; iterN < nIterWaveN; ++iterN) {
          for (int i = 0; i < 4; ++i) {
            x_local[i] = xs[k + Ak + i][yInBlockTile * WM + MFMA_M * iterM + Ai];
            w_local[i] = ws[k + Bk + i][xInBlockTile * WN + MFMA_N * iterN + Bj];
          }
          CDfloat y_local = acc_local[iterM * nIterWaveN + iterN];
          y_local = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(x_local, w_local, y_local, 0, 0, 0);
          acc_local[iterM * nIterWaveN + iterN] = y_local;
        }
      }
    }
    __syncthreads();
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int offset = i * strideX;
        xs[4 * loadXIdx_x + 0][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].x);
        xs[4 * loadXIdx_x + 1][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].y);
        xs[4 * loadXIdx_x + 2][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].z);
        xs[4 * loadXIdx_x + 3][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].w);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int offset = i * strideW;
        ws[8 * loadWIdx_x + 0][loadWIdx_y + offset] = (unsigned short)(regW[i].x & 0xFFFF);
        ws[8 * loadWIdx_x + 1][loadWIdx_y + offset] = (unsigned short)(regW[i].x >> 16);
        ws[8 * loadWIdx_x + 2][loadWIdx_y + offset] = (unsigned short)(regW[i].y & 0xFFFF);
        ws[8 * loadWIdx_x + 3][loadWIdx_y + offset] = (unsigned short)(regW[i].y >> 16);
        ws[8 * loadWIdx_x + 4][loadWIdx_y + offset] = (unsigned short)(regW[i].z & 0xFFFF);
        ws[8 * loadWIdx_x + 5][loadWIdx_y + offset] = (unsigned short)(regW[i].z >> 16);
        ws[8 * loadWIdx_x + 6][loadWIdx_y + offset] = (unsigned short)(regW[i].w & 0xFFFF);
        ws[8 * loadWIdx_x + 7][loadWIdx_y + offset] = (unsigned short)(regW[i].w >> 16);
      }
    }
    __syncthreads();
  }
  for (int iterM = 0; iterM < nIterWaveM; ++iterM) {
    for (int iterN = 0; iterN < nIterWaveN; ++iterN) {
      int acc_index = iterM * nIterWaveN + iterN;
      for (int i = 0; i < GPRs_CD; ++i) {
        int CDi = 4 * (laneIdx / 16) + (i % 4);
        int CDj = (laneIdx % 16);
        int global_x = blockIdx.x * BN + WN * xInBlockTile + iterN * MFMA_N + CDj;
        int global_y = blockIdx.y * BM + WM * yInBlockTile + iterM * MFMA_M + CDi;
        float result = acc_local[acc_index][i];
        if (global_x < N && global_y < M) {
          if (b) result += __bfloat162float(b[global_x]);
          xout[(size_t)global_y * N + global_x] = result;
        }
      }
    }
  }
}
__global__ void matmul_logits_bf16_kernel(float *xout, const float *x, const __hip_bfloat16 *w, const __hip_bfloat16 *b, int M, int N, int K) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;
  constexpr int GPRs_A = 2;
  constexpr int GPRs_B = 2;
  constexpr int GPRs_CD = 4;
  using CDfloat = __attribute__((__vector_size__(GPRs_CD * sizeof(float)))) float;
  using Afloat  = __attribute__((__vector_size__(GPRs_A  * sizeof(float)))) unsigned short;
  using Bfloat  = __attribute__((__vector_size__(GPRs_B  * sizeof(float)))) unsigned short;
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int WM = 64;
  constexpr int WN = 64;
  constexpr int nIterWaveM = WM / MFMA_M;
  constexpr int nIterWaveN = WN / MFMA_N;
  const int waveIdx = threadIdx.x / warpSize;
  const int xInBlockTile = waveIdx % (BM / WM);
  const int yInBlockTile = waveIdx / (BM / WM);
  const int laneIdx = threadIdx.x % warpSize;
  const int Ai = (laneIdx % 16);
  const int Ak = 4 * (laneIdx / 16);
  const int Bj = (laneIdx % 16);
  const int Bk = 4 * (laneIdx / 16);
  constexpr int nbLoadsX = BM * (BK / 4) / BLOCK_SIZE;
  constexpr int nbLoadsW = BN * (BK / 8) / BLOCK_SIZE;
  const int loadXIdx_y = threadIdx.x / (BK / 4);
  const int loadXIdx_x = threadIdx.x % (BK / 4);
  const int loadWIdx_y = threadIdx.x / (BK / 8);
  const int loadWIdx_x = threadIdx.x % (BK / 8);
  constexpr int strideX = BLOCK_SIZE / (BK / 4);
  constexpr int strideW = BLOCK_SIZE / (BK / 8);
  __shared__ unsigned short xs[BK][BM];
  __shared__ unsigned short ws[BK][BN];
  Afloat x_local;
  Bfloat w_local;
  CDfloat acc_local[nIterWaveM * nIterWaveN] = {0};
  for (int i = 0; i < nbLoadsX; ++i) {
    int offset = i * strideX;
    int index_x = 0 + 4 * loadXIdx_x;
    int index_y = BM * blockIdx.y + loadXIdx_y + offset;
    float4 tmp = index_x < K && index_y < M ? *reinterpret_cast<const float4 *>(&x[(size_t)index_y * K + index_x]) : make_float4(0.f, 0.f, 0.f, 0.f);
    xs[4 * loadXIdx_x + 0][loadXIdx_y + offset] = f32_to_bf16bits(tmp.x);
    xs[4 * loadXIdx_x + 1][loadXIdx_y + offset] = f32_to_bf16bits(tmp.y);
    xs[4 * loadXIdx_x + 2][loadXIdx_y + offset] = f32_to_bf16bits(tmp.z);
    xs[4 * loadXIdx_x + 3][loadXIdx_y + offset] = f32_to_bf16bits(tmp.w);
  }
  for (int i = 0; i < nbLoadsW; ++i) {
    int offset = i * strideW;
    int index_x = 0 + 8 * loadWIdx_x;
    int index_y = BN * blockIdx.x + loadWIdx_y + offset;
    uint4 tmp = index_x < K && index_y < N ? *reinterpret_cast<const uint4 *>(&w[(size_t)index_y * K + index_x]) : make_uint4(0,0,0,0);
    ws[8 * loadWIdx_x + 0][loadWIdx_y + offset] = (unsigned short)(tmp.x & 0xFFFF);
    ws[8 * loadWIdx_x + 1][loadWIdx_y + offset] = (unsigned short)(tmp.x >> 16);
    ws[8 * loadWIdx_x + 2][loadWIdx_y + offset] = (unsigned short)(tmp.y & 0xFFFF);
    ws[8 * loadWIdx_x + 3][loadWIdx_y + offset] = (unsigned short)(tmp.y >> 16);
    ws[8 * loadWIdx_x + 4][loadWIdx_y + offset] = (unsigned short)(tmp.z & 0xFFFF);
    ws[8 * loadWIdx_x + 5][loadWIdx_y + offset] = (unsigned short)(tmp.z >> 16);
    ws[8 * loadWIdx_x + 6][loadWIdx_y + offset] = (unsigned short)(tmp.w & 0xFFFF);
    ws[8 * loadWIdx_x + 7][loadWIdx_y + offset] = (unsigned short)(tmp.w >> 16);
  }
  __syncthreads();
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    float4 regX[nbLoadsX];
    uint4  regW[nbLoadsW];
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int offset = i * strideX;
        int index_x = bkIdx + BK + 4 * loadXIdx_x;
        int index_y = BM * blockIdx.y + loadXIdx_y + offset;
        regX[i] = index_x < K && index_y < M ? *reinterpret_cast<const float4 *>(&x[(size_t)index_y * K + index_x]) : make_float4(0.f,0.f,0.f,0.f);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int offset = i * strideW;
        int index_x = bkIdx + BK + 8 * loadWIdx_x;
        int index_y = BN * blockIdx.x + loadWIdx_y + offset;
        regW[i] = index_x < K && index_y < N ? *reinterpret_cast<const uint4 *>(&w[(size_t)index_y * K + index_x]) : make_uint4(0,0,0,0);
      }
    }
    for (int k = 0; k < BK; k += MFMA_K) {
      for (int iterM = 0; iterM < nIterWaveM; ++iterM) {
        for (int iterN = 0; iterN < nIterWaveN; ++iterN) {
          for (int i = 0; i < 4; ++i) {
            x_local[i] = xs[k + Ak + i][yInBlockTile * WM + MFMA_M * iterM + Ai];
            w_local[i] = ws[k + Bk + i][xInBlockTile * WN + MFMA_N * iterN + Bj];
          }
          CDfloat y_local = acc_local[iterM * nIterWaveN + iterN];
          y_local = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(x_local, w_local, y_local, 0, 0, 0);
          acc_local[iterM * nIterWaveN + iterN] = y_local;
        }
      }
    }
    __syncthreads();
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int offset = i * strideX;
        xs[4 * loadXIdx_x + 0][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].x);
        xs[4 * loadXIdx_x + 1][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].y);
        xs[4 * loadXIdx_x + 2][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].z);
        xs[4 * loadXIdx_x + 3][loadXIdx_y + offset] = f32_to_bf16bits(regX[i].w);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int offset = i * strideW;
        ws[8 * loadWIdx_x + 0][loadWIdx_y + offset] = (unsigned short)(regW[i].x & 0xFFFF);
        ws[8 * loadWIdx_x + 1][loadWIdx_y + offset] = (unsigned short)(regW[i].x >> 16);
        ws[8 * loadWIdx_x + 2][loadWIdx_y + offset] = (unsigned short)(regW[i].y & 0xFFFF);
        ws[8 * loadWIdx_x + 3][loadWIdx_y + offset] = (unsigned short)(regW[i].y >> 16);
        ws[8 * loadWIdx_x + 4][loadWIdx_y + offset] = (unsigned short)(regW[i].z & 0xFFFF);
        ws[8 * loadWIdx_x + 5][loadWIdx_y + offset] = (unsigned short)(regW[i].z >> 16);
        ws[8 * loadWIdx_x + 6][loadWIdx_y + offset] = (unsigned short)(regW[i].w & 0xFFFF);
        ws[8 * loadWIdx_x + 7][loadWIdx_y + offset] = (unsigned short)(regW[i].w >> 16);
      }
    }
    __syncthreads();
  }
  for (int iterM = 0; iterM < nIterWaveM; ++iterM) {
    for (int iterN = 0; iterN < nIterWaveN; ++iterN) {
      int acc_index = iterM * nIterWaveN + iterN;
      for (int i = 0; i < GPRs_CD; ++i) {
        int CDi = 4 * (laneIdx / 16) + (i % 4);
        int CDj = (laneIdx % 16);
        int global_x = blockIdx.x * BN + WN * xInBlockTile + iterN * MFMA_N + CDj;
        int global_y = blockIdx.y * BM + WM * yInBlockTile + iterM * MFMA_M + CDi;
        float result = acc_local[acc_index][i];
        if (global_x < N && global_y < M) {
          if (b) result += __bfloat162float(b[global_x]);
          xout[(size_t)global_y * N + global_x] = result;
        }
      }
    }
  }
}

static inline void getp_matmul_router_bf16(float *xout, float *x, __hip_bfloat16 *w, __hip_bfloat16 *b, int n, int d, int batch_size, hipStream_t stream) {
  PROFILE_FUNCTION();
  const int M = batch_size, N = d, K = n;
  constexpr int BM = 32, BN = 32, BK = 32;
  const int BNt = BN * GETP_BN_AGG;

  dim3 block(256);
  const int gx = (N + BNt - 1) / BNt;
  const int gy = (M + BM - 1) / BM;
  const int grid_xy = max(1, gx) * max(1, gy);

  int dev = 0, cu = 104;
  HIP_CHECK(hipGetDevice(&dev));
  HIP_CHECK(
      hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev));
  if (cu <= 0) cu = 104;

  const int target_cta = cu * 4;
  int splits =
      (grid_xy >= target_cta) ? 1 : (target_cta + grid_xy - 1) / grid_xy;
  splits = min(8, max(1, splits));
  const int max_splits_by_K = max(1, (K + BK - 1) / BK);
  splits = min(splits, max_splits_by_K);

  const size_t shmem = (size_t)2 * (size_t)BK * ((size_t)BM + (size_t)BNt) *
                       sizeof(unsigned short);

  if (splits == 1) {
    dim3 grid(gx, gy);
    matmul_kernel_nosplit<<<grid, block, shmem, stream>>>(
        xout, x, reinterpret_cast<const __hip_bfloat16 *>(w),
        reinterpret_cast<const __hip_bfloat16 *>(b), M, N, K);
    //HIP_CHECK(hipDeviceSynchronize());
    return;
  }

  float *partial = nullptr;
  HIP_CHECK(hipMallocAsync(
      &partial, (size_t)splits * (size_t)M * (size_t)N * sizeof(float),
      stream));

  dim3 grid(gx, gy, splits);
  matmul_kernel_splitk_store<<<grid, block, shmem, stream>>>(
      partial, x, reinterpret_cast<const __hip_bfloat16 *>(w), M, N, K, splits);

  const int total = M * N;
  dim3 rBlock(256);
  dim3 rGrid((total + rBlock.x - 1) / rBlock.x);
  reduce_splitk_with_bias<<<rGrid, rBlock, 0, stream>>>(
      xout, partial, reinterpret_cast<const __hip_bfloat16 *>(b), M, N, splits);

  HIP_CHECK(hipFreeAsync(partial, stream));
  //HIP_CHECK(hipDeviceSynchronize());
}

static inline void getp_matmul_attn_o_bf16(float *xout, float *x, __hip_bfloat16 *w, __hip_bfloat16 *b, int n, int d, int batch_size, hipStream_t stream) {
  PROFILE_FUNCTION();
  const int M = batch_size, N = d, K = n;
  const int BN = 128, BM = 128;
  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  matmul_attn_o_bf16_kernel<<<grid, block, 0, stream>>>(xout, x, w, b, M, N, K);
  //HIP_CHECK(hipDeviceSynchronize());
}

static inline void getp_matmul_logits_bf16(float *xout, float *x, __hip_bfloat16 *w, int n, int d, int batch_size, hipStream_t stream) {
  PROFILE_FUNCTION();
  const int M = batch_size, N = d, K = n;
  const int BN =128, BM = 128, BK = 16;
  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  matmul_logits_bf16_kernel<<<grid, block, 0, stream>>>(xout, x, w, (__hip_bfloat16*)NULL, M, N, K);
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
  if (tid == 0) {
    assert(sidx[0] >= 0 && sidx[0] < V);
    out[b] = sidx[0];
  }
}

static inline void getp_argmax_rows(const float *logits, int V, int *out, int B,
                                    hipStream_t stream) {
  PROFILE_FUNCTION();
  dim3 grid(B), block(1024);
  argmax_rows_kernel<<<grid, block, 0, stream>>>(logits, V, out);
  //HIP_CHECK(hipDeviceSynchronize());
  
}

inline void sync_workers(hipStream_t stream, Barrier &sync_point) {
  /**
    This implement makes sure that threads are synchronized to this point,
    and the stream corresponding to the thread is finished its work after the
    return of this function
   */
  HIP_CHECK(hipStreamSynchronize(stream));
  sync_point.wait();
  // Note: The last synchronization should be uncommented if 1 thread - 1 stream
  // is not guaranteed HIP_CHECK(hipStreamSynchronize(stream)); // for
  // multi-thread access the same stream
}

int *getp_forward_120b(Transformer * /*transformer*/,
                       DeviceTransformer **dev_transformers, GPUWorker *workers,
                       Barrier &sync_point, int thread_idx, int token[],
                       int pos) {
  Config *p = &dev_transformers[thread_idx]->config;
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int n_experts = p->n_experts;

  GPUWorker *worker = &workers[thread_idx];
  int device_index = worker->device_index;

  assert(device_index == thread_idx);

  DeviceTransformerWeights *dev_w = &dev_transformers[device_index]->weights;
  RunState *dev_s = &dev_transformers[device_index]->state;
  RunStateExt *ext = ext_get(device_index);
  hipStream_t compute_stream = dev_transformers[device_index]->compute_stream;
  hipStream_t memory_stream  = dev_transformers[device_index]->memory_stream;

  float *dev_x = dev_s->x;

  HIP_CHECK(hipSetDevice(device_index));

  hipEvent_t event_rmsnorm;
  hipEvent_t event_router_topk;
  hipEvent_t event_memset;
  hipEvent_t events_e_agg[MAXIMUM_GPU];
  HIP_CHECK(hipEventCreate(&event_rmsnorm));
  HIP_CHECK(hipEventCreate(&event_router_topk));
  HIP_CHECK(hipEventCreate(&event_memset));
  for (int i = 0; i < EXPERT_PARALLELISM; ++i) {
    HIP_CHECK(hipEventCreate(&events_e_agg[i]));
  }

  HIP_CHECK(hipMemcpyAsync(dev_s->topk_i, token, sizeof(int) * BATCH_SIZE,
                           hipMemcpyHostToDevice, compute_stream));
  getp_gather_embedding_bf16(dev_x, dev_w->token_embedding_table_bf16,
                             dev_s->topk_i, hidden_dim, BATCH_SIZE,
                             compute_stream);

  float *dev_cos_vals = dev_s->mlp1_out;
  float *dev_sin_vals = dev_s->gate;
  float *dev_inv_freq = dev_s->up;
  float ntk_beta = 32.0f, ntk_alpha = 1.0f;
  getp_compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                       p->initial_context_length, ntk_beta, ntk_alpha,
                       dev_cos_vals, dev_sin_vals, dev_inv_freq,
                       compute_stream);

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_attn_w + 1ll * l * hidden_dim,
                 BATCH_SIZE, hidden_dim, compute_stream);

    const int n_even = (p->n_layers + 1) / 2;
    const int is_even = (((int)l & 1) == 0);
    const int idx_even = ((int)l) >> 1;
    const int idx_odd = ((int)l) >> 1;
    const int tcap =
        (is_even && p->sliding_window > 0) ? p->sliding_window : p->seq_len;
    const size_t layer_toff =
        is_even ? (size_t)idx_even * (size_t)p->sliding_window
                : (size_t)n_even * (size_t)p->sliding_window +
                      (size_t)idx_odd * (size_t)p->seq_len;
    const int tslot =
        (is_even && p->sliding_window > 0) ? (pos % p->sliding_window) : pos;
    const size_t base_off = layer_toff * (size_t)BATCH_SIZE * (size_t)kv_dim;

    __hip_bfloat16 *key_cache =
        reinterpret_cast<__hip_bfloat16 *>(dev_s->key_cache);
    __hip_bfloat16 *value_cache =
        reinterpret_cast<__hip_bfloat16 *>(dev_s->value_cache);
    __hip_bfloat16 *k_slot =
        key_cache + base_off +
        (size_t)tslot * (size_t)BATCH_SIZE * (size_t)kv_dim;
    __hip_bfloat16 *v_slot =
        value_cache + base_off +
        (size_t)tslot * (size_t)BATCH_SIZE * (size_t)kv_dim;

    float *k_step = dev_s->qkv;
    float *v_step = dev_s->qkv + (size_t)BATCH_SIZE * (size_t)kv_dim;

    __hip_bfloat16 *dev_w_qkv =
        dev_w->w_qkv_bf16 +
        1ll * l * hidden_dim *
            (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    __hip_bfloat16 *dev_b_qkv =
        dev_w->b_qkv_bf16 +
        1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

    getp_matmul_qkv_fused_bf16(dev_s->q, k_step, v_step, dev_s->t, dev_w_qkv,
                               dev_b_qkv, hidden_dim, head_dim,
                               p->n_attn_heads, p->n_kv_heads, BATCH_SIZE,
                               compute_stream);

    getp_apply_rotary_emb(dev_s->q, dev_cos_vals, dev_sin_vals,
                          p->n_attn_heads, head_dim, BATCH_SIZE,
                          compute_stream);
    getp_apply_rotary_emb(k_step, dev_cos_vals, dev_sin_vals, p->n_kv_heads,
                          head_dim, BATCH_SIZE, compute_stream);

    kv_store_pair_fp32_to_bf16(k_slot, v_slot, k_step, v_step,
                               BATCH_SIZE * kv_dim, compute_stream);

    {
      const __hip_bfloat16 *k_layer = key_cache + base_off;
      const __hip_bfloat16 *v_layer = value_cache + base_off;
      const float *q_ptr = dev_s->q;
      const float *attn_sinks_layer =
          dev_w->attn_sinks + (size_t)l * p->n_attn_heads;
      getp_flash_attn_decode_bf16(
          dev_s->tb, k_layer, v_layer, q_ptr, attn_sinks_layer, head_dim,
          p->n_attn_heads, p->n_kv_heads, pos, p->seq_len, p->sliding_window,
          (int)l, BATCH_SIZE, tcap, compute_stream);
    }

    __hip_bfloat16 *dev_w_o =
        dev_w->w_o_bf16 + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    __hip_bfloat16 *dev_b_o = dev_w->b_o_bf16 + 1ll * l * hidden_dim;
    getp_matmul_attn_o_bf16(dev_s->tb2, dev_s->tb, dev_w_o, dev_b_o,
      head_dim * p->n_attn_heads, hidden_dim,
      BATCH_SIZE, compute_stream);


    getp_vecadd(dev_x, dev_s->tb2, hidden_dim, BATCH_SIZE, compute_stream);

    getp_rmsnorm(ext->ext_t + (size_t)device_index * BATCH_SIZE * hidden_dim,
                 dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim,
                 BATCH_SIZE, hidden_dim, compute_stream);

    HIP_CHECK(hipEventRecord(event_rmsnorm, compute_stream));

    __hip_bfloat16 *dev_w_router =
        dev_w->w_router_bf16 + 1ll * l * hidden_dim * n_experts;
    __hip_bfloat16 *dev_b_router = dev_w->b_router_bf16 + 1ll * l * n_experts;
    getp_matmul_router_bf16(
      dev_s->router_score,
      ext->ext_t + (size_t)device_index * BATCH_SIZE * hidden_dim,
      dev_w_router, dev_b_router, hidden_dim, n_experts,
      BATCH_SIZE, compute_stream);
  

    getp_router_topk_softmax_batch(
        dev_s->router_score, n_experts, p->experts_per_token,
        ext->ext_topk_v + (size_t)device_index * BATCH_SIZE * p->experts_per_token,
        ext->ext_topk_i + (size_t)device_index * BATCH_SIZE * p->experts_per_token,
        BATCH_SIZE, compute_stream);

    HIP_CHECK(hipEventRecord(event_router_topk, compute_stream));

    HIP_CHECK(hipStreamWaitEvent(memory_stream, event_rmsnorm));
    for (int delta = 1; delta < EXPERT_PARALLELISM; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      GPUWorker *peer_worker = &workers[i];
      int peer_device_index = peer_worker->device_index;
      RunStateExt *peer_ext = ext_get(peer_device_index);
      if (peer_device_index != device_index) {
        HIP_CHECK(hipMemcpyPeerAsync(
            peer_ext->ext_t + (size_t)device_index * BATCH_SIZE * hidden_dim,
            peer_device_index,
            ext->ext_t + (size_t)device_index * BATCH_SIZE * hidden_dim,
            device_index,
            sizeof(float) * BATCH_SIZE * hidden_dim, memory_stream));
      }
    }

    HIP_CHECK(hipStreamWaitEvent(memory_stream, event_router_topk));
    for (int delta = 1; delta < EXPERT_PARALLELISM; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      GPUWorker *peer_worker = &workers[i];
      int peer_device_index = peer_worker->device_index;
      RunStateExt *peer_ext = ext_get(peer_device_index);
      if (peer_device_index != device_index) {
        HIP_CHECK(hipMemcpyPeerAsync(
            peer_ext->ext_topk_i + (size_t)device_index * BATCH_SIZE * p->experts_per_token,
            peer_device_index,
            ext->ext_topk_i + (size_t)device_index * BATCH_SIZE * p->experts_per_token,
            device_index,
            sizeof(int) * BATCH_SIZE * p->experts_per_token, memory_stream));
        HIP_CHECK(hipMemcpyPeerAsync(
            peer_ext->ext_topk_v + (size_t)device_index * BATCH_SIZE * p->experts_per_token,
            peer_device_index,
            ext->ext_topk_v + (size_t)device_index * BATCH_SIZE * p->experts_per_token,
            device_index,
            sizeof(float) * BATCH_SIZE * p->experts_per_token, memory_stream));
      }
    }

    HIP_CHECK(hipStreamSynchronize(memory_stream));
    sync_workers(compute_stream, sync_point);

    HIP_CHECK(hipSetDevice(device_index));

    getp_map_global_to_local_batch(
        ext->ext_topk_i, ext->ext_topk_v, ext->local_ids, ext->local_wts,
        ext->n_local, p->experts_per_token, worker->expert_start,
        worker->expert_end, EXPERT_PARALLELISM * BATCH_SIZE, compute_stream);

    int experts_per_device = worker->expert_end - worker->expert_start;

    int cap_pairs = build_moe_buckets_local_pos(
        ext->local_ids, ext->local_wts, ext->n_local,
        EXPERT_PARALLELISM * BATCH_SIZE, p->experts_per_token,
        experts_per_device, ext->e_counts, ext->e_offsets, ext->e_dev,
        ext->w_dev, ext->pair_pos, compute_stream);

    int total_pairs = 0;
    HIP_CHECK(hipMemcpyAsync(&total_pairs, ext->e_offsets + experts_per_device,
                             sizeof(int), hipMemcpyDeviceToHost,
                             compute_stream));

    int total_blocks = build_moe_block_schedule(
        ext->e_offsets, experts_per_device, 64, ext->blk_counts,
        ext->blk_offsets, compute_stream);

    moe_scatter_acts_to_expert_bf16(ext->a_in, ext->ext_t, ext->e_dev,
                                    total_pairs, hidden_dim, compute_stream);

    HIP_CHECK(hipMemsetAsync(
        ext->ext_e_agg, 0,
        (size_t)EXPERT_PARALLELISM * BATCH_SIZE * hidden_dim * sizeof(float),
        memory_stream));
    HIP_CHECK(hipEventRecord(event_memset, memory_stream));

    __hip_bfloat16 *w1_base = dev_w->w_mlp1 + 1ll * l * experts_per_device * 2 *
                              p->intermediate_dim * hidden_dim;
    __hip_bfloat16 *b1_base = dev_w->b_mlp1 +
                              1ll * l * experts_per_device * 2 * p->intermediate_dim;

    launch_mlp1_swiglu_bf16_bucketed_outbf16(
        ext->gate_up_bf16, ext->a_in, w1_base, b1_base, ext->e_offsets,
        ext->blk_offsets, hidden_dim, p->intermediate_dim, experts_per_device,
        total_blocks, p->swiglu_limit, compute_stream);

    __hip_bfloat16 *w2_base = dev_w->w_mlp2 + 1ll * l * experts_per_device *
                              hidden_dim * p->intermediate_dim;
    __hip_bfloat16 *b2_base = dev_w->b_mlp2 +
                              1ll * l * experts_per_device * hidden_dim;

    launch_mlp2_partial_bf16_bucketed_frombf16(
        ext->z_partial, ext->gate_up_bf16, w2_base, b2_base, ext->w_dev,
        ext->e_offsets, ext->blk_offsets, ext->e_dev, p->intermediate_dim,
        hidden_dim, experts_per_device, total_blocks, compute_stream);

    HIP_CHECK(hipStreamWaitEvent(compute_stream, event_memset));
    moe_gather_pairs(ext->ext_e_agg, ext->z_partial, ext->pair_pos, hidden_dim,
                     p->experts_per_token, EXPERT_PARALLELISM * BATCH_SIZE,
                     compute_stream);

    sync_workers(compute_stream, sync_point);

    getp_vecadd(dev_x,
                ext->ext_e_agg + (size_t)device_index * BATCH_SIZE * hidden_dim,
                hidden_dim, BATCH_SIZE, compute_stream);

    for (int delta = 1, j = 0; delta < EXPERT_PARALLELISM; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      GPUWorker *peer_worker = &workers[i];
      int peer_device_index = peer_worker->device_index;
      RunStateExt *peer_ext = ext_get(peer_device_index);
      if (peer_device_index != device_index) {
        HIP_CHECK(hipMemcpyPeerAsync(
            ext->peer_e_agg + (size_t)j * BATCH_SIZE * hidden_dim, device_index,
            peer_ext->ext_e_agg + (size_t)device_index * BATCH_SIZE * hidden_dim,
            peer_device_index,
            sizeof(float) * BATCH_SIZE * hidden_dim, memory_stream));
        HIP_CHECK(hipEventRecord(events_e_agg[j], memory_stream));
        ++j;
      }
    }
    for (int delta = 1, j = 0; delta < EXPERT_PARALLELISM; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      GPUWorker *peer_worker = &workers[i];
      int peer_device_index = peer_worker->device_index;
      RunStateExt *peer_ext = ext_get(peer_device_index);
      if (peer_device_index != device_index) {
        HIP_CHECK(hipStreamWaitEvent(compute_stream, events_e_agg[j]));
        getp_vecadd(dev_x,
                    ext->peer_e_agg + (size_t)j * BATCH_SIZE * hidden_dim,
                    hidden_dim, BATCH_SIZE, compute_stream);
        ++j;
      }
    }
  }

  getp_rmsnorm(dev_x, dev_x, dev_w->rms_out_w, BATCH_SIZE, hidden_dim,
               compute_stream);
  getp_matmul_logits_bf16(dev_s->logits, dev_x, dev_w->out_bf16,
    hidden_dim, p->vocab_size,
    BATCH_SIZE, compute_stream);


  getp_argmax_rows(dev_s->logits, p->vocab_size, dev_s->topk_i, BATCH_SIZE,
                   compute_stream);

  int *next_host = (int *)malloc(sizeof(int) * BATCH_SIZE);
  HIP_CHECK(hipMemcpyAsync(next_host, dev_s->topk_i, sizeof(int) * BATCH_SIZE,
                           hipMemcpyDeviceToHost, compute_stream));
  HIP_CHECK(hipStreamSynchronize(compute_stream));
  return next_host;
}

int *getp_forward_20b(Transformer * /*transformer*/,
                      DeviceTransformer **dev_transformeres, GPUWorker *worker,
                      int token[], int pos, int batch_size) {
  
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

  HIP_CHECK(hipMemcpy(dev_s->topk_i, token, sizeof(int) * batch_size,
                      hipMemcpyHostToDevice));
  getp_gather_embedding_bf16(dev_x, dev_w->token_embedding_table_bf16,
                             dev_s->topk_i, hidden_dim, batch_size, nullptr);

  float *dev_cos_vals = dev_s->mlp1_out;
  float *dev_sin_vals = dev_s->gate;
  float *dev_inv_freq = dev_s->up;
  float ntk_beta = 32.0f, ntk_alpha = 1.0f;
  getp_compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                       p->initial_context_length, ntk_beta, ntk_alpha,
                       dev_cos_vals, dev_sin_vals, dev_inv_freq, nullptr);

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_attn_w + 1ll * l * hidden_dim,
                 batch_size, hidden_dim, nullptr);

    const int n_even = (p->n_layers + 1) / 2;
    const int is_even = (((int)l & 1) == 0);
    const int idx_even = ((int)l) >> 1;
    const int idx_odd = ((int)l) >> 1;
    const int tcap =
        (is_even && p->sliding_window > 0) ? p->sliding_window : p->seq_len;
    const size_t layer_toff = is_even
                                  ? (size_t)idx_even * (size_t)p->sliding_window
                                  : (size_t)n_even * (size_t)p->sliding_window +
                                        (size_t)idx_odd * (size_t)p->seq_len;
    const int tslot =
        (is_even && p->sliding_window > 0) ? (pos % p->sliding_window) : pos;
    const size_t base_off = layer_toff * (size_t)BATCH_SIZE * (size_t)kv_dim;

    __hip_bfloat16 *key_cache =
        reinterpret_cast<__hip_bfloat16 *>(dev_s->key_cache);
    __hip_bfloat16 *value_cache =
        reinterpret_cast<__hip_bfloat16 *>(dev_s->value_cache);
    __hip_bfloat16 *k_slot =
        key_cache + base_off +
        (size_t)tslot * (size_t)BATCH_SIZE * (size_t)kv_dim;
    __hip_bfloat16 *v_slot =
        value_cache + base_off +
        (size_t)tslot * (size_t)BATCH_SIZE * (size_t)kv_dim;

    float *k_step = dev_s->qkv;
    float *v_step = dev_s->qkv + (size_t)BATCH_SIZE * (size_t)kv_dim;

    __hip_bfloat16 *dev_w_qkv =
        dev_w->w_qkv_bf16 +
        1ll * l * hidden_dim *
            (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    __hip_bfloat16 *dev_b_qkv =
        dev_w->b_qkv_bf16 +
        1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    getp_matmul_qkv_fused_bf16(dev_s->q, k_step, v_step, dev_s->t, dev_w_qkv,
                               dev_b_qkv, hidden_dim, head_dim, p->n_attn_heads,
                               p->n_kv_heads, batch_size, nullptr);

    getp_apply_rotary_emb(dev_s->q, dev_cos_vals, dev_sin_vals, p->n_attn_heads,
                          head_dim, batch_size, nullptr);
    getp_apply_rotary_emb(k_step, dev_cos_vals, dev_sin_vals, p->n_kv_heads,
                          head_dim, batch_size, nullptr);

    kv_store_pair_fp32_to_bf16(k_slot, v_slot, k_step, v_step,
                               batch_size * kv_dim, nullptr);

    {
      const __hip_bfloat16 *k_layer = key_cache + base_off;
      const __hip_bfloat16 *v_layer = value_cache + base_off;
      const float *q_ptr = dev_s->q;
      const float *attn_sinks_layer =
          dev_w->attn_sinks + (size_t)l * p->n_attn_heads;
      getp_flash_attn_decode_bf16(
          dev_s->tb, k_layer, v_layer, q_ptr, attn_sinks_layer, head_dim,
          p->n_attn_heads, p->n_kv_heads, pos, p->seq_len, p->sliding_window,
          (int)l, batch_size, tcap, nullptr);
    }

    __hip_bfloat16 *dev_w_o =
        dev_w->w_o_bf16 + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    __hip_bfloat16 *dev_b_o = dev_w->b_o_bf16 + 1ll * l * hidden_dim;
    // getp_new_matmul(dev_s->tb2, dev_s->tb, dev_w_o, dev_b_o,
    //                     head_dim * p->n_attn_heads, hidden_dim,
    //                     batch_size, nullptr);
    getp_matmul_attn_o_bf16(dev_s->tb2, dev_s->tb, dev_w_o, dev_b_o,
      head_dim * p->n_attn_heads, hidden_dim,
      batch_size, nullptr);
    

    getp_vecadd(dev_x, dev_s->tb2, hidden_dim, batch_size, nullptr);

    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim,
                 batch_size, hidden_dim, nullptr);

    __hip_bfloat16 *dev_w_router =
        dev_w->w_router_bf16 + 1ll * l * hidden_dim * n_experts;
    __hip_bfloat16 *dev_b_router = dev_w->b_router_bf16 + 1ll * l * n_experts;
    // getp_matmul<__hip_bfloat16>(dev_s->router_score, dev_s->t, dev_w_router,
    //                             dev_b_router, hidden_dim, n_experts, batch_size,
    //                             nullptr);
    getp_matmul_router_bf16(dev_s->router_score, dev_s->t, dev_w_router,
      dev_b_router, hidden_dim, n_experts, batch_size, nullptr);
    

    getp_router_topk_softmax_batch(dev_s->router_score, n_experts,
                                   p->experts_per_token, dev_s->topk_v,
                                   dev_s->topk_i, batch_size, nullptr);

    getp_map_global_to_local_batch(dev_s->topk_i, dev_s->topk_v, ext->local_ids,
                                   ext->local_wts, ext->n_local,
                                   p->experts_per_token, worker->expert_start,
                                   worker->expert_end, batch_size, nullptr);

    int experts_per_device = worker->expert_end - worker->expert_start;

    int cap_pairs = build_moe_buckets_local_pos(
        ext->local_ids, ext->local_wts, ext->n_local, batch_size,
        p->experts_per_token, experts_per_device, ext->e_counts, ext->e_offsets,
        ext->e_dev, ext->w_dev, ext->pair_pos, nullptr);

    int total_pairs = 0;
    HIP_CHECK(hipMemcpy(&total_pairs, ext->e_offsets + experts_per_device,
                        sizeof(int), hipMemcpyDeviceToHost));

    int total_blocks =
        build_moe_block_schedule(ext->e_offsets, experts_per_device, 64,
                                 ext->blk_counts, ext->blk_offsets, nullptr);

    moe_scatter_acts_to_expert_bf16(ext->a_in, dev_s->t, ext->e_dev,
                                    total_pairs, hidden_dim, nullptr);

    HIP_CHECK(hipMemset(dev_s->e_agg, 0,
                        (size_t)batch_size * hidden_dim * sizeof(float)));

    __hip_bfloat16 *w1_base = dev_w->w_mlp1 + 1ll * l * experts_per_device * 2 *
                                                  p->intermediate_dim *
                                                  hidden_dim;
    __hip_bfloat16 *b1_base =
        dev_w->b_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim;

    launch_mlp1_swiglu_bf16_bucketed_outbf16(
        ext->gate_up_bf16, ext->a_in, w1_base, b1_base, ext->e_offsets,
        ext->blk_offsets, hidden_dim, p->intermediate_dim, experts_per_device,
        total_blocks, p->swiglu_limit, nullptr);

    __hip_bfloat16 *w2_base = dev_w->w_mlp2 + 1ll * l * experts_per_device *
                                                  hidden_dim *
                                                  p->intermediate_dim;
    __hip_bfloat16 *b2_base =
        dev_w->b_mlp2 + 1ll * l * experts_per_device * hidden_dim;

    launch_mlp2_partial_bf16_bucketed_frombf16(
        ext->z_partial, ext->gate_up_bf16, w2_base, b2_base, ext->w_dev,
        ext->e_offsets, ext->blk_offsets, ext->e_dev, p->intermediate_dim,
        hidden_dim, experts_per_device, total_blocks, nullptr);

    moe_gather_pairs(dev_s->e_agg, ext->z_partial, ext->pair_pos, hidden_dim,
                     p->experts_per_token, batch_size, nullptr);

    getp_vecadd(dev_x, dev_s->e_agg, hidden_dim, batch_size, nullptr);
  }

  getp_rmsnorm(dev_x, dev_x, dev_w->rms_out_w, batch_size, hidden_dim, nullptr);
  // getp_new_matmul(dev_s->logits, dev_x, dev_w->out_bf16,
  //                     (__hip_bfloat16 *)NULL, hidden_dim, p->vocab_size,
  //                     batch_size, nullptr);
  getp_matmul_logits_bf16(dev_s->logits, dev_x, dev_w->out_bf16,
    hidden_dim, p->vocab_size,
    batch_size, nullptr);
  

  getp_argmax_rows(dev_s->logits, p->vocab_size, dev_s->topk_i, batch_size, nullptr);
  int *next_host = (int *)malloc(sizeof(int) * batch_size);
  HIP_CHECK(hipMemcpy(next_host, dev_s->topk_i, sizeof(int) * batch_size,
                      hipMemcpyDeviceToHost));
                      //HIP_CHECK(hipDeviceSynchronize());
  return next_host;
}
