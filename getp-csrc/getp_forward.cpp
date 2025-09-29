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

__global__ void fp32_to_bf16_kernel(const float *__restrict__ src,
                                    __hip_bfloat16 *__restrict__ dst,
                                    int elems) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int base = idx << 2;
  if (base >= elems) return;
  if (base + 3 < elems) {
    float4 v = reinterpret_cast<const float4 *>(src)[idx];
    uint2 packed;
    packed.x = (unsigned)f32_to_bf16bits(v.x) |
               ((unsigned)f32_to_bf16bits(v.y) << 16);
    packed.y = (unsigned)f32_to_bf16bits(v.z) |
               ((unsigned)f32_to_bf16bits(v.w) << 16);
    reinterpret_cast<uint2 *>(dst)[idx] = packed;
  } else {
    unsigned short *dst_u16 = reinterpret_cast<unsigned short *>(dst);
    for (int t = 0; t < 4; ++t) {
      int e = base + t;
      if (e < elems) dst_u16[e] = f32_to_bf16bits(src[e]);
    }
  }
}

static inline void tensor_fp32_to_bf16(__hip_bfloat16 *dst, const float *src,
                                       int elems, hipStream_t stream) {
  PROFILE_FUNCTION();
  const int threads = 256;
  const int vec_elems = (elems + 3) / 4;
  const int blocks = (vec_elems + threads - 1) / threads;
  fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(src, dst, elems);
  // HIP_CHECK(hipGetLastError());
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

__global__ void moe_scatter_acts_to_expert_frombf16_kernel(
  __hip_bfloat16 *a_in, const __hip_bfloat16 *x_bf16,
  const int *tok_idx, int total_pairs, int H) {
int pos = blockIdx.x;
if (pos >= total_pairs) return;
int b = tok_idx[pos];
const __hip_bfloat16 *src = x_bf16 + (size_t)b * H;
__hip_bfloat16 *dst = a_in + (size_t)pos * H;
int t = threadIdx.x;
int step8 = blockDim.x * 8;
int H8 = H & ~7;
for (int kk8 = t * 8; kk8 < H8; kk8 += step8) {
  reinterpret_cast<uint4*>(dst)[kk8 >> 3] =
      reinterpret_cast<const uint4*>(src)[kk8 >> 3];
}
for (int kk = H8 + t; kk < H; kk += blockDim.x)
  reinterpret_cast<unsigned short*>(dst)[kk] =
      reinterpret_cast<const unsigned short*>(src)[kk];
}

static inline void moe_scatter_acts_to_expert_frombf16(
  __hip_bfloat16 *a_in, const __hip_bfloat16 *x_bf16, const int *tok_idx,
  int total_pairs, int H, hipStream_t stream) {
dim3 block(256), grid(total_pairs);
moe_scatter_acts_to_expert_frombf16_kernel<<<grid, block, 0, stream>>>(
    a_in, x_bf16, tok_idx, total_pairs, H);
}

#ifndef MATMUL_MLP1_BLOCK_ROWS
#define MATMUL_MLP1_BLOCK_ROWS 64
#endif
#ifndef MATMUL_MLP1_BLOCK_COLS
#define MATMUL_MLP1_BLOCK_COLS 64
#endif
#ifndef MATMUL_MLP1_BLOCK_DEPTH
#define MATMUL_MLP1_BLOCK_DEPTH 32
#endif
#ifndef MATMUL_MLP1_WARP_TILE_M
#define MATMUL_MLP1_WARP_TILE_M 32
#endif
#ifndef MATMUL_MLP1_WARP_TILE_N
#define MATMUL_MLP1_WARP_TILE_N 32
#endif
#ifndef MATMUL_MLP1_WAVES_PER_BLOCK
#define MATMUL_MLP1_WAVES_PER_BLOCK 4
#endif
#ifndef NUM_CTA_MLP1
#define NUM_CTA_MLP1 2
#endif

#ifndef MATMUL_MLP2_BLOCK_ROWS
#define MATMUL_MLP2_BLOCK_ROWS 64
#endif
#ifndef MATMUL_MLP2_BLOCK_COLS
#define MATMUL_MLP2_BLOCK_COLS 64
#endif
#ifndef MATMUL_MLP2_BLOCK_DEPTH
#define MATMUL_MLP2_BLOCK_DEPTH 32
#endif
#ifndef MATMUL_MLP2_WARP_TILE_M
#define MATMUL_MLP2_WARP_TILE_M 32
#endif
#ifndef MATMUL_MLP2_WARP_TILE_N
#define MATMUL_MLP2_WARP_TILE_N 32
#endif
#ifndef MATMUL_MLP2_WAVES_PER_BLOCK
#define MATMUL_MLP2_WAVES_PER_BLOCK 4
#endif
#ifndef NUM_CTA_MLP2
#define NUM_CTA_MLP2 4
#endif


#ifndef MATMUL_ROUTER_BLOCK_ROWS
#define MATMUL_ROUTER_BLOCK_ROWS 32
#endif
#ifndef MATMUL_ROUTER_BLOCK_COLS
#define MATMUL_ROUTER_BLOCK_COLS 32
#endif
#ifndef MATMUL_ROUTER_BLOCK_DEPTH
#define MATMUL_ROUTER_BLOCK_DEPTH 32
#endif
#ifndef MATMUL_ROUTER_WARP_TILE_M
#define MATMUL_ROUTER_WARP_TILE_M 16
#endif
#ifndef MATMUL_ROUTER_WARP_TILE_N
#define MATMUL_ROUTER_WARP_TILE_N 16
#endif
#ifndef MATMUL_ROUTER_WAVES_PER_BLOCK
#define MATMUL_ROUTER_WAVES_PER_BLOCK 4
#endif
#ifndef MATMUL_ROUTER_THREADS
#define MATMUL_ROUTER_THREADS (64 * MATMUL_ROUTER_WAVES_PER_BLOCK)
#endif

template<
  int BM, int BN, int BK,
  int WM, int WN,
  int BN_AGG, int CTA, 
  int TILE_N = 64 * ((BM/WM)*(BN/WN)),
  int WARPS_PER_BLOCK = (BM/WM)*(BN/WN)
>
__global__ void __launch_bounds__(TILE_N, CTA)
mlp1_swiglu_bf16_bucketed_kernel_outbf16_tuned(
  __hip_bfloat16* __restrict__ gate_up_bf16,
  const __hip_bfloat16* __restrict__ A_in,
  const __hip_bfloat16* __restrict__ W1,
  const __hip_bfloat16* __restrict__ B1,
  const int* __restrict__ offsets,
  const int* __restrict__ blk_offs,
  int H, int I, int E, float swiglu_limit
) {
  static_assert(BM % WM == 0 && BN % WN == 0, "tile mismatch");
  static_assert(BK % 16 == 0, "BK % 16");
  static_assert(WM % 16 == 0 && WN % 16 == 0, "WM/WN % 16");
  static_assert(TILE_N == 64 * WARPS_PER_BLOCK, "launch bounds");
  constexpr int BNt = BN * BN_AGG;
  constexpr int nIterM = WM / 16;
  constexpr int nIterN = WN / 16;

  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  constexpr int splitM = BM / WM;
  constexpr int splitN = BN / WN;
  static_assert(WARPS_PER_BLOCK == splitM * splitN, "warps cfg");
  const int xTile = wave % splitN;
  const int yTile = wave / splitN;
  const int xMF = lane & 15;
  const int yMF = lane >> 4;

  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  constexpr int strideX = TILE_N / (BK / 4);
  constexpr int strideW = TILE_N / (BK / 8);

  extern __shared__ unsigned short sm[];
  unsigned short* sx  = sm;
  unsigned short* swg = sx + (size_t)BK * BM;
  unsigned short* swu = swg + (size_t)BK * BNt;

  f32x4 dg_acc[BN_AGG][nIterM][nIterN];
  f32x4 du_acc[BN_AGG][nIterM][nIterN];
  #pragma unroll
  for (int g = 0; g < BN_AGG; ++g)
    for (int im = 0; im < nIterM; ++im)
      for (int in = 0; in < nIterN; ++in) { dg_acc[g][im][in] = {0,0,0,0}; du_acc[g][im][in] = {0,0,0,0}; }

  int by = blockIdx.y;
  int lo = 0, hi = E;
  while (lo + 1 < hi) { int mid = (lo + hi) >> 1; int off = blk_offs[mid]; if (by < off) hi = mid; else lo = mid; }
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
  const int colBase = blockIdx.x * BNt;

  for (int bk = 0; bk < H; bk += BK) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk = bk + 4 * loadX_x;
      const int r  = loadX_y + off;
      const int pos = base + r;
      if (r < tcnt) {
        const __hip_bfloat16* p = A_in + (size_t)pos * H + kk;
        if (kk + 3 < H) {
          uint2 v = *reinterpret_cast<const uint2*>(p);
          sx[(4 * loadX_x + 0) * BM + r] = (unsigned short)(v.x & 0xFFFF);
          sx[(4 * loadX_x + 1) * BM + r] = (unsigned short)(v.x >> 16);
          sx[(4 * loadX_x + 2) * BM + r] = (unsigned short)(v.y & 0xFFFF);
          sx[(4 * loadX_x + 3) * BM + r] = (unsigned short)(v.y >> 16);
        } else {
          const unsigned short* q = reinterpret_cast<const unsigned short*>(p);
          sx[(4 * loadX_x + 0) * BM + r] = (kk + 0 < H) ? q[0] : 0;
          sx[(4 * loadX_x + 1) * BM + r] = (kk + 1 < H) ? q[1] : 0;
          sx[(4 * loadX_x + 2) * BM + r] = (kk + 2 < H) ? q[2] : 0;
          sx[(4 * loadX_x + 3) * BM + r] = (kk + 3 < H) ? q[3] : 0;
        }
      } else {
        sx[(4 * loadX_x + 0) * BM + r] = 0;
        sx[(4 * loadX_x + 1) * BM + r] = 0;
        sx[(4 * loadX_x + 2) * BM + r] = 0;
        sx[(4 * loadX_x + 3) * BM + r] = 0;
      }
    }

    for (int off = 0; off < BNt; off += strideW) {
      if (loadW_y + off >= BNt) break;
      const int kk = bk + 8 * loadW_x;
      const int col = colBase + loadW_y + off;

      uint4 wg = {0,0,0,0}, wu = {0,0,0,0};
      if (col < I) {
        if (kk + 7 < H) {
          wg = *reinterpret_cast<const uint4*>(W1 + w1_e + (size_t)(2 * col + 0) * H + kk);
          wu = *reinterpret_cast<const uint4*>(W1 + w1_e + (size_t)(2 * col + 1) * H + kk);
        } else if (kk < H) {
          const __hip_bfloat16* pg = W1 + w1_e + (size_t)(2 * col + 0) * H + kk;
          const __hip_bfloat16* pu = W1 + w1_e + (size_t)(2 * col + 1) * H + kk;
          unsigned u0 = 0, u1 = 0, u2 = 0, u3 = 0, v0 = 0, v1 = 0, v2 = 0, v3 = 0;
          if (kk + 0 < H) { u0 |= (unsigned)(reinterpret_cast<const unsigned short*>(pg)[0]); v0 |= (unsigned)(reinterpret_cast<const unsigned short*>(pu)[0]); }
          if (kk + 1 < H) { u0 |= (unsigned)(reinterpret_cast<const unsigned short*>(pg)[1]) << 16; v0 |= (unsigned)(reinterpret_cast<const unsigned short*>(pu)[1]) << 16; }
          if (kk + 2 < H) { u1 |= (unsigned)(reinterpret_cast<const unsigned short*>(pg)[2]); v1 |= (unsigned)(reinterpret_cast<const unsigned short*>(pu)[2]); }
          if (kk + 3 < H) { u1 |= (unsigned)(reinterpret_cast<const unsigned short*>(pg)[3]) << 16; v1 |= (unsigned)(reinterpret_cast<const unsigned short*>(pu)[3]) << 16; }
          if (kk + 4 < H) { u2 |= (unsigned)(reinterpret_cast<const unsigned short*>(pg)[4]); v2 |= (unsigned)(reinterpret_cast<const unsigned short*>(pu)[4]); }
          if (kk + 5 < H) { u2 |= (unsigned)(reinterpret_cast<const unsigned short*>(pg)[5]) << 16; v2 |= (unsigned)(reinterpret_cast<const unsigned short*>(pu)[5]) << 16; }
          if (kk + 6 < H) { u3 |= (unsigned)(reinterpret_cast<const unsigned short*>(pg)[6]); v3 |= (unsigned)(reinterpret_cast<const unsigned short*>(pu)[6]); }
          wg = {u0,u1,u2,u3};
          wu = {v0,v1,v2,v3};
        }
      }

      unsigned short* dg = swg + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
      unsigned short* du = swu + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
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
        #pragma unroll
        for (int t2 = 0; t2 < 8; ++t2) { dg[t2 * BNt] = 0; du[t2 * BNt] = 0; }
      }
    }

    __syncthreads();

    for (int k = 0; k < BK; k += 16) {
      const int basek = k + yMF * 4;
      #pragma unroll
      for (int im = 0; im < nIterM; ++im) {
        const int row = yTile * WM + im * 16 + xMF;
        bf16x4 av = {sx[(basek + 0) * BM + row], sx[(basek + 1) * BM + row], sx[(basek + 2) * BM + row], sx[(basek + 3) * BM + row]};
        #pragma unroll
        for (int g = 0; g < BN_AGG; ++g) {
          bf16x4 gv_next = {0, 0, 0, 0};
          bf16x4 uv_next = {0, 0, 0, 0};
          #pragma unroll
          for (int in = 0; in < nIterN; ++in) {
            const int c = xTile * WN + in * 16 + xMF;
            if (in == 0) {
              gv_next = {swg[(basek + 0) * BNt + c + g * BN],
                         swg[(basek + 1) * BNt + c + g * BN],
                         swg[(basek + 2) * BNt + c + g * BN],
                         swg[(basek + 3) * BNt + c + g * BN]};
              uv_next = {swu[(basek + 0) * BNt + c + g * BN],
                         swu[(basek + 1) * BNt + c + g * BN],
                         swu[(basek + 2) * BNt + c + g * BN],
                         swu[(basek + 3) * BNt + c + g * BN]};
            }
            bf16x4 gv_cur = gv_next;
            bf16x4 uv_cur = uv_next;
            if (in + 1 < nIterN) {
              const int c_next = xTile * WN + (in + 1) * 16 + xMF;
              gv_next = {swg[(basek + 0) * BNt + c_next + g * BN],
                         swg[(basek + 1) * BNt + c_next + g * BN],
                         swg[(basek + 2) * BNt + c_next + g * BN],
                         swg[(basek + 3) * BNt + c_next + g * BN]};
              uv_next = {swu[(basek + 0) * BNt + c_next + g * BN],
                         swu[(basek + 1) * BNt + c_next + g * BN],
                         swu[(basek + 2) * BNt + c_next + g * BN],
                         swu[(basek + 3) * BNt + c_next + g * BN]};
            }
            dg_acc[g][im][in] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, gv_cur, dg_acc[g][im][in], 0, 0, 0);
            du_acc[g][im][in] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, uv_cur, du_acc[g][im][in], 0, 0, 0);
          }
        }
      }
    }

    __syncthreads();
  }

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);

  float bg_loc[BN_AGG][nIterN];
  float bu_loc[BN_AGG][nIterN];
  #pragma unroll
  for (int g = 0; g < BN_AGG; ++g) {
    #pragma unroll
    for (int in = 0; in < nIterN; ++in) {
      const int col = blockIdx.x * BNt + xTile * WN + in * 16 + xD + g * BN;
      if (col < I) {
        const size_t bias_idx = b1_e + (size_t)(2 * col);
        bg_loc[g][in] = B1 ? __bfloat162float(B1[bias_idx + 0]) : 0.f;
        bu_loc[g][in] = B1 ? __bfloat162float(B1[bias_idx + 1]) : 0.f;
      } else {
        bg_loc[g][in] = 0.f;
        bu_loc[g][in] = 0.f;
      }
    }
  }
  for (int im = 0; im < nIterM; ++im) {
    for (int i = 0; i < 4; ++i) {
      const int row = yTile * WM + im * 16 + (yD + i);
      if (row >= tcnt) continue;
      const int pos = base + row;
      for (int g = 0; g < BN_AGG; ++g) {
        #pragma unroll
        for (int in = 0; in < nIterN; ++in) {
          const int col = blockIdx.x * BNt + xTile * WN + in * 16 + xD + g * BN;
          if (col >= I) continue;
          float gval = dg_acc[g][im][in][i] + bg_loc[g][in];
          float uval = du_acc[g][im][in][i] + bu_loc[g][in];
          if (gval > swiglu_limit) gval = swiglu_limit;
          if (uval > swiglu_limit) uval = swiglu_limit;
          if (uval < -swiglu_limit) uval = -swiglu_limit;
          const float a = 1.702f;
          float sgm = 1.f / (1.f + __expf(-a * gval));
          float out = (gval * sgm) * (uval + 1.f);
          reinterpret_cast<unsigned short*>(gate_up_bf16)[(size_t)pos * I + col] = f32_to_bf16bits(out);
        }
      }
    }
  }
}

template<
  int BM, int BN, int BK,
  int WM, int WN,
  int BN_AGG, int CTA,
  int TILE_N = 64 * ((BM/WM)*(BN/WN)),
  int WARPS_PER_BLOCK = (BM/WM)*(BN/WN)
>
__global__ void __launch_bounds__(TILE_N, CTA)
mlp2_partial_bf16_bucketed_splitk_kernel_inbf16_tuned(
  float* __restrict__ z_partial,
  const __hip_bfloat16* __restrict__ gate_up_bf16,
  const __hip_bfloat16* __restrict__ W2,
  const __hip_bfloat16* __restrict__ B2,
  const float* __restrict__ w_by_bucket,
  const int* __restrict__ offsets,
  const int* __restrict__ blk_offs,
  const int* __restrict__ tok_idx,
  int I, int H, int E, int splits
) {
  static_assert(BM % WM == 0 && BN % WN == 0, "tile mismatch");
  static_assert(BK % 16 == 0, "BK % 16");
  static_assert(WM % 16 == 0 && WN % 16 == 0, "WM/WN % 16");
  static_assert(TILE_N == 64 * WARPS_PER_BLOCK, "launch bounds");
  constexpr int BNt = BN * BN_AGG;
  constexpr int nIterM = WM / 16;
  constexpr int nIterN = WN / 16;

  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  constexpr int splitM = BM / WM;
  constexpr int splitN = BN / WN;
  static_assert(WARPS_PER_BLOCK == splitM * splitN, "warps cfg");
  const int xTile = wave % splitN;
  const int yTile = wave / splitN;
  const int xMF = lane & 15;
  const int yMF = lane >> 4;

  extern __shared__ unsigned short sm[];
  unsigned short* sx = sm;
  unsigned short* sw = sx + (size_t)BK * (size_t)BM;

  f32x4 d_acc[BN_AGG][nIterM][nIterN];
  #pragma unroll
  for (int g = 0; g < BN_AGG; ++g)
    for (int im = 0; im < nIterM; ++im)
      for (int in = 0; in < nIterN; ++in)
        d_acc[g][im][in] = {0,0,0,0};

  const int loadX_x = threadIdx.x % (BK / 4);
  const int loadX_y = threadIdx.x / (BK / 4);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  constexpr int strideX = TILE_N / (BK / 4);
  constexpr int strideW = TILE_N / (BK / 8);

  int by = blockIdx.y;
  int lo = 0, hi = E;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    int off = blk_offs[mid];
    if (by < off) hi = mid; else lo = mid;
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

  const int split = blockIdx.z;
  const int kChunk = (I + splits - 1) / splits;
  const int kBeg = split * kChunk;
  const int kEnd = min(I, kBeg + kChunk);

  const int colBase = blockIdx.x * BNt;

  for (int bk = kBeg; bk < kEnd; bk += BK) {
    for (int off = 0; off < BM; off += strideX) {
      if (loadX_y + off >= BM) break;
      const int kk = bk + 4 * loadX_x;
      const int r = loadX_y + off;
      const int pos = base + r;
      if (r < tcnt) {
        const __hip_bfloat16* src = gate_up_bf16 + (size_t)pos * (size_t)I + kk;
        if (kk + 3 < kEnd) {
          uint2 v = *reinterpret_cast<const uint2*>(src);
          sx[(4 * loadX_x + 0) * BM + r] = (unsigned short)(v.x & 0xFFFF);
          sx[(4 * loadX_x + 1) * BM + r] = (unsigned short)(v.x >> 16);
          sx[(4 * loadX_x + 2) * BM + r] = (unsigned short)(v.y & 0xFFFF);
          sx[(4 * loadX_x + 3) * BM + r] = (unsigned short)(v.y >> 16);
        } else {
          const unsigned short* q = reinterpret_cast<const unsigned short*>(src);
          sx[(4 * loadX_x + 0) * BM + r] = (kk + 0 < kEnd) ? q[0] : 0;
          sx[(4 * loadX_x + 1) * BM + r] = (kk + 1 < kEnd) ? q[1] : 0;
          sx[(4 * loadX_x + 2) * BM + r] = (kk + 2 < kEnd) ? q[2] : 0;
          sx[(4 * loadX_x + 3) * BM + r] = (kk + 3 < kEnd) ? q[3] : 0;
        }
      } else {
        sx[(4 * loadX_x + 0) * BM + r] = 0;
        sx[(4 * loadX_x + 1) * BM + r] = 0;
        sx[(4 * loadX_x + 2) * BM + r] = 0;
        sx[(4 * loadX_x + 3) * BM + r] = 0;
      }
    }

    for (int off = 0; off < BNt; off += strideW) {
      if (loadW_y + off >= BNt) break;
      const int kk = bk + 8 * loadW_x;
      const int col = colBase + loadW_y + off;
      uint4 wb = {0,0,0,0};
      if (col < H && kk < kEnd) {
        if (kk + 7 < kEnd) {
          wb = *reinterpret_cast<const uint4*>(W2 + w2_e + (size_t)col * (size_t)I + kk);
        } else {
          const __hip_bfloat16* src = W2 + w2_e + (size_t)col * (size_t)I + kk;
          unsigned u0 = 0, u1 = 0, u2 = 0, u3 = 0;
          if (kk + 0 < kEnd) u0 |= (unsigned)(reinterpret_cast<const unsigned short*>(src)[0]);
          if (kk + 1 < kEnd) u0 |= (unsigned)(reinterpret_cast<const unsigned short*>(src)[1]) << 16;
          if (kk + 2 < kEnd) u1 |= (unsigned)(reinterpret_cast<const unsigned short*>(src)[2]);
          if (kk + 3 < kEnd) u1 |= (unsigned)(reinterpret_cast<const unsigned short*>(src)[3]) << 16;
          if (kk + 4 < kEnd) u2 |= (unsigned)(reinterpret_cast<const unsigned short*>(src)[4]);
          if (kk + 5 < kEnd) u2 |= (unsigned)(reinterpret_cast<const unsigned short*>(src)[5]) << 16;
          if (kk + 6 < kEnd) u3 |= (unsigned)(reinterpret_cast<const unsigned short*>(src)[6]);
          wb = {u0,u1,u2,u3};
        }
      }
      unsigned short* dst = sw + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
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
        #pragma unroll
        for (int t2 = 0; t2 < 8; ++t2) dst[t2 * BNt] = 0;
      }
    }

    __syncthreads();

    for (int k = 0; k < BK; k += 16) {
      const int basek = k + yMF * 4;
      #pragma unroll
      for (int im = 0; im < nIterM; ++im) {
        const int row = yTile * WM + im * 16 + xMF;
        bf16x4 av = {sx[(basek + 0) * BM + row],
                     sx[(basek + 1) * BM + row],
                     sx[(basek + 2) * BM + row],
                     sx[(basek + 3) * BM + row]};
        #pragma unroll
        for (int g = 0; g < BN_AGG; ++g) {
          #pragma unroll
          for (int in = 0; in < nIterN; ++in) {
            const int c = xTile * WN + in * 16 + xMF;
            bf16x4 bv = {sw[(basek + 0) * BNt + c + g * BN],
                         sw[(basek + 1) * BNt + c + g * BN],
                         sw[(basek + 2) * BNt + c + g * BN],
                         sw[(basek + 3) * BNt + c + g * BN]};
            d_acc[g][im][in] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, d_acc[g][im][in], 0, 0, 0);
          }
        }
      }
    }

    __syncthreads();
  }

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  for (int im = 0; im < nIterM; ++im) {
    for (int i = 0; i < 4; ++i) {
      const int row = yTile * WM + im * 16 + (yD + i);
      if (row >= tcnt) continue;
      const int pos = base + row;
      const float w = w_by_bucket[pos];
      for (int g = 0; g < BN_AGG; ++g) {
        for (int in = 0; in < nIterN; ++in) {
          const int col = blockIdx.x * (BN * BN_AGG) + xTile * WN + in * 16 + xD + g * BN;
          if (col >= H) continue;
          const float bia = B2 ? __bfloat162float(B2[b2_e + col]) : 0.f;
          float* dst = z_partial + (size_t)pos * (size_t)H + col;
          if (splits == 1) {
            *dst = (d_acc[g][im][in][i] + bia) * w;
          } else {
            atomicAdd(dst, d_acc[g][im][in][i] * w);
            if (blockIdx.z == 0) atomicAdd(dst, bia * w);
          }
        }
      }
    }
  }
}


static inline void launch_mlp1_swiglu_bf16_bucketed_outbf16(
  __hip_bfloat16* gate_up_bf16, const __hip_bfloat16* a_in,
  const __hip_bfloat16* w1_layer, const __hip_bfloat16* b1_layer,
  const int* offsets, const int* blk_offs, int H, int I, int E,
  int total_blocks, float swiglu_limit, hipStream_t stream
) {
  PROFILE_FUNCTION();
  constexpr int BM = MATMUL_MLP1_BLOCK_ROWS;
  constexpr int BN = MATMUL_MLP1_BLOCK_COLS;
  constexpr int BK = MATMUL_MLP1_BLOCK_DEPTH;
  constexpr int WM = MATMUL_MLP1_WARP_TILE_M;
  constexpr int WN = MATMUL_MLP1_WARP_TILE_N;
  constexpr int WARPS = MATMUL_MLP1_WAVES_PER_BLOCK;
  constexpr int CTA = NUM_CTA_MLP1;
  constexpr int THREADS = 64 * WARPS;
  constexpr int BNt = BN * 2;
  dim3 block(THREADS);
  dim3 grid((I + BNt - 1) / BNt, total_blocks);
  size_t shmem = (size_t)BK * ((size_t)BM + (size_t)BNt + (size_t)BNt) * sizeof(unsigned short);
  mlp1_swiglu_bf16_bucketed_kernel_outbf16_tuned<BM,BN,BK,WM,WN,2,CTA>
    <<<grid, block, shmem, stream>>>(gate_up_bf16, a_in, w1_layer, b1_layer, offsets, blk_offs, H, I, E, swiglu_limit);
    //HIP_CHECK(hipDeviceSynchronize());
}

static inline void launch_mlp2_partial_bf16_bucketed_frombf16(
  float* z_partial, const __hip_bfloat16* gate_up_bf16,
  const __hip_bfloat16* w2_layer, const __hip_bfloat16* b2_layer,
  const float* w_by_bucket, const int* offsets, const int* blk_offs,
  const int* tok_idx, int I, int H, int E, int total_blocks, hipStream_t stream
) {
  PROFILE_FUNCTION();
  constexpr int BM = MATMUL_MLP2_BLOCK_ROWS;
  constexpr int BN = MATMUL_MLP2_BLOCK_COLS;
  constexpr int BK = MATMUL_MLP2_BLOCK_DEPTH;
  constexpr int WM = MATMUL_MLP2_WARP_TILE_M;
  constexpr int WN = MATMUL_MLP2_WARP_TILE_N;
  constexpr int WARPS = MATMUL_MLP2_WAVES_PER_BLOCK;
  constexpr int THREADS = 64 * WARPS;
  constexpr int BNt = BN * 3;

  int dev = 0, cu = 104;
  HIP_CHECK(hipGetDevice(&dev));
  HIP_CHECK(hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev));
  if (cu <= 0) cu = 104;

  const int gx = (H + BNt - 1) / BNt;
  const int gy = total_blocks;
  const int grid_xy = max(1, gx) * max(1, gy);
  const int target_cta = cu * 4;
  int splits = (grid_xy >= target_cta) ? 1 : (target_cta + grid_xy - 1) / grid_xy;
  splits = min(8, max(1, splits));
  const int max_splits_by_K = max(1, (I + BK - 1) / BK);
  splits = min(splits, max_splits_by_K);

  if (splits > 1) {
    int total_pairs = 0;
    HIP_CHECK(hipMemcpyAsync(&total_pairs, offsets + E, sizeof(int), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemsetAsync(z_partial, 0, (size_t)total_pairs * (size_t)H * sizeof(float), stream));
  }

  dim3 block(THREADS);
  dim3 grid(max(1, gx), max(1, gy), splits);
  size_t shmem = (size_t)BK * ((size_t)BM + (size_t)BNt) * sizeof(unsigned short);

  mlp2_partial_bf16_bucketed_splitk_kernel_inbf16_tuned<BM,BN,BK,WM,WN,3,NUM_CTA_MLP2>
    <<<grid, block, shmem, stream>>>(z_partial, gate_up_bf16, w2_layer, b2_layer, w_by_bucket,
                                     offsets, blk_offs, tok_idx, I, H, E, splits);
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

#ifndef MATMUL_QKV_BLOCK_ROWS
#define MATMUL_QKV_BLOCK_ROWS 128
#endif
#ifndef MATMUL_QKV_BLOCK_COLS
#define MATMUL_QKV_BLOCK_COLS 128
#endif
#ifndef MATMUL_QKV_BLOCK_DEPTH
#define MATMUL_QKV_BLOCK_DEPTH 32
#endif
#ifndef MATMUL_QKV_WARP_TILE_M
#define MATMUL_QKV_WARP_TILE_M 32
#endif
#ifndef MATMUL_QKV_WARP_TILE_N
#define MATMUL_QKV_WARP_TILE_N 32
#endif
#ifndef MATMUL_QKV_WAVES_PER_BLOCK
#define MATMUL_QKV_WAVES_PER_BLOCK 16
#endif
#ifndef QKV_LDS_PAD
#define QKV_LDS_PAD 8
#endif

template<
  int BM, int BN, int BK,
  int WM, int WN,
  int WARPS_PER_BLOCK,
  int PAD = QKV_LDS_PAD,
  int THREADS = 64 * WARPS_PER_BLOCK
>
__global__ void __launch_bounds__(THREADS)
matmul_qkv_fused_bf16_kernel_opt(
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ w,
    const __hip_bfloat16* __restrict__ b,
    int n, int q_len, int k_len, int v_len, int batch_size) {
  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;
  static_assert(BK % MFMA_K == 0, "");
  static_assert(WM % MFMA_M == 0 && WN % MFMA_N == 0, "");
  static_assert(BM % WM == 0 && BN % WN == 0, "");
  static_assert(WARPS_PER_BLOCK == (BM/WM)*(BN/WN), "");

  using CDfloat = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  using Afloat  = __attribute__((__vector_size__(2 * sizeof(float)))) unsigned short;
  using Bfloat  = __attribute__((__vector_size__(2 * sizeof(float)))) unsigned short;

  const int M = batch_size;
  const int N = q_len + k_len + v_len;
  const int K = n;

  const int wave = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int xTile = wave % (BM / WM);
  const int yTile = wave / (BM / WM);
  const int xMF = lane & 15;
  const int yMF = lane >> 4;

  const int loadX_x = threadIdx.x % (BK / 8);
  const int loadX_y = threadIdx.x / (BK / 8);
  const int loadW_x = threadIdx.x % (BK / 8);
  const int loadW_y = threadIdx.x / (BK / 8);
  constexpr int strideX = THREADS / (BK / 8);
  constexpr int strideW = THREADS / (BK / 8);

  constexpr int BM_PAD = BM + PAD;
  constexpr int BN_PAD = BN + PAD;

  __shared__ unsigned short sx[2][BK][BM_PAD];
  __shared__ unsigned short sw[2][BK][BN_PAD];

  CDfloat acc[(WM / MFMA_M) * (WN / MFMA_N)];
  #pragma unroll
  for (int i = 0; i < (WM / MFMA_M) * (WN / MFMA_N); ++i) acc[i] = CDfloat{0,0,0,0};

  const int rowBase = blockIdx.y * BM;
  const int colBase = blockIdx.x * BN;

  auto prefetch = [&](int bk, unsigned short (*px)[BM_PAD], unsigned short (*pw)[BN_PAD]) {
    for (int off = 0; off < BM; off += strideX) {
      int r = loadX_y + off;
      if (r >= BM) break;
      int rg = rowBase + r;
      int kk = bk + 8 * loadX_x;
      uint4 vx = make_uint4(0,0,0,0);
      if (rg < M && kk < K) {
        if (kk + 7 < K) vx = *reinterpret_cast<const uint4*>(x + (size_t)rg * K + kk);
        else {
          const unsigned short* p = reinterpret_cast<const unsigned short*>(x + (size_t)rg * K + kk);
          unsigned u0=0,u1=0,u2=0,u3=0;
          if (kk + 0 < K) u0 |= (unsigned)p[0];
          if (kk + 1 < K) u0 |= (unsigned)p[1] << 16;
          if (kk + 2 < K) u1 |= (unsigned)p[2];
          if (kk + 3 < K) u1 |= (unsigned)p[3] << 16;
          if (kk + 4 < K) u2 |= (unsigned)p[4];
          if (kk + 5 < K) u2 |= (unsigned)p[5] << 16;
          if (kk + 6 < K) u3 |= (unsigned)p[6];
          vx = {u0,u1,u2,u3};
        }
      }
      px[8 * loadX_x + 0][r] = (unsigned short)(vx.x & 0xFFFF);
      px[8 * loadX_x + 1][r] = (unsigned short)(vx.x >> 16);
      px[8 * loadX_x + 2][r] = (unsigned short)(vx.y & 0xFFFF);
      px[8 * loadX_x + 3][r] = (unsigned short)(vx.y >> 16);
      px[8 * loadX_x + 4][r] = (unsigned short)(vx.z & 0xFFFF);
      px[8 * loadX_x + 5][r] = (unsigned short)(vx.z >> 16);
      px[8 * loadX_x + 6][r] = (unsigned short)(vx.w & 0xFFFF);
      px[8 * loadX_x + 7][r] = (unsigned short)(vx.w >> 16);
    }

    for (int off = 0; off < BN; off += strideW) {
      int c = loadW_y + off;
      if (c >= BN) break;
      int cg = colBase + c;
      int kk = bk + 8 * loadW_x;
      uint4 vw = make_uint4(0,0,0,0);
      if (cg < N && kk < K) {

        if (kk + 7 < K) vw = *reinterpret_cast<const uint4*>(w + (size_t)cg * K + kk);
        else {
          const unsigned short* p = reinterpret_cast<const unsigned short*>(w + (size_t)cg * K + kk);
          unsigned u0=0,u1=0,u2=0,u3=0;
          if (kk + 0 < K) u0 |= (unsigned)p[0];
          if (kk + 1 < K) u0 |= (unsigned)p[1] << 16;
          if (kk + 2 < K) u1 |= (unsigned)p[2];
          if (kk + 3 < K) u1 |= (unsigned)p[3] << 16;
          if (kk + 4 < K) u2 |= (unsigned)p[4];
          if (kk + 5 < K) u2 |= (unsigned)p[5] << 16;
          if (kk + 6 < K) u3 |= (unsigned)p[6];
          vw = {u0,u1,u2,u3};
        }

      }
      pw[8 * loadW_x + 0][c] = (unsigned short)(vw.x & 0xFFFF);
      pw[8 * loadW_x + 1][c] = (unsigned short)(vw.x >> 16);
      pw[8 * loadW_x + 2][c] = (unsigned short)(vw.y & 0xFFFF);
      pw[8 * loadW_x + 3][c] = (unsigned short)(vw.y >> 16);
      pw[8 * loadW_x + 4][c] = (unsigned short)(vw.z & 0xFFFF);
      pw[8 * loadW_x + 5][c] = (unsigned short)(vw.z >> 16);
      pw[8 * loadW_x + 6][c] = (unsigned short)(vw.w & 0xFFFF);
      pw[8 * loadW_x + 7][c] = (unsigned short)(vw.w >> 16);
    }
  };

  if (K > 0) prefetch(0, sx[0], sw[0]);
  __syncthreads();
  int ping = 0;

  for (int bk = 0; bk < K; bk += BK) {
    unsigned short (*px)[BM_PAD] = sx[ping];
    unsigned short (*pw)[BN_PAD] = sw[ping];
    int nxt = bk + BK;
    if (nxt < K) prefetch(nxt, sx[ping ^ 1], sw[ping ^ 1]);

    for (int k = 0; k < BK; k += MFMA_K) {
      for (int im = 0; im < (WM / MFMA_M); ++im) {
        for (int in = 0; in < (WN / MFMA_N); ++in) {
          const int row = yTile * WM + MFMA_M * im + xMF;
          const int col = xTile * WN + MFMA_N * in + (lane & 15);
          Afloat av;
          Bfloat bv;
          const int basek = k + 4 * (lane >> 4);
          av[0] = px[basek + 0][row];
          av[1] = px[basek + 1][row];
          av[2] = px[basek + 2][row];
          av[3] = px[basek + 3][row];
          bv[0] = pw[basek + 0][col];
          bv[1] = pw[basek + 1][col];
          bv[2] = pw[basek + 2][col];
          bv[3] = pw[basek + 3][col];
          CDfloat acc_vec = acc[im * (WN / MFMA_N) + in];
          acc_vec = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc_vec, 0, 0, 0);
          acc[im * (WN / MFMA_N) + in] = acc_vec;
        }
      }
    }
    __syncthreads();
    ping ^= 1;
  }

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  for (int im = 0; im < (WM / MFMA_M); ++im) {
    for (int in = 0; in < (WN / MFMA_N); ++in) {
      int idx = im * (WN / MFMA_N) + in;
      for (int i = 0; i < 4; ++i) {
        int CDi = 4 * (lane >> 4) + (i & 3);
        int CDj = lane & 15;
        int gx = blockIdx.x * BN + WN * xTile + in * MFMA_N + CDj;
        int gy = blockIdx.y * BM + WM * yTile + im * MFMA_M + CDi;
        if (gx < N && gy < M) {
          float v = acc[idx][i];
          if (b) v += __bfloat162float(b[gx]);
          if (gx < q_len) {
            q_out[(size_t)gy * q_len + gx] = v;
          } else if (gx < q_len + k_len) {
            int off = gx - q_len;
            k_out[(size_t)gy * k_len + off] = v;
          } else {
            int off = gx - q_len - k_len;
            v_out[(size_t)gy * v_len + off] = v;
          }
        }
      }
    }
  }
}

static inline void getp_matmul_qkv_fused_bf16(
    float *q, float *k, float *v,
    const __hip_bfloat16 *x_bf16,
    const __hip_bfloat16 *w_qkv_bf16,
    const __hip_bfloat16 *b_qkv_bf16,
    int n, int head_dim, int n_attn_heads, int n_kv_heads, int batch_size,
    hipStream_t stream) {
      PROFILE_FUNCTION();
  const int q_len = head_dim * n_attn_heads;
  const int k_len = head_dim * n_kv_heads;
  const int v_len = head_dim * n_kv_heads;
  const int M = batch_size;
  const int N = q_len + k_len + v_len;
  constexpr int BM = MATMUL_QKV_BLOCK_ROWS;
  constexpr int BN = MATMUL_QKV_BLOCK_COLS;
  constexpr int BK = MATMUL_QKV_BLOCK_DEPTH;
  constexpr int WM = MATMUL_QKV_WARP_TILE_M;
  constexpr int WN = MATMUL_QKV_WARP_TILE_N;
  constexpr int WARPS = MATMUL_QKV_WAVES_PER_BLOCK;
  dim3 block(64 * WARPS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  matmul_qkv_fused_bf16_kernel_opt<BM,BN,BK,WM,WN,WARPS>
      <<<grid, block, 0, stream>>>(q, k, v, x_bf16, w_qkv_bf16, b_qkv_bf16,
                                   n, q_len, k_len, v_len, batch_size);
  //HIP_CHECK(hipDeviceSynchronize());
}

__global__ void __launch_bounds__(MATMUL_ROUTER_THREADS)
    matmul_kernel_nosplit(float *__restrict__ xout, const float *__restrict__ x,
                          const __hip_bfloat16 *__restrict__ w,
                          const __hip_bfloat16 *__restrict__ b, int M, int N,
                          int K) {
  constexpr int BM = MATMUL_ROUTER_BLOCK_ROWS;
  constexpr int BN = MATMUL_ROUTER_BLOCK_COLS;
  constexpr int BK = MATMUL_ROUTER_BLOCK_DEPTH;
  constexpr int WM = MATMUL_ROUTER_WARP_TILE_M;
  constexpr int WN = MATMUL_ROUTER_WARP_TILE_N;
  constexpr int THREADS = MATMUL_ROUTER_THREADS;
  constexpr int WARPS_PER_BLOCK = THREADS / 64;

  static_assert(BK % 16 == 0, "BK must be multiple of 16");
  static_assert(WM % 16 == 0 && WN % 16 == 0, "WM/WN must be multiples of 16");
  static_assert(BM % WM == 0 && BN % WN == 0, "BM/BM or BN/WN mismatch");
  static_assert(WARPS_PER_BLOCK == (BM / WM) * (BN / WN),
                "Wave decomposition mismatch");
  static_assert(THREADS % (BK / 4) == 0, "Threads must divide BK/4 loads");
  static_assert(THREADS % (BK / 8) == 0, "Threads must divide BK/8 loads");
  static_assert(GETP_BN_AGG > 0, "GETP_BN_AGG must be > 0");

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
  constexpr int strideX = THREADS / (BK / 4);
  constexpr int strideW = THREADS / (BK / 8);

  extern __shared__ unsigned short sm[];
  unsigned short *sx0 = sm;
  unsigned short *sw0 = sx0 + (size_t)BK * BM;
  unsigned short *sx1 = sw0 + (size_t)BK * BNt;
  unsigned short *sw1 = sx1 + (size_t)BK * BM;

  f32x4 d_acc[GETP_BN_AGG];
  #pragma unroll
  for (int g = 0; g < GETP_BN_AGG; ++g) d_acc[g] = {0, 0, 0, 0};

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
      #pragma unroll
      for (int g = 0; g < GETP_BN_AGG; ++g) {
        bf16x4 bv = {sw[(basek + 0) * BNt + c + g * BN],
                     sw[(basek + 1) * BNt + c + g * BN],
                     sw[(basek + 2) * BNt + c + g * BN],
                     sw[(basek + 3) * BNt + c + g * BN]};
        d_acc[g] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, d_acc[g],
                                                             0, 0, 0);
      }
    }
    __syncthreads();
    ping = !ping;
  }

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  for (int i = 0; i < 4; ++i) {
    const int row = rowBase + yTile * WM + (yD + i);
    if (row >= M) continue;
    #pragma unroll
    for (int g = 0; g < GETP_BN_AGG; ++g) {
      const int col = colBase + xTile * WN + xD + g * BN;
      if (col >= N) continue;
      float v = d_acc[g][i];
      if (b) v += __bfloat162float(b[col]);
      xout[(size_t)row * N + col] = v;
    }
  }
}

template<
  int BM, int BN, int BK,
  int WM, int WN,
  int BN_AGG,
  int WARPS_PER_BLOCK,
  int THREADS = 64 * WARPS_PER_BLOCK
>
__global__ void __launch_bounds__(THREADS)
matmul_kernel_splitk_store_tuned(
    float *__restrict__ partial, const float *__restrict__ x,
    const __hip_bfloat16 *__restrict__ w, int M, int N, int K, int splits) {

  static_assert(BK % 16 == 0, "BK%16");
  static_assert(WM % 16 == 0 && WN % 16 == 0, "WM/WN%16");
  static_assert(BM % WM == 0 && BN % WN == 0, "BM%WM,BN%WN");
  static_assert(WARPS_PER_BLOCK == (BM/WM)*(BN/WN), "warps cfg");

  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;
  constexpr int BNt = BN * BN_AGG;

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
  constexpr int strideX = THREADS / (BK / 4);
  constexpr int strideW = THREADS / (BK / 8);

  extern __shared__ unsigned short sm[];
  unsigned short *sx0 = sm;
  unsigned short *sw0 = sx0 + (size_t)BK * BM;
  unsigned short *sx1 = sw0 + (size_t)BK * BNt;
  unsigned short *sw1 = sx1 + (size_t)BK * BM;

  f32x4 d_acc[BN_AGG] = {};
  #pragma unroll
  for (int g = 0; g < BN_AGG; ++g) d_acc[g] = f32x4{0,0,0,0};

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
          float4 v = *reinterpret_cast<const float4 *>(x + (size_t)row * K + kk);
          sx[(4 * loadX_x + 0) * BM + (loadX_y + off)] = f32_to_bf16bits(v.x);
          sx[(4 * loadX_x + 1) * BM + (loadX_y + off)] = f32_to_bf16bits(v.y);
          sx[(4 * loadX_x + 2) * BM + (loadX_y + off)] = f32_to_bf16bits(v.z);
          sx[(4 * loadX_x + 3) * BM + (loadX_y + off)] = f32_to_bf16bits(v.w);
        } else {
          for (int t = 0; t < 4; ++t) {
            int kx = kk + t;
            unsigned short val = (kx < kEnd) ? f32_to_bf16bits(x[(size_t)row * K + kx]) : 0;
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
          for (int t2 = 0; t2 < 8; ++t2) {
            int kx = kk + t2;
            unsigned short val = (kx < kEnd) ? reinterpret_cast<const unsigned short *>(w + (size_t)col * K + kx)[0] : 0;
            dst[t2 * BNt] = val;
          }
        }
      } else {
        for (int t2 = 0; t2 < 8; ++t2) dst[t2 * BNt] = 0;
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

    for (int k = 0; k < BK; k += MFMA_K) {
      const int basek = k + yMF * 4;
      const int row = yTile * WM + xMF;
      const int c = xTile * WN + xMF;
      bf16x4 av = {sx[(basek + 0) * BM + row], sx[(basek + 1) * BM + row], sx[(basek + 2) * BM + row], sx[(basek + 3) * BM + row]};
      #pragma unroll
      for (int g = 0; g < BN_AGG; ++g) {
        bf16x4 bv = {sw[(basek + 0) * BNt + c + g * BN], sw[(basek + 1) * BNt + c + g * BN], sw[(basek + 2) * BNt + c + g * BN], sw[(basek + 3) * BNt + c + g * BN]};
        d_acc[g] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, d_acc[g], 0, 0, 0);
      }
    }
    __syncthreads();
    ping = !ping;
  }

  const size_t slice_stride = (size_t)M * N;
  const size_t base_out = (size_t)blockIdx.z * slice_stride;

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  for (int i = 0; i < 4; ++i) {
    const int row = rowBase + yTile * WM + (yD + i);
    if (row >= M) continue;
    #pragma unroll
    for (int g = 0; g < BN_AGG; ++g) {
      const int col = colBase + xTile * WN + xD + g * BN;
      if (col < N) partial[base_out + (size_t)row * N + col] = d_acc[g][i];
    }
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
  const int *__restrict__ mask_on,
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

  if (!mask_on[b]) return;

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
  const int *__restrict__ mask_on,
  const float *__restrict__ attn_sinks, int head_dim, int n_attn_heads,
  int n_kv_heads, int pos, int batch_size, int kv_dim, int kv_mul, 
  int cache_tcap, float inv_sqrt_d
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

  if (!mask_on[b]) return;
  
  const int loadKVIdx_y = threadIdx.x / (HEAD_DIM);
  const int loadKVIdx_x = threadIdx.x % (HEAD_DIM);
  constexpr int strideKV = BLOCK_SIZE / (HEAD_DIM);

  const int t_start = MAX(0, pos + 1 - cache_tcap);
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
        } else {
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

#ifndef FLASH_DECODE_TILE_T
#define FLASH_DECODE_TILE_T 16
#endif
static inline void getp_flash_attn_decode_bf16(
    float *tb, const __hip_bfloat16 *key_cache_layer,
    const __hip_bfloat16 *value_cache_layer, const float *q, const int *mask_on,
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
    flash_attn_decode_even_layer_bf16_matrix_core_kernel<FLASH_DECODE_TILE_T><<<grid, block, 0, stream>>>
      (tb, key_cache_layer, 
       value_cache_layer, q, 
       mask_on, 
       attn_sinks_layer, head_dim, n_attn_heads, 
       n_kv_heads, pos, batch_size, kv_dim, kv_mul, 
       sliding_window, cache_tcap, invsd);
  }
  else {
    dim3 block(128);
    dim3 grid(n_kv_heads, batch_size);
    flash_attn_decode_odd_layer_bf16_matrix_core_kernel<FLASH_DECODE_TILE_T><<<grid, block, 0, stream>>>
      (tb, key_cache_layer, 
      value_cache_layer, q, 
      mask_on,
      attn_sinks_layer, head_dim, n_attn_heads, 
      n_kv_heads, pos, batch_size, kv_dim, kv_mul, 
      cache_tcap, invsd);
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

  // dim3 grid(gx, gy, splits);
  // matmul_kernel_splitk_store<<<grid, block, shmem, stream>>>(
  //     partial, x, reinterpret_cast<const __hip_bfloat16 *>(w), M, N, K, splits);
  {
    dim3 block(256);
    dim3 grid(gx, gy, splits);
    size_t shmem = (size_t)2 * (size_t)BK * ((size_t)BM + (size_t)BNt) * sizeof(unsigned short);
    matmul_kernel_splitk_store_tuned<
        MATMUL_ROUTER_BLOCK_ROWS, MATMUL_ROUTER_BLOCK_COLS, MATMUL_ROUTER_BLOCK_DEPTH,
        MATMUL_ROUTER_WARP_TILE_M, MATMUL_ROUTER_WARP_TILE_N,
        GETP_BN_AGG, MATMUL_ROUTER_WAVES_PER_BLOCK>
      <<<grid, block, shmem, stream>>>(partial, x, w, M, N, K, splits);
  }
  

  const int total = M * N;
  dim3 rBlock(256);
  dim3 rGrid((total + rBlock.x - 1) / rBlock.x);
  reduce_splitk_with_bias<<<rGrid, rBlock, 0, stream>>>(
      xout, partial, reinterpret_cast<const __hip_bfloat16 *>(b), M, N, splits);

  HIP_CHECK(hipFreeAsync(partial, stream));
  //HIP_CHECK(hipDeviceSynchronize());
}

#ifndef MATMUL_ATTN_O_BLOCK_ROWS
#define MATMUL_ATTN_O_BLOCK_ROWS 128
#endif
#ifndef MATMUL_ATTN_O_BLOCK_COLS
#define MATMUL_ATTN_O_BLOCK_COLS 128
#endif
#ifndef MATMUL_ATTN_O_BLOCK_DEPTH
#define MATMUL_ATTN_O_BLOCK_DEPTH 16
#endif
#ifndef MATMUL_ATTN_O_WARP_TILE_M
#define MATMUL_ATTN_O_WARP_TILE_M 64
#endif
#ifndef MATMUL_ATTN_O_WARP_TILE_N
#define MATMUL_ATTN_O_WARP_TILE_N 64
#endif
#ifndef MATMUL_ATTN_O_WAVES_PER_BLOCK
#define MATMUL_ATTN_O_WAVES_PER_BLOCK 4
#endif

template<
  int BM, int BN, int BK,
  int WM, int WN,
  int WARPS_PER_BLOCK,
  int THREADS = 64 * WARPS_PER_BLOCK
>
__global__ void __launch_bounds__(THREADS)
matmul_attn_o_bf16_kernel_tuned(
    float *xout, const __hip_bfloat16 *__restrict__ x,
    const __hip_bfloat16 *__restrict__ w,
    const __hip_bfloat16 *__restrict__ b,
    int M, int N, int K) {

  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;
  static_assert(BK % 16 == 0, "BK%16");
  static_assert(WM % 16 == 0 && WN % 16 == 0, "WM/WN%16");
  static_assert(BM % WM == 0 && BN % WN == 0, "BM%WM,BN%WN");
  static_assert(WARPS_PER_BLOCK == (BM/WM)*(BN/WN), "warps cfg");

  using CDfloat = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  using Afloat  = __attribute__((__vector_size__(2 * sizeof(float)))) unsigned short;
  using Bfloat  = __attribute__((__vector_size__(2 * sizeof(float)))) unsigned short;

  const int waveIdx = threadIdx.x >> 6;
  const int laneIdx = threadIdx.x & 63;
  const int xInBlockTile = waveIdx % (BM / WM);
  const int yInBlockTile = waveIdx / (BM / WM);

  const int Ai = laneIdx & 15;
  const int Ak = 4 * (laneIdx >> 4);
  const int Bj = laneIdx & 15;
  const int Bk = 4 * (laneIdx >> 4);

  constexpr int nIterWaveM = WM / MFMA_M;
  constexpr int nIterWaveN = WN / MFMA_N;

  constexpr int nbLoadsX = BM * (BK / 4) / THREADS;
  constexpr int nbLoadsW = BN * (BK / 8) / THREADS;
  const int loadXIdx_y = threadIdx.x / (BK / 4);
  const int loadXIdx_x = threadIdx.x % (BK / 4);
  const int loadWIdx_y = threadIdx.x / (BK / 8);
  const int loadWIdx_x = threadIdx.x % (BK / 8);
  constexpr int strideX = THREADS / (BK / 4);
  constexpr int strideW = THREADS / (BK / 8);

  __shared__ unsigned short xs[BK][BM];
  __shared__ unsigned short ws[BK][BN];

  Afloat x_local;
  Bfloat w_local;
  CDfloat acc_local[nIterWaveM * nIterWaveN] = {0};

  for (int i = 0; i < nbLoadsX; ++i) {
    int off = i * strideX;
    int row_tile = loadXIdx_y + off;
    if (row_tile >= BM) break;
    const int rg = BM * blockIdx.y + row_tile;
    const int kk = 4 * loadXIdx_x;
    if (rg < M) {
      const __hip_bfloat16 *src = x + (size_t)rg * K + kk;
      if (kk + 3 < K) {
        uint2 v = *reinterpret_cast<const uint2 *>(src);
        xs[4 * loadXIdx_x + 0][row_tile] = (unsigned short)(v.x & 0xFFFF);
        xs[4 * loadXIdx_x + 1][row_tile] = (unsigned short)(v.x >> 16);
        xs[4 * loadXIdx_x + 2][row_tile] = (unsigned short)(v.y & 0xFFFF);
        xs[4 * loadXIdx_x + 3][row_tile] = (unsigned short)(v.y >> 16);
      } else {
        const unsigned short *p = reinterpret_cast<const unsigned short *>(src);
        xs[4 * loadXIdx_x + 0][row_tile] = (kk + 0 < K) ? p[0] : 0;
        xs[4 * loadXIdx_x + 1][row_tile] = (kk + 1 < K) ? p[1] : 0;
        xs[4 * loadXIdx_x + 2][row_tile] = (kk + 2 < K) ? p[2] : 0;
        xs[4 * loadXIdx_x + 3][row_tile] = (kk + 3 < K) ? p[3] : 0;
      }
    } else {
      xs[4 * loadXIdx_x + 0][row_tile] = 0;
      xs[4 * loadXIdx_x + 1][row_tile] = 0;
      xs[4 * loadXIdx_x + 2][row_tile] = 0;
      xs[4 * loadXIdx_x + 3][row_tile] = 0;
    }
  }
  for (int i = 0; i < nbLoadsW; ++i) {
    int off = i * strideW;
    int kk = 0 + 8 * loadWIdx_x;
    int cy = BN * blockIdx.x + loadWIdx_y + off;
    uint4 tmp = (kk < K && cy < N) ? *reinterpret_cast<const uint4 *>(&w[(size_t)cy * K + kk]) : make_uint4(0,0,0,0);
    ws[8 * loadWIdx_x + 0][loadWIdx_y + off] = (unsigned short)(tmp.x & 0xFFFF);
    ws[8 * loadWIdx_x + 1][loadWIdx_y + off] = (unsigned short)(tmp.x >> 16);
    ws[8 * loadWIdx_x + 2][loadWIdx_y + off] = (unsigned short)(tmp.y & 0xFFFF);
    ws[8 * loadWIdx_x + 3][loadWIdx_y + off] = (unsigned short)(tmp.y >> 16);
    ws[8 * loadWIdx_x + 4][loadWIdx_y + off] = (unsigned short)(tmp.z & 0xFFFF);
    ws[8 * loadWIdx_x + 5][loadWIdx_y + off] = (unsigned short)(tmp.z >> 16);
    ws[8 * loadWIdx_x + 6][loadWIdx_y + off] = (unsigned short)(tmp.w & 0xFFFF);
    ws[8 * loadWIdx_x + 7][loadWIdx_y + off] = (unsigned short)(tmp.w >> 16);
  }
  __syncthreads();

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    uint2 regX[nbLoadsX];
    uint4 regW[nbLoadsW];
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX;
        int row_tile = loadXIdx_y + off;
        const int rg = BM * blockIdx.y + row_tile;
        const int kk = bkIdx + BK + 4 * loadXIdx_x;
        if (row_tile < BM && rg < M) {
          const __hip_bfloat16 *src = x + (size_t)rg * K + kk;
          if (kk + 3 < K) regX[i] = *reinterpret_cast<const uint2 *>(src);
          else {
            const unsigned short *p = reinterpret_cast<const unsigned short *>(src);
            unsigned short v0 = 0, v1 = 0, v2 = 0, v3 = 0;
            if (kk + 0 < K) v0 = p[0];
            if (kk + 1 < K) v1 = p[1];
            if (kk + 2 < K) v2 = p[2];
            if (kk + 3 < K) v3 = p[3];
            regX[i].x = (unsigned)v0 | ((unsigned)v1 << 16);
            regX[i].y = (unsigned)v2 | ((unsigned)v3 << 16);
          }
        } else regX[i] = make_uint2(0,0);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;
        int kk = bkIdx + BK + 8 * loadWIdx_x;
        int cy = BN * blockIdx.x + loadWIdx_y + off;
        regW[i] = (kk < K && cy < N) ? *reinterpret_cast<const uint4 *>(&w[(size_t)cy * K + kk]) : make_uint4(0,0,0,0);
      }
    }

    for (int k = 0; k < BK; k += MFMA_K) {
      for (int im = 0; im < nIterWaveM; ++im) {
        for (int in = 0; in < nIterWaveN; ++in) {
          const int row = yInBlockTile * WM + MFMA_M * im + Ai;
          const int col = xInBlockTile * WN + MFMA_N * in + Bj;
          Afloat av;
          Bfloat bv;
          for (int t = 0; t < 4; ++t) { av[t] = xs[k + Ak + t][row]; bv[t] = ws[k + Bk + t][col]; }
          CDfloat acc = acc_local[im * nIterWaveN + in];
          acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc, 0, 0, 0);
          acc_local[im * nIterWaveN + in] = acc;
        }
      }
    }
    __syncthreads();

    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX;
        int row_tile = loadXIdx_y + off;
        if (row_tile >= BM) break;
        uint2 v = regX[i];
        xs[4 * loadXIdx_x + 0][row_tile] = (unsigned short)(v.x & 0xFFFF);
        xs[4 * loadXIdx_x + 1][row_tile] = (unsigned short)(v.x >> 16);
        xs[4 * loadXIdx_x + 2][row_tile] = (unsigned short)(v.y & 0xFFFF);
        xs[4 * loadXIdx_x + 3][row_tile] = (unsigned short)(v.y >> 16);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;
        ws[8 * loadWIdx_x + 0][loadWIdx_y + off] = (unsigned short)(regW[i].x & 0xFFFF);
        ws[8 * loadWIdx_x + 1][loadWIdx_y + off] = (unsigned short)(regW[i].x >> 16);
        ws[8 * loadWIdx_x + 2][loadWIdx_y + off] = (unsigned short)(regW[i].y & 0xFFFF);
        ws[8 * loadWIdx_x + 3][loadWIdx_y + off] = (unsigned short)(regW[i].y >> 16);
        ws[8 * loadWIdx_x + 4][loadWIdx_y + off] = (unsigned short)(regW[i].z & 0xFFFF);
        ws[8 * loadWIdx_x + 5][loadWIdx_y + off] = (unsigned short)(regW[i].z >> 16);
        ws[8 * loadWIdx_x + 6][loadWIdx_y + off] = (unsigned short)(regW[i].w & 0xFFFF);
        ws[8 * loadWIdx_x + 7][loadWIdx_y + off] = (unsigned short)(regW[i].w >> 16);
      }
    }
    __syncthreads();
  }

  for (int im = 0; im < nIterWaveM; ++im) {
    for (int in = 0; in < nIterWaveN; ++in) {
      int acc_index = im * nIterWaveN + in;
      for (int i = 0; i < 4; ++i) {
        int CDi = 4 * (laneIdx >> 4) + (i & 3);
        int CDj = laneIdx & 15;
        int gx = blockIdx.x * BN + WN * xInBlockTile + in * MFMA_N + CDj;
        int gy = blockIdx.y * BM + WM * yInBlockTile + im * MFMA_M + CDi;
        float v = acc_local[acc_index][i];
        if (gx < N && gy < M) {
          if (b) v += __bfloat162float(b[gx]);
          xout[(size_t)gy * N + gx] = v;
        }
      }
    }
  }
}

static inline void getp_matmul_attn_o_bf16(
    float *xout, const __hip_bfloat16 *x_bf16,
    const __hip_bfloat16 *w, const __hip_bfloat16 *b,
    int n, int d, int batch_size, hipStream_t stream) {
      PROFILE_FUNCTION();
  const int M = batch_size, N = d, K = n;

  constexpr int BM = MATMUL_ATTN_O_BLOCK_ROWS;
  constexpr int BN = MATMUL_ATTN_O_BLOCK_COLS;
  constexpr int BK = MATMUL_ATTN_O_BLOCK_DEPTH;
  constexpr int WM = MATMUL_ATTN_O_WARP_TILE_M;
  constexpr int WN = MATMUL_ATTN_O_WARP_TILE_N;
  constexpr int WARPS = MATMUL_ATTN_O_WAVES_PER_BLOCK;

  dim3 block(64 * WARPS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  matmul_attn_o_bf16_kernel_tuned<BM,BN,BK,WM,WN,WARPS>
      <<<grid, block, 0, stream>>>(xout, x_bf16, w, b, M, N, K);
  //HIP_CHECK(hipDeviceSynchronize());
}


__device__ inline unsigned long long pack_val_idx(float v, unsigned int idx) {
  unsigned int vb = __float_as_uint(v);
  return (static_cast<unsigned long long>(vb) << 32) | static_cast<unsigned long long>(idx);
}
__device__ inline float unpack_val(unsigned long long p) {
  unsigned int vb = static_cast<unsigned int>(p >> 32);
  return __uint_as_float(vb);
}
__device__ inline unsigned int unpack_idx(unsigned long long p) {
  return static_cast<unsigned int>(p & 0xffffffffu);
}
__device__ inline float atomicMaxFloat(float* addr, float val) {
  int* a = reinterpret_cast<int*>(addr);
  int old = *a;
  float oldf = __int_as_float(old);
  while (val > oldf) {
    int assumed = old;
    old = atomicCAS(a, assumed, __float_as_int(val));
    oldf = __int_as_float(old);
  }
  return oldf;
}
__device__ inline void atomic_update_max_pair(unsigned long long* addr, float v, unsigned int idx) {
  while (true) {
    unsigned long long old = *addr;
    float ov = unpack_val(old);
    unsigned int oi = unpack_idx(old);
    float thr = 1e-6f * fmaxf(fabsf(v), fabsf(ov));
    bool take = (v > ov + thr) || (fabsf(v - ov) <= thr && idx < oi);
    if (!take) break;
    if (atomicCAS(reinterpret_cast<unsigned long long*>(addr), old, pack_val_idx(v, idx)) == old) break;
  }
}
__global__ void init_argmax_pairs(unsigned long long* pairs, int B, int V) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B) pairs[i] = pack_val_idx(-INFINITY, static_cast<unsigned int>(V));
}
__global__ void extract_argmax_pairs(const unsigned long long* pairs, int* out, int B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < B) out[i] = static_cast<int>(unpack_idx(pairs[i]));
}

#ifndef MATMUL_LOGITS_BLOCK_ROWS
#define MATMUL_LOGITS_BLOCK_ROWS 128
#endif
#ifndef MATMUL_LOGITS_BLOCK_COLS
#define MATMUL_LOGITS_BLOCK_COLS 128
#endif
#ifndef MATMUL_LOGITS_BLOCK_DEPTH
#define MATMUL_LOGITS_BLOCK_DEPTH 16
#endif
#ifndef MATMUL_LOGITS_WARP_TILE_M
#define MATMUL_LOGITS_WARP_TILE_M 64
#endif
#ifndef MATMUL_LOGITS_WARP_TILE_N
#define MATMUL_LOGITS_WARP_TILE_N 64
#endif
#ifndef MATMUL_LOGITS_WAVES_PER_BLOCK
#define MATMUL_LOGITS_WAVES_PER_BLOCK 4
#endif

template<
  int BM, int BN, int BK,
  int WM, int WN,
  int WARPS_PER_BLOCK,
  int THREADS = 64 * WARPS_PER_BLOCK
>
__global__ void __launch_bounds__(THREADS)
matmul_logits_argmax_bf16_kernel_tuned(
    const float* __restrict__ x,
    const __hip_bfloat16* __restrict__ w,
    int M, int N, int K,
    unsigned long long* __restrict__ pairs) {

  constexpr int MFMA_M = 16;
  constexpr int MFMA_N = 16;
  constexpr int MFMA_K = 16;
  static_assert(BK % 16 == 0, "BK%16");
  static_assert(WM % 16 == 0 && WN % 16 == 0, "WM/WN%16");
  static_assert(BM % WM == 0 && BN % WN == 0, "BM%WM,BN%WN");
  static_assert(WARPS_PER_BLOCK == (BM/WM)*(BN/WN), "warps cfg");

  using CDfloat = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  using Afloat  = __attribute__((__vector_size__(2 * sizeof(float)))) unsigned short;
  using Bfloat  = __attribute__((__vector_size__(2 * sizeof(float)))) unsigned short;

  const int waveIdx = threadIdx.x >> 6;
  const int laneIdx = threadIdx.x & 63;
  const int xInBlockTile = waveIdx % (BM / WM);
  const int yInBlockTile = waveIdx / (BM / WM);

  const int Ai = laneIdx & 15;
  const int Ak = 4 * (laneIdx >> 4);
  const int Bj = laneIdx & 15;
  const int Bk = 4 * (laneIdx >> 4);

  constexpr int nIterWaveM = WM / MFMA_M;
  constexpr int nIterWaveN = WN / MFMA_N;

  constexpr int nbLoadsX = BM * (BK / 4) / THREADS;
  constexpr int nbLoadsW = BN * (BK / 8) / THREADS;
  const int loadXIdx_y = threadIdx.x / (BK / 4);
  const int loadXIdx_x = threadIdx.x % (BK / 4);
  const int loadWIdx_y = threadIdx.x / (BK / 8);
  const int loadWIdx_x = threadIdx.x % (BK / 8);
  constexpr int strideX = THREADS / (BK / 4);
  constexpr int strideW = THREADS / (BK / 8);

  __shared__ unsigned short xs[BK][BM];
  __shared__ unsigned short ws[BK][BN];
  __shared__ float smax[BM];
  __shared__ int sidx[BM];

  for (int i = threadIdx.x; i < BM; i += blockDim.x) {
    smax[i] = -INFINITY;
    sidx[i] = INT_MAX;
  }
  __syncthreads();

  Afloat x_local;
  Bfloat w_local;
  CDfloat acc_local[nIterWaveM * nIterWaveN] = {0};

  for (int i = 0; i < nbLoadsX; ++i) {
    int off = i * strideX;
    int ry = loadXIdx_y + off;
    if (ry >= BM) break;
    const int rg = BM * blockIdx.y + ry;
    const int kk = 4 * loadXIdx_x;
    float4 tmp = (kk < K && rg < M) ? *reinterpret_cast<const float4 *>(&x[(size_t)rg * K + kk]) : make_float4(0,0,0,0);
    xs[4 * loadXIdx_x + 0][ry] = f32_to_bf16bits(tmp.x);
    xs[4 * loadXIdx_x + 1][ry] = f32_to_bf16bits(tmp.y);
    xs[4 * loadXIdx_x + 2][ry] = f32_to_bf16bits(tmp.z);
    xs[4 * loadXIdx_x + 3][ry] = f32_to_bf16bits(tmp.w);
  }
  for (int i = 0; i < nbLoadsW; ++i) {
    int off = i * strideW;
    int kk = 0 + 8 * loadWIdx_x;
    int cy = BN * blockIdx.x + loadWIdx_y + off;
    uint4 t = (kk < K && cy < N) ? *reinterpret_cast<const uint4 *>(&w[(size_t)cy * K + kk]) : make_uint4(0,0,0,0);
    ws[8 * loadWIdx_x + 0][loadWIdx_y + off] = (unsigned short)(t.x & 0xFFFF);
    ws[8 * loadWIdx_x + 1][loadWIdx_y + off] = (unsigned short)(t.x >> 16);
    ws[8 * loadWIdx_x + 2][loadWIdx_y + off] = (unsigned short)(t.y & 0xFFFF);
    ws[8 * loadWIdx_x + 3][loadWIdx_y + off] = (unsigned short)(t.y >> 16);
    ws[8 * loadWIdx_x + 4][loadWIdx_y + off] = (unsigned short)(t.z & 0xFFFF);
    ws[8 * loadWIdx_x + 5][loadWIdx_y + off] = (unsigned short)(t.z >> 16);
    ws[8 * loadWIdx_x + 6][loadWIdx_y + off] = (unsigned short)(t.w & 0xFFFF);
    ws[8 * loadWIdx_x + 7][loadWIdx_y + off] = (unsigned short)(t.w >> 16);
  }
  __syncthreads();

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    float4 regX[nbLoadsX];
    uint4  regW[nbLoadsW];
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX;
        int ry = loadXIdx_y + off;
        int rg = BM * blockIdx.y + ry;
        int kk = bkIdx + BK + 4 * loadXIdx_x;
        regX[i] = (kk < K && rg < M) ? *reinterpret_cast<const float4 *>(&x[(size_t)rg * K + kk]) : make_float4(0,0,0,0);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;
        int kk = bkIdx + BK + 8 * loadWIdx_x;
        int cy = BN * blockIdx.x + loadWIdx_y + off;
        regW[i] = (kk < K && cy < N) ? *reinterpret_cast<const uint4 *>(&w[(size_t)cy * K + kk]) : make_uint4(0,0,0,0);
      }
    }
    for (int k = 0; k < BK; k += MFMA_K) {
      for (int im = 0; im < nIterWaveM; ++im) {
        for (int in = 0; in < nIterWaveN; ++in) {
          const int row = yInBlockTile * WM + MFMA_M * im + Ai;
          const int col = xInBlockTile * WN + MFMA_N * in + Bj;
          Afloat av; Bfloat bv;
          for (int t = 0; t < 4; ++t) { av[t] = xs[k + Ak + t][row]; bv[t] = ws[k + Bk + t][col]; }
          CDfloat acc = acc_local[im * nIterWaveN + in];
          acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc, 0, 0, 0);
          acc_local[im * nIterWaveN + in] = acc;
        }
      }
    }
    __syncthreads();
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX;
        int ry = loadXIdx_y + off;
        if (ry >= BM) break;
        float4 v = regX[i];
        xs[4 * loadXIdx_x + 0][ry] = f32_to_bf16bits(v.x);
        xs[4 * loadXIdx_x + 1][ry] = f32_to_bf16bits(v.y);
        xs[4 * loadXIdx_x + 2][ry] = f32_to_bf16bits(v.z);
        xs[4 * loadXIdx_x + 3][ry] = f32_to_bf16bits(v.w);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;
        ws[8 * loadWIdx_x + 0][loadWIdx_y + off] = (unsigned short)(regW[i].x & 0xFFFF);
        ws[8 * loadWIdx_x + 1][loadWIdx_y + off] = (unsigned short)(regW[i].x >> 16);
        ws[8 * loadWIdx_x + 2][loadWIdx_y + off] = (unsigned short)(regW[i].y & 0xFFFF);
        ws[8 * loadWIdx_x + 3][loadWIdx_y + off] = (unsigned short)(regW[i].y >> 16);
        ws[8 * loadWIdx_x + 4][loadWIdx_y + off] = (unsigned short)(regW[i].z & 0xFFFF);
        ws[8 * loadWIdx_x + 5][loadWIdx_y + off] = (unsigned short)(regW[i].z >> 16);
        ws[8 * loadWIdx_x + 6][loadWIdx_y + off] = (unsigned short)(regW[i].w & 0xFFFF);
        ws[8 * loadWIdx_x + 7][loadWIdx_y + off] = (unsigned short)(regW[i].w >> 16);
      }
    }
    __syncthreads();
  }

  for (int im = 0; im < nIterWaveM; ++im) {
    for (int in = 0; in < nIterWaveN; ++in) {
      int acc_index = im * nIterWaveN + in;
      for (int i = 0; i < 4; ++i) {
        int CDi = 4 * (laneIdx >> 4) + (i & 3);
        int CDj = laneIdx & 15;
        int gx = blockIdx.x * BN + WN * xInBlockTile + in * MFMA_N + CDj;
        int ly = WM * yInBlockTile + MFMA_M * im + CDi;
        int gy = blockIdx.y * BM + ly;
        float v = acc_local[acc_index][i];
        if (gx < N && gy < M) {
          float mv = atomicMaxFloat(&smax[ly], v);
        }
      }
    }
  }
  __syncthreads();
  for (int im = 0; im < nIterWaveM; ++im) {
    for (int in = 0; in < nIterWaveN; ++in) {
      int acc_index = im * nIterWaveN + in;
      for (int i = 0; i < 4; ++i) {
        int CDi = 4 * (laneIdx >> 4) + (i & 3);
        int CDj = laneIdx & 15;
        int gx = blockIdx.x * BN + WN * xInBlockTile + in * MFMA_N + CDj;
        int ly = WM * yInBlockTile + MFMA_M * im + CDi;
        int gy = blockIdx.y * BM + ly;
        float v = acc_local[acc_index][i];
        if (gx < N && gy < M) {
          float mv = smax[ly];
          float thr = 1e-6f * fmaxf(fabsf(v), fabsf(mv));
          if (fabsf(v - mv) <= thr) atomicMin(&sidx[ly], gx);
        }
      }
    }
  }
  __syncthreads();
  for (int r = threadIdx.x; r < BM; r += blockDim.x) {
    int gy = blockIdx.y * BM + r;
    if (gy < M) atomic_update_max_pair(&pairs[gy], smax[r], (unsigned)sidx[r]);
  }
}

static inline void getp_matmul_logits_argmax_bf16(
    const float* x, const __hip_bfloat16* w, int d, int n, int batch_size,
    int* out, hipStream_t stream) {
      PROFILE_FUNCTION();
  const int M = batch_size, N = n, K = d;

  constexpr int BM = MATMUL_LOGITS_BLOCK_ROWS;
  constexpr int BN = MATMUL_LOGITS_BLOCK_COLS;
  constexpr int BK = MATMUL_LOGITS_BLOCK_DEPTH;
  constexpr int WM = MATMUL_LOGITS_WARP_TILE_M;
  constexpr int WN = MATMUL_LOGITS_WARP_TILE_N;
  constexpr int WARPS = MATMUL_LOGITS_WAVES_PER_BLOCK;

  dim3 block(64 * WARPS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  unsigned long long* pairs = nullptr;
  HIP_CHECK(hipMalloc(&pairs, sizeof(unsigned long long) * M));
  dim3 gi((M + 255) / 256), bi(256);
  init_argmax_pairs<<<gi, bi, 0, stream>>>(pairs, M, N);

  matmul_logits_argmax_bf16_kernel_tuned<BM,BN,BK,WM,WN,WARPS>
      <<<grid, block, 0, stream>>>(x, w, M, N, K, pairs);

  extract_argmax_pairs<<<gi, bi, 0, stream>>>(pairs, out, M);
  HIP_CHECK(hipFree(pairs));
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
                       int pos, int *mask_on) {
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
  for (int i = 0; i < EXPERT_PARALLELISM; ++i) HIP_CHECK(hipEventCreate(&events_e_agg[i]));

  HIP_CHECK(hipMemcpyAsync(ext->mask_on, mask_on, sizeof(int) * BATCH_SIZE, 
                           hipMemcpyHostToDevice, compute_stream));
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

  const int even_layers = (p->n_layers + 1) / 2;
  const int even_tcap = (p->sliding_window > 0 ? p->sliding_window : 1);
  int odd_tcap = p->seq_len / 4;
  if (odd_tcap < 1) odd_tcap = 1;
  const size_t even_stride = (size_t)even_tcap;
  const size_t odd_stride = (size_t)odd_tcap;

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_attn_w + 1ll * l * hidden_dim,
                 BATCH_SIZE, hidden_dim, compute_stream);

    const bool is_even = (((int)l & 1) == 0);
    const int idx_half = ((int)l) >> 1;
    const int cache_tcap = is_even ? even_tcap : odd_tcap;
    const size_t layer_toff =
        is_even ? (size_t)idx_half * even_stride
                : (size_t)even_layers * even_stride +
                      (size_t)idx_half * odd_stride;
    const int tslot = pos % cache_tcap;
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

    float *k_step = dev_s->tb2;
    float *v_step = dev_s->tb2 + (size_t)BATCH_SIZE * (size_t)kv_dim;

    __hip_bfloat16 *dev_w_qkv =
        dev_w->w_qkv_bf16 +
        1ll * l * hidden_dim *
            (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    __hip_bfloat16 *dev_b_qkv =
        dev_w->b_qkv_bf16 +
        1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    tensor_fp32_to_bf16(ext->pre_qkv_bf16, dev_s->t,
                        BATCH_SIZE * hidden_dim, compute_stream);
    getp_matmul_qkv_fused_bf16(dev_s->q, k_step, v_step, ext->pre_qkv_bf16,
                               dev_w_qkv, dev_b_qkv, hidden_dim, head_dim,
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
          dev_s->tb, k_layer, v_layer, q_ptr, ext->mask_on, attn_sinks_layer, head_dim,
          p->n_attn_heads, p->n_kv_heads, pos, p->seq_len, p->sliding_window,
          (int)l, BATCH_SIZE, cache_tcap, compute_stream);
    }

    __hip_bfloat16 *dev_w_o =
        dev_w->w_o_bf16 + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    __hip_bfloat16 *dev_b_o = dev_w->b_o_bf16 + 1ll * l * hidden_dim;
    tensor_fp32_to_bf16(ext->attn_o_bf16, dev_s->tb,
                        BATCH_SIZE * head_dim * p->n_attn_heads,
                        compute_stream);
    getp_matmul_attn_o_bf16(dev_s->tb2, ext->attn_o_bf16, dev_w_o, dev_b_o,
                            head_dim * p->n_attn_heads, hidden_dim, BATCH_SIZE,
                            compute_stream);
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
    tensor_fp32_to_bf16(
        ext->ext_t_bf16 + (size_t)device_index * BATCH_SIZE * hidden_dim,
        ext->ext_t + (size_t)device_index * BATCH_SIZE * hidden_dim,
        BATCH_SIZE * hidden_dim, memory_stream);

    for (int delta = 1; delta < EXPERT_PARALLELISM; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      GPUWorker *peer_worker = &workers[i];
      int peer_device_index = peer_worker->device_index;
      RunStateExt *peer_ext = ext_get(peer_device_index);
      if (peer_device_index != device_index) {
        HIP_CHECK(hipMemcpyPeerAsync(
            peer_ext->ext_t_bf16 + (size_t)device_index * BATCH_SIZE * hidden_dim,
            peer_device_index,
            ext->ext_t_bf16 + (size_t)device_index * BATCH_SIZE * hidden_dim,
            device_index,
            sizeof(__hip_bfloat16) * (size_t)BATCH_SIZE * (size_t)hidden_dim,
            memory_stream));
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
            sizeof(int) * (size_t)BATCH_SIZE * (size_t)p->experts_per_token,
            memory_stream));
        HIP_CHECK(hipMemcpyPeerAsync(
            peer_ext->ext_topk_v + (size_t)device_index * BATCH_SIZE * p->experts_per_token,
            peer_device_index,
            ext->ext_topk_v + (size_t)device_index * BATCH_SIZE * p->experts_per_token,
            device_index,
            sizeof(float) * (size_t)BATCH_SIZE * (size_t)p->experts_per_token,
            memory_stream));
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

    // int total_blocks_mlp1 = build_moe_block_schedule(
    //   ext->e_offsets, experts_per_device, MATMUL_MLP1_BLOCK_ROWS,
    //   ext->blk_counts, ext->blk_offsets, compute_stream);
    int total_blocks_mlp = build_moe_block_schedule(
      ext->e_offsets, experts_per_device, MATMUL_MLP1_BLOCK_ROWS,
      ext->blk_counts, ext->blk_offsets, compute_stream);
  
  moe_scatter_acts_to_expert_frombf16(
      ext->a_in,
      ext->ext_t_bf16,
      ext->e_dev,
      total_pairs, hidden_dim, compute_stream);
  
  HIP_CHECK(hipMemsetAsync(
      ext->ext_e_agg, 0,
      (size_t)EXPERT_PARALLELISM * BATCH_SIZE * hidden_dim * sizeof(float),
      memory_stream));
  HIP_CHECK(hipEventRecord(event_memset, memory_stream));
  
  __hip_bfloat16 *w1_base = dev_w->w_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim * hidden_dim;
  __hip_bfloat16 *b1_base = dev_w->b_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim;
  
  launch_mlp1_swiglu_bf16_bucketed_outbf16(
      ext->gate_up_bf16, ext->a_in, w1_base, b1_base, ext->e_offsets,
      ext->blk_offsets, hidden_dim, p->intermediate_dim, experts_per_device,
      total_blocks_mlp, p->swiglu_limit, compute_stream);
  
  __hip_bfloat16 *w2_base = dev_w->w_mlp2 + 1ll * l * experts_per_device * hidden_dim * p->intermediate_dim;
  __hip_bfloat16 *b2_base = dev_w->b_mlp2 + 1ll * l * experts_per_device * hidden_dim;
  
  // int total_blocks_mlp2 = build_moe_block_schedule(
  //     ext->e_offsets, experts_per_device, MATMUL_MLP2_BLOCK_ROWS,
  //     ext->blk_counts, ext->blk_offsets, compute_stream);
  
  launch_mlp2_partial_bf16_bucketed_frombf16(
      ext->z_partial, ext->gate_up_bf16, w2_base, b2_base, ext->w_dev,
      ext->e_offsets, ext->blk_offsets, ext->e_dev, p->intermediate_dim,
      hidden_dim, experts_per_device, total_blocks_mlp, compute_stream);
  
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
  getp_matmul_logits_argmax_bf16(dev_x, dev_w->out_bf16,
    hidden_dim, p->vocab_size,
    BATCH_SIZE, dev_s->topk_i, compute_stream);

  int *next_host = (int *)malloc(sizeof(int) * BATCH_SIZE);
  HIP_CHECK(hipMemcpyAsync(next_host, dev_s->topk_i, sizeof(int) * BATCH_SIZE,
                           hipMemcpyDeviceToHost, compute_stream));
  HIP_CHECK(hipStreamSynchronize(compute_stream));
  return next_host;
}


int *getp_forward_20b(Transformer * /*transformer*/,
                      DeviceTransformer **dev_transformeres, GPUWorker *worker,
                      int token[], int pos, int *mask_on, int batch_size) {
  
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

  HIP_CHECK(hipMemcpy(ext->mask_on, mask_on, sizeof(int) * batch_size, hipMemcpyHostToDevice));
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

  const int even_layers = (p->n_layers + 1) / 2;
  const int even_tcap = (p->sliding_window > 0 ? p->sliding_window : 1);
  int odd_tcap = p->seq_len / 4;
  if (odd_tcap < 1) odd_tcap = 1;
  const size_t even_stride = (size_t)even_tcap;
  const size_t odd_stride = (size_t)odd_tcap;

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_attn_w + 1ll * l * hidden_dim,
                 batch_size, hidden_dim, nullptr);

    const bool is_even = (((int)l & 1) == 0);
    const int idx_half = ((int)l) >> 1;
    const int cache_tcap = is_even ? even_tcap : odd_tcap;
    const size_t layer_toff =
        is_even ? (size_t)idx_half * even_stride
                : (size_t)even_layers * even_stride +
                      (size_t)idx_half * odd_stride;
    const int tslot = pos % cache_tcap;
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

    // float *k_step = dev_s->qkv;
    // float *v_step = dev_s->qkv + (size_t)BATCH_SIZE * (size_t)kv_dim;
    float *k_step = dev_s->tb2;
    float *v_step = dev_s->tb2 + (size_t)BATCH_SIZE * (size_t)kv_dim;

    __hip_bfloat16 *dev_w_qkv =
        dev_w->w_qkv_bf16 +
        1ll * l * hidden_dim *
            (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    __hip_bfloat16 *dev_b_qkv =
        dev_w->b_qkv_bf16 +
        1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    tensor_fp32_to_bf16(ext->pre_qkv_bf16, dev_s->t,
                        batch_size * hidden_dim, nullptr);
    getp_matmul_qkv_fused_bf16(dev_s->q, k_step, v_step, ext->pre_qkv_bf16,
                               dev_w_qkv, dev_b_qkv, hidden_dim, head_dim,
                               p->n_attn_heads, p->n_kv_heads, batch_size,
                               nullptr);

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
          dev_s->tb, k_layer, v_layer, q_ptr, ext->mask_on, attn_sinks_layer, head_dim,
          p->n_attn_heads, p->n_kv_heads, pos, p->seq_len, p->sliding_window,
          (int)l, batch_size, cache_tcap, nullptr);
    }

    __hip_bfloat16 *dev_w_o =
        dev_w->w_o_bf16 + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    __hip_bfloat16 *dev_b_o = dev_w->b_o_bf16 + 1ll * l * hidden_dim;
    // getp_new_matmul(dev_s->tb2, dev_s->tb, dev_w_o, dev_b_o,
    //                     head_dim * p->n_attn_heads, hidden_dim,
    //                     batch_size, nullptr);
    tensor_fp32_to_bf16(ext->attn_o_bf16, dev_s->tb,
                        batch_size * head_dim * p->n_attn_heads, nullptr);
    getp_matmul_attn_o_bf16(dev_s->tb2, ext->attn_o_bf16, dev_w_o, dev_b_o,
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

    // int total_blocks_mlp1 = build_moe_block_schedule(
    //   ext->e_offsets, experts_per_device, MATMUL_MLP1_BLOCK_ROWS,
    //   ext->blk_counts, ext->blk_offsets, nullptr);
    // Change because mlp1 and mlp2 have the same block rows
    int total_blocks_mlp = build_moe_block_schedule(
      ext->e_offsets, experts_per_device, MATMUL_MLP1_BLOCK_ROWS,
      ext->blk_counts, ext->blk_offsets, nullptr);
  
  moe_scatter_acts_to_expert_bf16(
      ext->a_in, dev_s->t, ext->e_dev,
      total_pairs, hidden_dim, nullptr);
  
  HIP_CHECK(hipMemset(dev_s->e_agg, 0,
                      (size_t)batch_size * hidden_dim * sizeof(float)));
  
  __hip_bfloat16 *w1_base = dev_w->w_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim * hidden_dim;
  __hip_bfloat16 *b1_base = dev_w->b_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim;
  
  launch_mlp1_swiglu_bf16_bucketed_outbf16(
      ext->gate_up_bf16, ext->a_in, w1_base, b1_base, ext->e_offsets,
      ext->blk_offsets, hidden_dim, p->intermediate_dim, experts_per_device,
      total_blocks_mlp, p->swiglu_limit, nullptr);
  
  __hip_bfloat16 *w2_base = dev_w->w_mlp2 + 1ll * l * experts_per_device * hidden_dim * p->intermediate_dim;
  __hip_bfloat16 *b2_base = dev_w->b_mlp2 + 1ll * l * experts_per_device * hidden_dim;
  
  // int total_blocks_mlp2 = build_moe_block_schedule(
  //     ext->e_offsets, experts_per_device, MATMUL_MLP2_BLOCK_ROWS,
  //     ext->blk_counts, ext->blk_offsets, nullptr);
  
  launch_mlp2_partial_bf16_bucketed_frombf16(
      ext->z_partial, ext->gate_up_bf16, w2_base, b2_base, ext->w_dev,
      ext->e_offsets, ext->blk_offsets, ext->e_dev, p->intermediate_dim,
      hidden_dim, experts_per_device, total_blocks_mlp, nullptr);
  

    moe_gather_pairs(dev_s->e_agg, ext->z_partial, ext->pair_pos, hidden_dim,
                     p->experts_per_token, batch_size, nullptr);

    getp_vecadd(dev_x, dev_s->e_agg, hidden_dim, batch_size, nullptr);
  }

  getp_rmsnorm(dev_x, dev_x, dev_w->rms_out_w, batch_size, hidden_dim, nullptr);
  getp_matmul_logits_argmax_bf16(dev_x, dev_w->out_bf16,
    hidden_dim, p->vocab_size,
    batch_size, dev_s->topk_i, nullptr);

  int *next_host = (int *)malloc(sizeof(int) * batch_size);
  HIP_CHECK(hipMemcpy(next_host, dev_s->topk_i, sizeof(int) * batch_size,
                      hipMemcpyDeviceToHost));
                      //HIP_CHECK(hipDeviceSynchronize());
  return next_host;
}
