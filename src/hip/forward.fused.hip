#include <hip/hip_runtime.h>
#include <malloc.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "collectives.hpp"
#include "barrier.hpp"
#include "state_ext.hpp"
#include "getp/transformer.cpp"
#include "transformer.hpp"

// Padding tren truc nhanh cua tile LDS. Voi BM=BN=128 buoc giua hai mat k la
// 64 dword, chia het 32 bank -> moi nhom 16 lane doc/ghi trung bank. Them 4
// phan tu lam buoc thanh 66 dword, trai deu tren cac bank.
#ifndef GETP_LDS_PAD
#define GETP_LDS_PAD 4
#endif
// GETP_LDS_PAD o tren van la pad theo TRUC HANG cua bo cuc [BK][BM+PAD] va van
// dung NGUYEN cho matmul_attn_o_bf16_kernel_tuned va
// matmul_logits_argmax_bf16_kernel_tuned. Kernel QKV doi sang bo cuc o (slot)
// [k/4][hang][4]: pad cua no dem theo O 8 byte chu khong theo short, va dieu
// kien hop le phu thuoc BK qua NKC = BK/8 - moi kernel mot khac. Vi vay QKV co
// macro RIENG; dung lai mot macro cho hai DON VI khac nhau la cach chac chan
// nhat de cap phat sai tile mot cach im lang.
//   -1 = tu suy ra gia tri nho nhat khong xung dot bank (xem GetpQkvLdsTile).
#ifndef GETP_QKV_LDS_PAD_SLOTS
#define GETP_QKV_LDS_PAD_SLOTS (-1)
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

// rmsnorm da chuyen xuong DUOI cac ham doi bf16 (tim rmsnorm_kernel_out), vi
// no can f32_to_bf16bits de ghi thang bf16.

union __bf16_bits_u {
  __hip_bfloat16 b;
  unsigned short u;
};

__device__ inline unsigned short bf16_to_bf16bits(__hip_bfloat16 b) {
  __bf16_bits_u t;
  t.b = b;
  return t.u;
}

__device__ inline __hip_bfloat16 bf16bits_to_bf16(unsigned short u) {
  __bf16_bits_u t;
  t.u = u;
  return t.b;
}

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

// ---------------------------------------------------------------------------
// rmsnorm voi dau ra chon duoc: fp32, bf16, hoac CA HAI.
//
// Gia tri fp32 duoc tinh y HET nhu ban cu - weight * (ss * x), hai phep nhan,
// khong co phep cong nen trinh dich khong the hop thanh FMA va doi ket qua.
// Vi mot lan ghi roi doc lai fp32 qua HBM la BIT-CHINH-XAC, ban bf16 ghi thang
// o day bang dung nhung bit ma fp32_to_bf16_kernel rieng se tao ra. Bo kernel
// do di la mot thay doi BIT-EXACT, khong phai xap xi.
//
// Vi sao khong can trao doi qua LDS: viec lam tron chay tren DUNG thread da
// sinh ra gia tri, tren dung phan tu do. Dau ra bf16 la 2 byte moi lane, buoc
// 2 byte -> mot wave ghi 128 byte lien mach.
// ---------------------------------------------------------------------------
template <bool WRITE_F32, bool WRITE_BF16>
__global__ void rmsnorm_kernel_out(float *o, __hip_bfloat16 *ob,
                                   const float *x, const float *weight,
                                   int size) {
  extern __shared__ float smem[];
  float *ss_ptr = &smem[blockDim.x];
  int tid = threadIdx.x;
  // blockIdx.y equal to batch index
  const size_t row = (size_t)blockIdx.y * (size_t)size;
  if constexpr (WRITE_F32) o += row;
  unsigned short *obu = nullptr;
  if constexpr (WRITE_BF16) obu = reinterpret_cast<unsigned short *>(ob) + row;
  x += row;
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
      const float v = weight[tid + offset] * (ss * x[tid + offset]);
      if constexpr (WRITE_F32)  o[tid + offset] = v;
      if constexpr (WRITE_BF16) obu[tid + offset] = f32_to_bf16bits(v);
    }
  }
}

void getp_rmsnorm(float *o, float *x, float *weight, int batch_size, int dim,
                  hipStream_t stream) {
  dim3 blockDim(1024);
  dim3 gridDim(1, batch_size);
  rmsnorm_kernel_out<true, false>
      <<<gridDim, blockDim, sizeof(float) * (blockDim.x + 1), stream>>>(
          o, nullptr, x, weight, dim);
}

// Chi bf16: dung khi ban fp32 khong con ai doc (duong pre-qkv).
void getp_rmsnorm_bf16(__hip_bfloat16 *ob, float *x, float *weight,
                       int batch_size, int dim, hipStream_t stream) {
  dim3 blockDim(1024);
  dim3 gridDim(1, batch_size);
  rmsnorm_kernel_out<false, true>
      <<<gridDim, blockDim, sizeof(float) * (blockDim.x + 1), stream>>>(
          nullptr, ob, x, weight, dim);
}

// Ca hai: dung khi van con mot consumer doc fp32 (router doc ext_t).
void getp_rmsnorm_f32_bf16(float *o, __hip_bfloat16 *ob, float *x,
                           float *weight, int batch_size, int dim,
                           hipStream_t stream) {
  dim3 blockDim(1024);
  dim3 gridDim(1, batch_size);
  rmsnorm_kernel_out<true, true>
      <<<gridDim, blockDim, sizeof(float) * (blockDim.x + 1), stream>>>(
          o, ob, x, weight, dim);
}

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
  
                                                int threads = 256;
  int blocks = ((elems + 3) / 4 + threads - 1) / threads;
  kv_store_pair_fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
      k, v, kdst, vdst, elems);
  
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
  
  const int threads = 256;
  const int vec_elems = (elems + 3) / 4;
  const int blocks = (vec_elems + threads - 1) / threads;
  fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(src, dst, elems);
  // HIP_CHECK(hipGetLastError());
  
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
                                            
  dim3 b(256), g((E + b.x - 1) / b.x);
  moe_blocks_per_expert_kernel<<<g, b, 0, stream>>>(offsets, E, BM, blk_counts);
  exclusive_scan_small_kernel<<<dim3(1), dim3(256), sizeof(int) * 256,
                                stream>>>(blk_counts, blk_offsets, E);
  int total_blocks = 0;
  HIP_CHECK(hipMemcpyAsync(&total_blocks, blk_offsets + E, sizeof(int),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  
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
  
  dim3 block(256), grid(total_pairs);
  moe_scatter_acts_to_expert_bf16_kernel<<<grid, block, 0, stream>>>(
      a_in, x, tok_idx, total_pairs, H);
  
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

// Padding truc nhanh cua tile LDS trong hai kernel MoE. Truoc day hang so 8
// nay duoc viet lai o CA than kernel lan launcher; lech nhau la cap phat
// thieu shared memory va hong tile mot cach im lang.
#ifndef MATMUL_MLP1_LDS_PAD
#define MATMUL_MLP1_LDS_PAD 4
#endif
// mlp1 chuyen sang bo cuc LDS [k/4][hang][4]: pad dem theo O (4 short = 8 byte),
// khong theo short. 2 o = 16 byte moi plane k0; xem Mlp1LdsPlan.
#ifndef MATMUL_MLP1_LDS_PAD_SLOTS
#define MATMUL_MLP1_LDS_PAD_SLOTS 2
#endif
// mlp2 da chuyen sang bo cuc LDS [k/4][hang][4]: pad dem theo O (4 short =
// 8 byte), khong theo short nua, nen no can macro RIENG. MATMUL_MLP2_LDS_PAD
// cu khong con y nghia; bao loi ngay neu con ai truyen -D cu, thay vi de no
// thanh no-op im lang roi bi doc nham thanh nhieu do.
// (GETP_LDS_PAD cua qkv/attn_o/logits va MATMUL_MLP1_LDS_PAD KHONG doi.)
#ifdef MATMUL_MLP2_LDS_PAD
#error "MATMUL_MLP2_LDS_PAD het tac dung: dung MATMUL_MLP2_LDS_PAD_SLOTS (xem Mlp2LdsPlan) va sua dong sweep dang truyen no (run3.sh:33)."
#endif
// -1 = tu suy ra pad nho nhat sach bank tu (BM, BNt, BK). BK=32 -> 2, BK=64 -> 1.
#ifndef MATMUL_MLP2_LDS_PAD_SLOTS
#define MATMUL_MLP2_LDS_PAD_SLOTS (-1)
#endif

#ifndef MATMUL_MLP1_BLOCK_ROWS
#define MATMUL_MLP1_BLOCK_ROWS 128
#endif
#ifndef MATMUL_MLP1_BLOCK_COLS
#define MATMUL_MLP1_BLOCK_COLS 64
#endif
#ifndef MATMUL_MLP1_BLOCK_DEPTH
#define MATMUL_MLP1_BLOCK_DEPTH 32
#endif
#ifndef MATMUL_MLP1_WARP_TILE_M
#define MATMUL_MLP1_WARP_TILE_M 64
#endif
#ifndef MATMUL_MLP1_WARP_TILE_N
#define MATMUL_MLP1_WARP_TILE_N 32
#endif
#ifndef MATMUL_MLP1_WAVES_PER_BLOCK
// Suy tu tile: than kernel dung TILE_N = 64 * (BM/WM)*(BN/WN) thread.
// Ghi de bang -D chi duoc phep neu van bang bieu thuc nay (co static_assert).
#define MATMUL_MLP1_WAVES_PER_BLOCK (((MATMUL_MLP1_BLOCK_ROWS) / (MATMUL_MLP1_WARP_TILE_M)) * ((MATMUL_MLP1_BLOCK_COLS) / (MATMUL_MLP1_WARP_TILE_N)))
#endif
#ifndef NUM_CTA_MLP1
#define NUM_CTA_MLP1 2
#endif

#ifndef MATMUL_MLP2_BLOCK_ROWS
// 96, not 64 and not 128. mlp2 packs 1.78x less MFMA into each LDS read than
// mlp1 does (1.50 against 2.67 per ds_read_b64), which is the whole of why it
// reached 34% of MFMA peak where mlp1 reaches 55%. A taller block raises that
// ratio. 128 is measurably faster still (+6.9%) but produces WRONG output -
// see the note on NUM_CTA_MLP2 - so 96 is the tallest block that is bit-exact.
#define MATMUL_MLP2_BLOCK_ROWS 96
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
// Suy tu tile: than kernel dung TILE_N = 64 * (BM/WM)*(BN/WN) thread.
// Ghi de bang -D chi duoc phep neu van bang bieu thuc nay (co static_assert).
#define MATMUL_MLP2_WAVES_PER_BLOCK (((MATMUL_MLP2_BLOCK_ROWS) / (MATMUL_MLP2_WARP_TILE_M)) * ((MATMUL_MLP2_BLOCK_COLS) / (MATMUL_MLP2_WARP_TILE_N)))
#endif
#ifndef NUM_CTA_MLP2
// The second argument of __launch_bounds__ is a MINIMUM waves-per-SIMD promise,
// and it does not set occupancy - it DIVIDES THE REGISTER BUDGET. gfx90a has
// 512 VGPR per lane per SIMD, so promising N waves caps each wave at 512/N.
//
// The old value of 4 promised an occupancy LDS can never deliver: this tile
// needs 16,640 B of LDS, so a CU holds floor(65536/16640) = 3 blocks = 3 waves
// per SIMD. Occupancy stayed at 3 either way; the only effect of the 4 was to
// cut the budget from 512/3 = 168 registers down to 128, for nothing. mlp1 has
// no such mismatch - its LDS allows 2 blocks and NUM_CTA_MLP1 is exactly 2.
//
// The previous sweep could not have found this: it tried 2, 4 and 8 only. 8
// cuts the budget to 64 and spills; 2 lets the compiler take 256 registers,
// which drops real occupancy to 2 blocks per CU. Both lose, for opposite
// reasons, and the optimum is 3 - which is not a power of two and so was never
// on the grid. The rule is floor(65536 / LDS_bytes) * WARPS / 4.
#define NUM_CTA_MLP2 3
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
// Suy tu tile: than kernel dung TILE_N = 64 * (BM/WM)*(BN/WN) thread.
// Ghi de bang -D chi duoc phep neu van bang bieu thuc nay (co static_assert).
#define MATMUL_ROUTER_WAVES_PER_BLOCK (((MATMUL_ROUTER_BLOCK_ROWS) / (MATMUL_ROUTER_WARP_TILE_M)) * ((MATMUL_ROUTER_BLOCK_COLS) / (MATMUL_ROUTER_WARP_TILE_N)))
#endif
#ifndef MATMUL_ROUTER_THREADS
#define MATMUL_ROUTER_THREADS (64 * MATMUL_ROUTER_WAVES_PER_BLOCK)
#endif

// ===========================================================================
// Bo cuc LDS cua mlp1: KPack = 4  ->  tile[k/4][hang][4]
//
// Mot o 4 short = 8 byte chinh la 4 phan tu k LIEN TIEP ma mot lane can cho
// v_mfma_f32_16x16x16bf16_1k (A[i][k]: lane L thanh ghi r -> i = L%16,
// k = 4*(L/16)+r; B[k][j] doi xung). Nen:
//   doc : 1 x ds_read_b64   thay cho 4 x ds_read_u16
//   ghi : 2 x ds_write_b64  thay cho 8 x ds_write_b16 (mot uint4 tu HBM)
//
// Bank (32 bank x 4 byte = 128 byte/chu ky):
//   DOC  16 lane cua mot pha co cung k0 va hang lien tiep:
//        byte = C + 8*(L%16) -> 128 byte lien tuc = phu kin 32 bank dung mot
//        lan. Dung voi MOI gia tri pad.
//   GHI  16 lane cua mot pha co a = lane%NKC (chunk k, truc nhanh cua loader)
//        va b = lane/NKC (hang): dword offset = 4*S*a + 2*b, S = so o tren mot
//        plane k0. Pad phai lam 16 lane phu kin 16 bank chan - ham duoi kiem.
//        S = BM + 2 = 130: 4*130 = 520 == 8 (mod 32) -> bank = 8a + 2b voi
//        a in [0,4), b in [0,4) -> {0,2,...,30}, moi bank dung mot lan.
//        S = BM + 0 = 128: 4*128 = 512 == 0 (mod 32) -> bank = 2b -> xung dot 4 chieu.
// ===========================================================================
constexpr bool mlp1_lds_write_conflict_free(int S, int NKC) {
  if (NKC <= 0 || NKC > 16 || 16 % NKC != 0) return false;
  int seen = 0;
  for (int l = 0; l < 16; ++l) {
    const int a = l % NKC;
    const int b = l / NKC;
    int bank = (4 * S * a + 2 * b) % 32;
    if (bank < 0) bank += 32;
    if (bank & 1) return false;            // o phai can le 8 byte
    const int bit = 1 << (bank >> 1);
    if (seen & bit) return false;          // hai lane cung bank
    seen |= bit;
  }
  return seen == 0xFFFF;                   // phu kin 16 bank chan
}

// Ke hoach LDS dung CHUNG cho than kernel va launcher: chi con MOT ban so hoc
// padding nen hai ben khong the lech nhau (loi im lang kieu cu).
template <int BM_, int BNt_, int BK_, int PADS_>
struct Mlp1LdsPlan {
  static constexpr int KP   = 4;                 // short / o = 8 byte = 1 ds_read_b64
  static constexpr int KSL  = BK_ / KP;          // so plane k0
  static constexpr int PADS = PADS_;             // pad dem theo O
  static constexpr int SM   = BM_  + PADS;       // o tren mot plane, tile A
  static constexpr int SN   = BNt_ + PADS;       // o tren mot plane, tile W
  static constexpr size_t bytes =
      (size_t)KSL * ((size_t)SM + 2 * (size_t)SN) * KP * sizeof(unsigned short);

  static_assert(BK_ % KP == 0, "BK phai chia het cho KPack = 4");
  static_assert(BK_ % 8 == 0,  "loader nap uint4 = 8 short theo truc k");
  static_assert(mlp1_lds_write_conflict_free(SM, BK_ / 8),
                "MATMUL_MLP1_LDS_PAD_SLOTS sai: ds_write_b64 tile A xung dot bank");
  static_assert(mlp1_lds_write_conflict_free(SN, BK_ / 8),
                "MATMUL_MLP1_LDS_PAD_SLOTS sai: ds_write_b64 tile W xung dot bank");
  static_assert(((size_t)KSL * SM * KP * sizeof(unsigned short)) % 16 == 0,
                "goc tile gate phai can le 16 byte");
  static_assert(((size_t)KSL * SN * KP * sizeof(unsigned short)) % 16 == 0,
                "goc tile up phai can le 16 byte");
};

// Gop hai dword thanh mot o 4 x bf16. Thuan tuy doi kieu: hai VGPR da o dung
// cho, khong sinh lenh VALU nao (khac han 4 lenh trich nua-word cua ban cu).
__device__ __forceinline__ bf16x4 mlp1_pack2(unsigned int lo, unsigned int hi) {
  struct Pair { unsigned int a, b; };
  union { Pair p; bf16x4 h; } cvt;
  cvt.p.a = lo;
  cvt.p.b = hi;
  return cvt.h;
}

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
  // BM = BN = 64, BK = 32
  // WM = WN = 32
  // TILE_N = 64 * ((BM/WM) * (BN/WN)) = 64 * 2 * 2 = 256
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

  // Bo cuc LDS [k/4][hang][4] - xem Mlp1LdsPlan.
  using LdsPlan = Mlp1LdsPlan<BM, BNt, BK, MATMUL_MLP1_LDS_PAD_SLOTS>;
  constexpr int KP  = LdsPlan::KP;    // 4 short / o
  constexpr int KSL = LdsPlan::KSL;   // so plane k0 = BK/4
  constexpr int SM  = LdsPlan::SM;    // o / plane, tile A
  constexpr int SN  = LdsPlan::SN;    // o / plane, tile W

  extern __shared__ unsigned short sm[];
  bf16x4* const sxs  = reinterpret_cast<bf16x4*>(sm);
  bf16x4* const swgs = sxs  + (size_t)KSL * SM;
  bf16x4* const swus = swgs + (size_t)KSL * SN;

#ifdef DEBUG_LDS_POISON
  // Kiem launch: than kernel gia dinh dung TILE_N thread.
  if (blockDim.x != TILE_N) {
    if (threadIdx.x == 0)
      printf("BAD LAUNCH: blockDim=%d nhung TILE_N=%d\n", (int)blockDim.x, (int)TILE_N);
    return;
  }
  // Do day LDS bang bf16 qNaN: o nao khong duoc loader ghi se thanh NaN.
  {
    const int lds_shorts = KSL * (SM + 2 * SN) * KP;
    for (int idx = threadIdx.x; idx < lds_shorts; idx += blockDim.x) sm[idx] = 0x7FC0;
  }
  __syncthreads();
#endif

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

  constexpr int LOAD_VEC = 8;
  const int loadX_x8 = threadIdx.x % (BK / LOAD_VEC);
  const int loadX_y8 = threadIdx.x / (BK / LOAD_VEC);
  constexpr int strideX8 = TILE_N / (BK / LOAD_VEC); // 256 / (32 / 8) = 64
  constexpr int nbLoadsX = BM / strideX8;
  constexpr int nbLoadsW = BNt / strideW;

  // Fast/Slow Path Split optimization: Separate boundary-safe and boundary-check loads
  constexpr int safe_iters = (BM >= strideX8) ? (BM - strideX8) / strideX8 : 0;
  
  // Fast path: No boundary checks (all safe iterations)
  #pragma unroll
  for (int i = 0; i < safe_iters; ++i) {
    const int off = i * strideX8;
    const int kk = 0 + LOAD_VEC * loadX_x8;
    const int r = loadX_y8 + off;
    const int pos = base + r;
    
    // 8 short lien tiep theo k cua mot hang = 2 o ke nhau theo truc plane k0.
    bf16x4* dst = &sxs[(size_t)((LOAD_VEC * loadX_x8) / KP) * SM + r];

    if (r < tcnt && kk + 7 < H) {
      const uint4* p = reinterpret_cast<const uint4*>(A_in + (size_t)pos * H + kk);
      const uint4 v = *p;

      dst[0]  = mlp1_pack2(v.x, v.y);   // k+0 .. k+3
      dst[SM] = mlp1_pack2(v.z, v.w);   // k+4 .. k+7
    } else {
      dst[0]  = mlp1_pack2(0u, 0u);
      dst[SM] = mlp1_pack2(0u, 0u);
    }
  }
  
  // Slow path: With boundary checks (remaining iterations)
  #pragma unroll
  for (int i = safe_iters; i * strideX8 < BM; ++i) {
    const int off = i * strideX8;
    const int kk = 0 + LOAD_VEC * loadX_x8;
    const int r = loadX_y8 + off;
    const int pos = base + r;
    
    // 8 short lien tiep theo k cua mot hang = 2 o ke nhau theo truc plane k0.
    bf16x4* dst = &sxs[(size_t)((LOAD_VEC * loadX_x8) / KP) * SM + r];

    if (r < tcnt && kk + 7 < H) {
      const uint4* p = reinterpret_cast<const uint4*>(A_in + (size_t)pos * H + kk);
      const uint4 v = *p;

      dst[0]  = mlp1_pack2(v.x, v.y);   // k+0 .. k+3
      dst[SM] = mlp1_pack2(v.z, v.w);   // k+4 .. k+7
    } else {
      dst[0]  = mlp1_pack2(0u, 0u);
      dst[SM] = mlp1_pack2(0u, 0u);
    }
  }

  #pragma unroll
  for (int off = 0; off < BNt; off += strideW) {
    const int kk = 0 + 8 * loadW_x;
    const int col = colBase + loadW_y + off;

    static_assert(LOAD_VEC == 2 * KP, "mot uint4 tu HBM phai phu dung 2 o KPack");
    bf16x4* dg = &swgs[(size_t)((8 * loadW_x) / KP) * SN + (loadW_y + off)];
    bf16x4* du = &swus[(size_t)((8 * loadW_x) / KP) * SN + (loadW_y + off)];

    if (col < I && kk + 7 < H) {
      const size_t w1_gate_off = w1_e + (size_t)(2 * col + 0) * H + kk;
      const size_t w1_up_off = w1_e + (size_t)(2 * col + 1) * H + kk;
      const uint4* pg = reinterpret_cast<const uint4*>(W1 + w1_gate_off);
      const uint4* pu = reinterpret_cast<const uint4*>(W1 + w1_up_off);
      const uint4 wg = *pg;
      const uint4 wu = *pu;

      dg[0]  = mlp1_pack2(wg.x, wg.y);
      dg[SN] = mlp1_pack2(wg.z, wg.w);
      du[0]  = mlp1_pack2(wu.x, wu.y);
      du[SN] = mlp1_pack2(wu.z, wu.w);
    } else {
      dg[0]  = mlp1_pack2(0u, 0u);
      dg[SN] = mlp1_pack2(0u, 0u);
      du[0]  = mlp1_pack2(0u, 0u);
      du[SN] = mlp1_pack2(0u, 0u);
    }
  }
  __syncthreads();

  #pragma unroll 1
  for (int bk = 0; bk < H; bk += BK) {
    uint4 regX[nbLoadsX];
    uint4 regWU[nbLoadsW];
    uint4 regWG[nbLoadsW];
    if (bk + BK < H) {
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX8;
        const int kk = bk + BK + LOAD_VEC * loadX_x8;
        const int r = loadX_y8 + off;
        const int pos = base + r;
        
        if (r < tcnt && kk + 7 < H) {
          const uint4* p = reinterpret_cast<const uint4*>(A_in + (size_t)pos * H + kk);
          regX[i] = *p;
        } else {
          regX[i] = make_uint4(0, 0, 0, 0);
        }
      }
  
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;
        const int kk = bk + BK + 8 * loadW_x;
        const int col = colBase + loadW_y + off;
  
        if (col < I && kk + 7 < H) {
          const size_t w1_gate_off = w1_e + (size_t)(2 * col + 0) * H + kk;
          const size_t w1_up_off = w1_e + (size_t)(2 * col + 1) * H + kk;
          const uint4* pg = reinterpret_cast<const uint4*>(W1 + w1_gate_off);
          const uint4* pu = reinterpret_cast<const uint4*>(W1 + w1_up_off);
          
          regWG[i] = *pg;
          regWU[i] = *pu;
          
        } else {
          regWG[i] = make_uint4(0, 0, 0, 0);
          regWU[i] = make_uint4(0, 0, 0, 0);
        }
      }
    }

    // MFMA ILP optimization: Pre-load all vectors to improve instruction-level parallelism
    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
      // basek = k + 4*yMF luon chia het 4 -> plane k0 = basek/4 = (k>>2) + yMF.
      const int kslot = (k >> 2) + yMF;

      // Pre-load all activation vectors
      bf16x4 av_array[nIterM];
      #pragma unroll
      for (int im = 0; im < nIterM; ++im) {
        const int row = yTile * WM + im * 16 + xMF;
        av_array[im] = sxs[(size_t)kslot * SM + row];            // 1 x ds_read_b64
      }

      // Compute with better ILP
      #pragma unroll
      for (int im = 0; im < nIterM; ++im) {
        const bf16x4 av = av_array[im];

        #pragma unroll
        for (int g = 0; g < BN_AGG; ++g) {
          const int col_base = xTile * WN + xMF + g * BN;

          // Pre-load all weight vectors for this aggregation group
          bf16x4 gv_array[nIterN];
          bf16x4 uv_array[nIterN];
          #pragma unroll
          for (int in = 0; in < nIterN; ++in) {
            const int c = col_base + in * 16;
            gv_array[in] = swgs[(size_t)kslot * SN + c];         // 1 x ds_read_b64
            uv_array[in] = swus[(size_t)kslot * SN + c];         // 1 x ds_read_b64
          }
          
          // Execute MFMAs with all data in registers
          #pragma unroll
          for (int in = 0; in < nIterN; ++in) {
            dg_acc[g][im][in] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, gv_array[in], dg_acc[g][im][in], 0, 0, 0);
            du_acc[g][im][in] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, uv_array[in], du_acc[g][im][in], 0, 0, 0);
          }
        }
      }
    }

    __syncthreads();

    if (bk + BK < H) {
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX8;
        const int r = loadX_y8 + off;

        bf16x4* dst = &sxs[(size_t)((LOAD_VEC * loadX_x8) / KP) * SM + r];

        const uint4 v = regX[i];

        dst[0]  = mlp1_pack2(v.x, v.y);
        dst[SM] = mlp1_pack2(v.z, v.w);
      }

      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;

        bf16x4* dg = &swgs[(size_t)((8 * loadW_x) / KP) * SN + (loadW_y + off)];
        bf16x4* du = &swus[(size_t)((8 * loadW_x) / KP) * SN + (loadW_y + off)];

        const uint4 wg = regWG[i];
        const uint4 wu = regWU[i];

        dg[0]  = mlp1_pack2(wg.x, wg.y);
        dg[SN] = mlp1_pack2(wg.z, wg.w);
        du[0]  = mlp1_pack2(wu.x, wu.y);
        du[SN] = mlp1_pack2(wu.z, wu.w);
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
      if (col < I && B1) {
        const size_t bias_idx = b1_e + (size_t)(2 * col);
        const uint bias_pair = *reinterpret_cast<const uint*>(&B1[bias_idx]);
        bg_loc[g][in] = __bfloat162float(bf16bits_to_bf16((unsigned short)(bias_pair & 0xFFFF)));
        bu_loc[g][in] = __bfloat162float(bf16bits_to_bf16((unsigned short)(bias_pair >> 16)));
      } else {
        bg_loc[g][in] = 0.f;
        bu_loc[g][in] = 0.f;
      }
    }
  }
  
  constexpr float swiglu_a = 1.702f;
  const float swiglu_limit_neg = -swiglu_limit;
  
  #pragma unroll
  for (int im = 0; im < nIterM; ++im) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = yTile * WM + im * 16 + (yD + i);
      if (row >= tcnt) continue;
      
      const int pos = base + row;
      unsigned short* output_row = reinterpret_cast<unsigned short*>(gate_up_bf16) + (size_t)pos * I;
      
      #pragma unroll
      for (int g = 0; g < BN_AGG; ++g) {
        #pragma unroll
        for (int in = 0; in < nIterN; ++in) {
          const int col = blockIdx.x * BNt + xTile * WN + in * 16 + xD + g * BN;
          if (col >= I) continue;
          
          float gval = fminf(fmaxf(dg_acc[g][im][in][i] + bg_loc[g][in], swiglu_limit_neg), swiglu_limit);
          float uval = fminf(fmaxf(du_acc[g][im][in][i] + bu_loc[g][in], swiglu_limit_neg), swiglu_limit);
          
          const float sgm = 1.f / (1.f + __expf(-swiglu_a * gval));
          const float out = (gval * sgm) * (uval + 1.f);
          
          output_row[col] = f32_to_bf16bits(out);
        }
      }
    }
  }
}

// ===========================================================================
// Bo cuc LDS cua mlp2: KPack = 4  ->  tile[k/4][hang][4]   (cu: tile[k][hang])
//
// Mot o 4 short = 8 byte chinh la 4 phan tu k LIEN TIEP ma mot lane can cho
// v_mfma_f32_16x16x16bf16_1k (A[i][k]: lane L thanh ghi r -> i = L%16,
// k = 4*(L/16)+r; B[k][j] doi xung). Nen:
//   doc : 1 x ds_read_b64   thay cho 4 x ds_read_u16
//   ghi : 2 x ds_write_b64  thay cho 8 x ds_write_b16 (mot uint4 tu HBM)
//
// Bank (32 bank x 4 byte = 128 byte/chu ky), so cua cau hinh mlp2 mac dinh
// BM=64, BN=64, BN_AGG=3 -> BNt=192, BK=32, WM=WN=32, 256 thread:
//
//   DOC  mot pha ds_read_b64 = 16 lane cung yMF (nen CUNG plane k0), xMF=0..15
//        chay tren 16 HANG lien tiep:  byte(L) = C + 8*L  ->  128 byte lien
//        tuc = 32 bank, moi bank dung mot lan  => SACH VOI MOI PAD.
//        (Ban cu: 16 lane doc 16 short lien tiep = 8 dword -> 8 bank; hai lane
//         chung mot dword la broadcast nen "khong xung dot", nhung moi chu ky
//         chi chuyen 64 byte co ich = mot nua bang thong, va ton 4 lan so lenh.)
//
//   GHI  loader giu NGUYEN anh xa thread -> (hang, chunk k):
//          a = tid % NKC (chunk 8 short theo k), b = tid / NKC (hang), NKC=BK/8.
//        Mot uint4 = 8 short theo k = 2 o KE NHAU theo truc plane k0:
//          dword(a,b) = 4*S*a + 2*b   (S = so o tren mot plane k0)
//        S = BM  + 2 =  66 : 4*66  = 264 == 8 (mod 32) -> bank = 8a + 2b voi
//                           a in [0,4), b in [0,4) -> {0,2,...,30}, moi bank
//                           dung dung mot lan  => GHI SACH.
//        S = BNt + 2 = 194 : 4*194 = 776 == 8 (mod 32) -> y het tren => SACH.
//        Ban cu (PAD=8): buoc chunk 8*BM_PAD/2 = 8*72/2 = 288 dword == 0 (mod
//        32), va 8*BNt_PAD/2 = 800 == 0 (mod 32) -> CA BON chunk roi vao cung
//        nhom bank: ds_write_b16 xung dot 4 CHIEU tren ca hai tile.
//
// Vi sao khong dung bo cuc phang [hang][k] (KPAD = BK + 4 short): o do doc chi
// sach khi (BK+PAD)/4 le, con GHI van 2 chieu voi MOI pad; bo cuc o nay ghi
// SACH that. No cung ton it LDS hon: 16640 B, so voi 18432 B cua [hang][k] va
// 17408 B cua ban hien tai.
// ===========================================================================
constexpr bool mlp2_lds_write_conflict_free(int S, int NKC) {
  if (NKC <= 0 || NKC > 16 || 16 % NKC != 0) return false;
  int seen = 0;
  for (int l = 0; l < 16; ++l) {         // mot pha ds_write_b64 = 16 lane
    const int a = l % NKC;               // chunk k (truc nhanh cua loader)
    const int b = l / NKC;               // hang
    int bank = (4 * S * a + 2 * b) % 32;
    if (bank < 0) bank += 32;
    if (bank & 1) return false;          // o 8 byte can le -> bank phai chan
    const int bit = 1 << (bank >> 1);
    if (seen & bit) return false;        // hai lane cung bank
    seen |= bit;
  }
  return seen == 0xFFFF;                 // phu kin 16 bank chan
}

// Pad nho nhat hop le cho CA hai tile, suy tu (BM, BNt, BK) thay vi hang so
// cung: BK=32 -> 2 (2, 6, 10 deu duoc); BK=64 -> 1 (1,3,5,7); BK=16 -> 4.
// Nho vay moi dong sweep mlp2 dang co (bm128 / bn128 / 128x128 / bk64) van
// bien dich duoc ma khong phai suy lai pad bang tay.
constexpr int mlp2_lds_pad_slots_auto(int BM_, int BNt_, int NKC) {
  for (int p = 0; p <= 16; ++p)
    if (mlp2_lds_write_conflict_free(BM_ + p, NKC) &&
        mlp2_lds_write_conflict_free(BNt_ + p, NKC))
      return p;
  return -1;
}

// Ke hoach LDS dung CHUNG cho than kernel va launcher: chi con MOT ban so hoc
// padding nen hai ben khong the lech nhau (loi im lang kieu cu: launcher cap
// thieu dynamic LDS thi tile hong ma khong ai bao).
template <int BM_, int BNt_, int BK_, int PADS_REQ_>
struct Mlp2LdsPlan {
  static constexpr int KP   = 4;                 // short / o = 8 byte = 1 ds_read_b64
  static constexpr int KSL  = BK_ / KP;          // so plane k0
  static constexpr int NKC  = BK_ / 8;           // so chunk 8-short tren mot hang
  static constexpr int PADS = (PADS_REQ_ >= 0)
                                ? PADS_REQ_
                                : mlp2_lds_pad_slots_auto(BM_, BNt_, NKC);
  static constexpr int SM   = BM_  + PADS;       // o tren mot plane, tile A
  static constexpr int SN   = BNt_ + PADS;       // o tren mot plane, tile W
  static constexpr size_t bytes =
      (size_t)KSL * ((size_t)SM + (size_t)SN) * KP * sizeof(unsigned short);

  static_assert(BK_ % KP == 0, "BK phai chia het cho KPack = 4");
  static_assert(BK_ % 8 == 0,  "loader nap uint4 = 8 short theo truc k");
  static_assert(PADS >= 0,
                "khong tim duoc pad sach bank cho (BM,BNt,BK) nay - xem mlp2_lds_pad_slots_auto");
  static_assert(mlp2_lds_write_conflict_free(SM, NKC),
                "MATMUL_MLP2_LDS_PAD_SLOTS sai: ds_write_b64 tile A xung dot bank"
                " (BK=32 dung 2, BK=64 dung 1, BK=16 dung 4)");
  static_assert(mlp2_lds_write_conflict_free(SN, NKC),
                "MATMUL_MLP2_LDS_PAD_SLOTS sai: ds_write_b64 tile W xung dot bank"
                " (BK=32 dung 2, BK=64 dung 1, BK=16 dung 4)");
  static_assert(((size_t)KSL * SM * KP * sizeof(unsigned short)) % 16 == 0,
                "goc tile W phai can le 16 byte");
  static_assert(bytes <= 64u * 1024u, "tile LDS mlp2 vuot 64 KB/CU");
};

// Gop hai dword thanh mot o 4 x bf16. Thuan tuy doi kieu: hai VGPR da nam dung
// cho nen khong sinh lenh VALU nao (khac 4 lenh trich nua-word cua ban cu).
// __builtin_bit_cast la cach viet chuan, khong phai punning qua union.
__device__ __forceinline__ bf16x4 mlp2_pack2(unsigned int lo, unsigned int hi) {
  struct Pair { unsigned int a, b; };
  return __builtin_bit_cast(bf16x4, Pair{lo, hi});
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

  // Bo cuc LDS [k/4][hang][4] - xem Mlp2LdsPlan. Hinh hoc tinh MOT lan o
  // day, va launcher dung lai dung ke hoach nay.
  using LdsPlan = Mlp2LdsPlan<BM, BNt, BK, MATMUL_MLP2_LDS_PAD_SLOTS>;
  constexpr int KP  = LdsPlan::KP;    // 4 short / o
  constexpr int KSL = LdsPlan::KSL;   // so plane k0 = BK/4
  constexpr int SM  = LdsPlan::SM;    // o / plane, tile A
  constexpr int SN  = LdsPlan::SN;    // o / plane, tile W

  // __align__(16): goc vung LDS dong duoc BAO DAM can le 16 byte, nen ep
  // kieu sang bf16x4 (can le 8) la dung theo kieu, khong phai tinh co dung.
  // Ten rieng mlp2_sm de thuoc tinh align khong va voi `sm` cua kernel khac.
  extern __shared__ __align__(16) unsigned short mlp2_sm[];
  bf16x4* const sxs = reinterpret_cast<bf16x4*>(mlp2_sm);
  bf16x4* const sws = sxs + KSL * SM;

#ifdef DEBUG_LDS_POISON
  // Kiem launch: than kernel gia dinh dung TILE_N thread.
  if (blockDim.x != TILE_N) {
    if (threadIdx.x == 0)
      printf("BAD LAUNCH: blockDim=%d nhung TILE_N=%d\n", (int)blockDim.x, (int)TILE_N);
    return;
  }
  // Do day LDS bang bf16 qNaN: o nao khong duoc loader ghi se thanh NaN.
  {
    const int lds_shorts = KSL * (SM + SN) * KP;
    for (int idx = threadIdx.x; idx < lds_shorts; idx += blockDim.x) mlp2_sm[idx] = 0x7FC0;
  }
  __syncthreads();
#endif

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

  constexpr int LOAD_VEC = 8;
  const int loadX_x8 = threadIdx.x % (BK / LOAD_VEC);
  const int loadX_y8 = threadIdx.x / (BK / LOAD_VEC);
  constexpr int strideX8 = TILE_N / (BK / LOAD_VEC);

  static_assert(BM % strideX8 == 0);
  static_assert(BNt % strideX8 == 0);
  constexpr int nbLoadsX = BM / strideX8;
  constexpr int nbLoadsW = BNt / strideW;

  // Fast/Slow Path Split for MLP2 initial load
  constexpr int safe_iters_mlp2 = (BM >= strideX8) ? (BM - strideX8) / strideX8 : 0;
  
  // Fast path: boundary-safe loads
  #pragma unroll
  for (int i = 0; i < safe_iters_mlp2; ++i) {
    const int off = i * strideX8;
    const int kk = kBeg + LOAD_VEC * loadX_x8;
    const int r = loadX_y8 + off;
    const int pos = base + r;
    
    // 8 short lien tiep theo k cua MOT hang = 2 o ke nhau theo truc plane k0.
    bf16x4* dst = &sxs[((LOAD_VEC * loadX_x8) / KP) * SM + r];
    
    if (r < tcnt && kk + 7 < kEnd) {
      const __hip_bfloat16* src = gate_up_bf16 + (size_t)pos * (size_t)I + kk;
      const uint4* p = reinterpret_cast<const uint4*>(src);
      const uint4 v = *p;
      
      dst[0]  = mlp2_pack2(v.x, v.y);   // k+0 .. k+3   ds_write_b64
      dst[SM] = mlp2_pack2(v.z, v.w);   // k+4 .. k+7   ds_write_b64
    } else {
      dst[0]  = mlp2_pack2(0u, 0u);
      dst[SM] = mlp2_pack2(0u, 0u);
    }
  }
  
  // Slow path: with boundary checks
  #pragma unroll
  for (int i = safe_iters_mlp2; i * strideX8 < BM; ++i) {
    const int off = i * strideX8;
    const int kk = kBeg + LOAD_VEC * loadX_x8;
    const int r = loadX_y8 + off;
    const int pos = base + r;
    
    // 8 short lien tiep theo k cua MOT hang = 2 o ke nhau theo truc plane k0.
    bf16x4* dst = &sxs[((LOAD_VEC * loadX_x8) / KP) * SM + r];
    
    if (r < tcnt && kk + 7 < kEnd) {
      const __hip_bfloat16* src = gate_up_bf16 + (size_t)pos * (size_t)I + kk;
      const uint4* p = reinterpret_cast<const uint4*>(src);
      const uint4 v = *p;
      
      dst[0]  = mlp2_pack2(v.x, v.y);   // k+0 .. k+3   ds_write_b64
      dst[SM] = mlp2_pack2(v.z, v.w);   // k+4 .. k+7   ds_write_b64
    } else {
      dst[0]  = mlp2_pack2(0u, 0u);
      dst[SM] = mlp2_pack2(0u, 0u);
    }
  }

  // Weight load with padding
  #pragma unroll
  for (int off = 0; off < BNt; off += strideW) {
    const int kk = kBeg + 8 * loadW_x;
    const int col = colBase + loadW_y + off;
    
    static_assert(LOAD_VEC == 2 * KP, "mot uint4 tu HBM phai phu dung 2 o KPack");
    bf16x4* dst = &sws[((8 * loadW_x) / KP) * SN + (loadW_y + off)];
    
    if (col < H && kk + 7 < kEnd) {
      const size_t w2_off = w2_e + (size_t)col * (size_t)I + kk;
      const uint4* pw = reinterpret_cast<const uint4*>(W2 + w2_off);
      const uint4 wb = *pw;
      
      dst[0]  = mlp2_pack2(wb.x, wb.y);   // k+0 .. k+3   ds_write_b64
      dst[SN] = mlp2_pack2(wb.z, wb.w);   // k+4 .. k+7   ds_write_b64
    } else {
      dst[0]  = mlp2_pack2(0u, 0u);
      dst[SN] = mlp2_pack2(0u, 0u);
    }
  }

  __syncthreads();

  #pragma unroll 1
  for (int bk = kBeg; bk < kEnd; bk += BK) {
    uint4 regX[nbLoadsX];
    uint4 regW[nbLoadsW];

    if (bk + BK < kEnd) {
      // Optimized activation load: uint4 (8 bf16)
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX8;
        const int kk = bk + BK + LOAD_VEC * loadX_x8;
        const int r = loadX_y8 + off;
        const int pos = base + r;
        
        if (r < tcnt && kk + 7 < kEnd) {
          // Vectorized uint4 load (8 bf16)
          const __hip_bfloat16* src = gate_up_bf16 + (size_t)pos * (size_t)I + kk;
          const uint4* p = reinterpret_cast<const uint4*>(src);
          regX[i] = *p;
        } else {
          regX[i] = make_uint4(0, 0, 0, 0);
        }
      }

      // Optimized weight load: uint4 with full unroll
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;
        const int kk = bk + BK + 8 * loadW_x;
        const int col = colBase + loadW_y + off;
        
        if (col < H && kk + 7 < kEnd) {
          // Vectorized uint4 load (8 bf16)
          const size_t w2_off = w2_e + (size_t)col * (size_t)I + kk;
          const uint4* pw = reinterpret_cast<const uint4*>(W2 + w2_off);
          regW[i] = *pw;
        } else {
          regW[i] = make_uint4(0, 0, 0, 0);
        }
      }
    }

    // MFMA ILP optimization for MLP2: Pre-load all vectors
    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
      // basek = k + 4*yMF luon chia het 4 -> plane k0 = basek/4 = (k>>2) + yMF;
      // 4 phan tu mot lane can (k..k+3 cua CUNG mot hang) nam gon trong mot o.
      const int kslot = (k >> 2) + yMF;
      
      // Pre-load all activation vectors
      bf16x4 av_array[nIterM];
      #pragma unroll
      for (int im = 0; im < nIterM; ++im) {
        const int row = yTile * WM + im * 16 + xMF;
        av_array[im] = sxs[kslot * SM + row];              // 1 x ds_read_b64
      }
      
      // Compute with improved ILP
      #pragma unroll
      for (int im = 0; im < nIterM; ++im) {
        const bf16x4 av = av_array[im];
        
        #pragma unroll
        for (int g = 0; g < BN_AGG; ++g) {
          const int col_base = xTile * WN + xMF + g * BN;
          
          // Pre-load all weight vectors
          bf16x4 bv_array[nIterN];
          #pragma unroll
          for (int in = 0; in < nIterN; ++in) {
            const int c = col_base + in * 16;
            bv_array[in] = sws[kslot * SN + c];                // 1 x ds_read_b64
          }
          
          // Execute MFMAs with all data in registers
          #pragma unroll
          for (int in = 0; in < nIterN; ++in) {
            d_acc[g][im][in] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv_array[in], d_acc[g][im][in], 0, 0, 0);
          }
        }
      }
    }
    __syncthreads();

    if (bk + BK < kEnd) {
      // Store activation with padding
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX8;
        const int r = loadX_y8 + off;
        const int pos = base + r;
        
        // 8 short lien tiep theo k cua MOT hang = 2 o ke nhau theo truc plane k0.
        bf16x4* dst = &sxs[((LOAD_VEC * loadX_x8) / KP) * SM + r];
        
        const uint4 v = regX[i];
        
        dst[0]  = mlp2_pack2(v.x, v.y);   // k+0 .. k+3   ds_write_b64
        dst[SM] = mlp2_pack2(v.z, v.w);   // k+4 .. k+7   ds_write_b64
      }

      // Store weight with padding
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;
        const int col = colBase + loadW_y + off;
        
        static_assert(LOAD_VEC == 2 * KP, "mot uint4 tu HBM phai phu dung 2 o KPack");
        bf16x4* dst = &sws[((8 * loadW_x) / KP) * SN + (loadW_y + off)];
        
        const uint4 wb = regW[i];
        
        dst[0]  = mlp2_pack2(wb.x, wb.y);   // k+0 .. k+3   ds_write_b64
        dst[SN] = mlp2_pack2(wb.z, wb.w);   // k+4 .. k+7   ds_write_b64
      }
    }

    __syncthreads();
  }

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  
  // Optimized output with full unroll
  #pragma unroll
  for (int im = 0; im < nIterM; ++im) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = yTile * WM + im * 16 + (yD + i);
      if (row >= tcnt) continue;
      
      const int pos = base + row;
      const float w = w_by_bucket[pos];
      float* output_row = z_partial + (size_t)pos * (size_t)H;
      
      #pragma unroll
      for (int g = 0; g < BN_AGG; ++g) {
        #pragma unroll
        for (int in = 0; in < nIterN; ++in) {
          const int col = blockIdx.x * (BN * BN_AGG) + xTile * WN + in * 16 + xD + g * BN;
          if (col >= H) continue;
          
          const float bia = (B2 && splits == 1) ? __bfloat162float(B2[b2_e + col]) : 0.f;
          float* dst = output_row + col;
          
          if (splits == 1) {
            *dst = (d_acc[g][im][in][i] + bia) * w;
          } else {
            atomicAdd(dst, d_acc[g][im][in][i] * w);
            if (blockIdx.z == 0 && B2) atomicAdd(dst, __bfloat162float(B2[b2_e + col]) * w);
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
  
  constexpr int BM = MATMUL_MLP1_BLOCK_ROWS;
  constexpr int BN = MATMUL_MLP1_BLOCK_COLS;
  constexpr int BK = MATMUL_MLP1_BLOCK_DEPTH;
  constexpr int WM = MATMUL_MLP1_WARP_TILE_M;
  constexpr int WN = MATMUL_MLP1_WARP_TILE_N;
  constexpr int WARPS = MATMUL_MLP1_WAVES_PER_BLOCK;
  constexpr int CTA = NUM_CTA_MLP1;
  constexpr int THREADS = 64 * WARPS;
  static_assert(WARPS == (BM / WM) * (BN / WN),
                "WAVES_PER_BLOCK phai bang (BM/WM)*(BN/WN): than kernel cap"
                " TILE_N = 64*(BM/WM)*(BN/WN) thread, launcher cap 64*WARPS.");
  // BN_AGG la MOT hang so duy nhat cho ca BNt lan tham so mau cua kernel: truoc
  // day so 2 duoc viet lai o hai cho, lech nhau la cap phat thieu LDS im lang.
  constexpr int BN_AGG = 2;
  constexpr int BNt = BN * BN_AGG;
  using LdsPlan = Mlp1LdsPlan<BM, BNt, BK, MATMUL_MLP1_LDS_PAD_SLOTS>;
  dim3 block(THREADS);
  dim3 grid((I + BNt - 1) / BNt, total_blocks);
  // MOT nguon su that cho kich thuoc LDS - dung ke hoach ma than kernel dung.
  size_t shmem = LdsPlan::bytes;
  mlp1_swiglu_bf16_bucketed_kernel_outbf16_tuned<BM,BN,BK,WM,WN,BN_AGG,CTA>
    <<<grid, block, shmem, stream>>>(gate_up_bf16, a_in, w1_layer, b1_layer, offsets, blk_offs, H, I, E, swiglu_limit);
    
}

static inline void launch_mlp2_partial_bf16_bucketed_frombf16(
  float* z_partial, const __hip_bfloat16* gate_up_bf16,
  const __hip_bfloat16* w2_layer, const __hip_bfloat16* b2_layer,
  const float* w_by_bucket, const int* offsets, const int* blk_offs,
  const int* tok_idx, int I, int H, int E, int total_blocks, hipStream_t stream
) {
  
  constexpr int BM = MATMUL_MLP2_BLOCK_ROWS;
  constexpr int BN = MATMUL_MLP2_BLOCK_COLS;
  constexpr int BK = MATMUL_MLP2_BLOCK_DEPTH;
  constexpr int WM = MATMUL_MLP2_WARP_TILE_M;
  constexpr int WN = MATMUL_MLP2_WARP_TILE_N;
  constexpr int WARPS = MATMUL_MLP2_WAVES_PER_BLOCK;
  constexpr int THREADS = 64 * WARPS;
  static_assert(WARPS == (BM / WM) * (BN / WN),
                "WAVES_PER_BLOCK phai bang (BM/WM)*(BN/WN): than kernel cap"
                " TILE_N = 64*(BM/WM)*(BN/WN) thread, launcher cap 64*WARPS.");
  // BN_AGG la MOT hang so duy nhat cho ca BNt lan tham so mau cua kernel:
  // truoc day so 3 duoc viet lai o hai cho, lech nhau la hong tile im lang.
  constexpr int BN_AGG = 3;
  constexpr int BNt = BN * BN_AGG;

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

  // MOT nguon su that cho hinh hoc LDS: dung chinh ke hoach cua than kernel.
  using LdsPlan = Mlp2LdsPlan<BM, BNt, BK, MATMUL_MLP2_LDS_PAD_SLOTS>;
  dim3 block(THREADS);
  dim3 grid(max(1, gx), max(1, gy), splits);
  // __launch_bounds__(TILE_N, NUM_CTA_MLP2): doi so thu hai cua HIP la SO
  // WAVE TOI THIEU TREN MOI SIMD, khong phai so block tren CU. Voi 256
  // thread (4 wave/block) va NUM_CTA_MLP2 = 4 thi so block 'hua' la
  // 4*CTA/WARPS = 4, can 4*16640 = 66560 B > 64 KB, nen LDS van chan o
  // 3 block/CU (3*16640 = 49920 B) - dung nhu ban cu (3*17408 = 52224 B).
  // Occupancy KHONG doi, chi du them 2304 B. Ghi ro de lan sau doi tile
  // khong ai tuong doi so CTA la so block tren CU.
  size_t shmem = LdsPlan::bytes;

  mlp2_partial_bf16_bucketed_splitk_kernel_inbf16_tuned<BM,BN,BK,WM,WN,BN_AGG,NUM_CTA_MLP2>
    <<<grid, block, shmem, stream>>>(z_partial, gate_up_bf16, w2_layer, b2_layer, w_by_bucket,
                                     offsets, blk_offs, tok_idx, I, H, E, splits);
  
}


// Lat cat [own_lo, own_hi) cua e_agg la phan danh cho TOKEN CUA CHINH THIET BI
// NAY: khong peer nao doc no (peer keo lat cat mang chi so cua PEER tu buffer
// cua ta). Nen thay vi ghi ra roi de vecadd doc lai, cong thang vao dong du.
// Nhanh re theo blockIdx.y nen dong nhat trong ca block, khong phan ky.
__global__ void moe_gather_pairs_kernel(float *__restrict__ e_agg,
                                        float *__restrict__ x_resid,
                                        int own_lo, int own_hi,
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
  if (b >= own_lo && b < own_hi) {
    // Bit-exact: s la dung so fp32 ma e_agg se chua, va vecadd cung cong
    // dung mot lan nhu the.
    x_resid[(size_t)(b - own_lo) * (size_t)H + row] += s;
  } else {
    // Ghi DE, khong cong don: moi o duoc ghi dung mot lan (luoi phu tron
    // B x H), nen khong can memset 35,4 MB truoc do nua.
    e_agg[(size_t)b * (size_t)H + row] = s;
  }
}

static inline void moe_gather_pairs(float *e_agg, const float *z_partial,
                                    const int *pair_pos, int H, int K, int B,
                                    hipStream_t stream) {

  dim3 blk(256), grd((H + blk.x - 1) / blk.x, B);
  // own_lo == own_hi: khong hang nao di vao dong du, y het ban cu.
  moe_gather_pairs_kernel<<<grd, blk, 0, stream>>>(e_agg, nullptr, 0, 0,
                                                   z_partial, pair_pos,
                                                   H, K, B);

}

static inline void moe_gather_pairs_acc(float *e_agg, float *x_resid,
                                        int own_lo, int own_hi,
                                        const float *z_partial,
                                        const int *pair_pos, int H, int K,
                                        int B, hipStream_t stream) {
  dim3 blk(256), grd((H + blk.x - 1) / blk.x, B);
  moe_gather_pairs_kernel<<<grd, blk, 0, stream>>>(e_agg, x_resid,
                                                   own_lo, own_hi,
                                                   z_partial, pair_pos,
                                                   H, K, B);
}

// ===========================================================================
// Bo cuc LDS cua new_matmul_qkv_fused_kernel: KPack = 4 -> tile[k/4][hang][4]
//
// Mot O = 4 short = 8 byte = DUNG 4 phan tu k lien tiep ma mot lane can cho
// v_mfma_f32_16x16x16bf16_1k (A[i][k]: lane L thanh ghi r -> i = L%16,
// k = 4*(L/16)+r; B[k][j] doi xung). Nen:
//   doc : 1 x ds_read_b64   thay cho 4 x ds_read_u16
//   ghi : 2 x ds_write_b64  thay cho 8 x ds_write_b16 (mot uint4 tu HBM)
//
// Bank (32 bank x 4 byte = 128 byte/chu ky):
//   DOC  16 lane cua mot pha co CUNG plane k0 (Ak = 4*(lane>>4) khong doi) va
//        hang = row0 + (lane&15) chay lien tiep:
//          byte = C + 8*(lane&15) -> 128 byte lien tuc -> phu kin 32 bank dung
//          mot lan. Khong xung dot VOI MOI gia tri pad.
//   GHI  loader: a = threadIdx.x % (BK/8) la chunk k, b = threadIdx.x / (BK/8)
//        la hang. Trong mot pha 16 lane: a in [0,NKC), b in [0,16/NKC).
//        O ghi thu nhat co chi so o = (2a)*S + b, S = so o tren mot plane k0
//          -> byte = 16*S*a + 8*b  -> dword = 4*S*a + 2*b.
//        S = 128 + 2 = 130: 4*130 = 520 == 8 (mod 32) -> bank = 8a + 2b,
//          a in [0,4), b in [0,4) -> {0,2,...,30}: moi bank dung mot lan.
//          O thu hai lech them 2*S = 260 == 4 (mod 32) -> van la hoan vi.
//          => KHONG xung dot ca doc lan ghi.
//        S = 128 + 0 = 128: 4*128 = 512 == 0 (mod 32) -> bank = 2b: 4 chieu.
//        S = 128 + 4 = 132: 4*132 = 528 == 16 (mod 32) -> {0,16,0,16}: 2 chieu.
//        (Bo cuc [hang][k] "phang" cung lam doc sach, nhung ghi thi con 2 chieu
//         voi MOI pad; bo cuc o nay khong mat gi ma het xung dot ca hai chieu.)
//
// Can le: o 8 byte, goc tile can le 16 byte -> moi dia chi ds_read_b64 /
// ds_write_b64 deu can le 8 byte. b128 KHONG dung duoc o day (hai o cua mot
// uint4 cach nhau S o, khong ke nhau) - day la lua chon co y: b64 khong xung
// dot hon la b128 xung dot.
// ===========================================================================
constexpr bool getp_qkv_lds_write_conflict_free(int S, int NKC) {
  if (NKC <= 0 || NKC > 16 || 16 % NKC != 0) return false;
  int seen = 0;
  for (int l = 0; l < 16; ++l) {
    const int a = l % NKC;                 // chunk k (truc nhanh cua loader)
    const int b = l / NKC;                 // hang trong pha 16 lane
    int bank = (4 * S * a + 2 * b) % 32;
    if (bank < 0) bank += 32;
    if (bank & 1) return false;            // o phai can le 8 byte
    const int bit = 1 << (bank >> 1);
    if (seen & bit) return false;          // hai lane trung bank
    seen |= bit;
  }
  return seen == 0xFFFF;                   // phu kin 16 bank chan
}

// Pad tu suy ra: lay gia tri nho nhat hop le nen khong bao gio tra thua LDS.
// (hang=128, NKC=4) -> 2 ; (hang=128, NKC=2, tuc BK=16) -> 4. Nho vay bo cuc
// nay tong quat theo BK thay vi khoa cung mot hang so.
constexpr int getp_qkv_pick_pad_slots(int rows, int NKC) {
  for (int p = 0; p < 16; ++p)
    if (getp_qkv_lds_write_conflict_free(rows + p, NKC)) return p;
  return -1;
}

// Ke hoach LDS dung CHUNG cho than kernel va launcher.
template <int ROWS, int BK_, int PADS_>
struct GetpQkvLdsTile {
  static constexpr int KP   = 4;                  // short / o = 8 byte
  static constexpr int KSL  = BK_ / KP;           // so plane k0
  static constexpr int NKC  = BK_ / 8;            // so chunk k cua loader
  static constexpr int PADS =
      (PADS_ >= 0) ? PADS_ : getp_qkv_pick_pad_slots(ROWS, NKC);
  static constexpr int S     = ROWS + PADS;       // o tren mot plane k0
  static constexpr int slots = KSL * S;           // o cua ca tile
  static constexpr size_t bytes =
      (size_t)slots * KP * sizeof(unsigned short);

  static_assert(BK_ % 8 == 0, "loader nap uint4 = 8 short theo truc k");
  static_assert(BK_ % KP == 0, "BK phai chia het cho KPack = 4");
  static_assert(PADS >= 0,
                "khong pad nao lam ds_write_b64 het xung dot: phai doi loader");
  static_assert(getp_qkv_lds_write_conflict_free(S, NKC),
                "GETP_QKV_LDS_PAD_SLOTS sai: ds_write_b64 xung dot bank "
                "(ROWS=128: BK=32 -> dung 2, BK=16 -> dung 4)");
};

// Hinh khoi cua QKV: MOT ban duy nhat cho ca than kernel lan launcher. Truoc
// day launcher giu ban sao rieng va DA lech - "const int BK = 16;" o launcher
// trong khi than kernel dung BK = 32. Bien do chet (chi BM/BN vao grid) nen vo
// hai, nhung dung la lop loi "hang so cu song sot qua mot lan sua".
struct GetpQkvTile {
  static constexpr int BM = 128;
  static constexpr int BN = 128;
  static constexpr int BK = 32;
  static constexpr int WM = 64;
  static constexpr int WN = 64;
  static constexpr int THREADS = 256;
};

// Gop hai dword thanh mot o 4 x bf16. Thuan tuy doi kieu: hai VGPR da nam dung
// cho nen khong sinh lenh VALU nao (khac 8 lenh trich nua-word cua ban cu).
// __builtin_bit_cast la cach viet chuan, khong phai union punning.
__device__ __forceinline__ bf16x4 getp_qkv_pack2(unsigned int lo,
                                                 unsigned int hi) {
  struct Pair { unsigned int a, b; };
  static_assert(sizeof(Pair) == sizeof(bf16x4), "o phai la 8 byte");
  Pair p;
  p.a = lo;
  p.b = hi;
  return __builtin_bit_cast(bf16x4, p);
}

__global__ void __launch_bounds__(256)
new_matmul_qkv_fused_kernel(
  float *__restrict__ q_out, float *__restrict__ k_out,
  float *__restrict__ v_out, const __hip_bfloat16 *__restrict__ x,
  const __hip_bfloat16 *__restrict__ w, const __hip_bfloat16 *__restrict__ b,
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
  constexpr int BK = 32;
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

  constexpr int nbLoadsX = BM * (BK / 8) / BLOCK_SIZE;
  constexpr int nbLoadsW = BN * (BK / 8) / BLOCK_SIZE;

  const int loadXIdx_y = threadIdx.x / (BK / 8);
  const int loadXIdx_x = threadIdx.x % (BK / 8);
  const int loadWIdx_y = threadIdx.x / (BK / 8);
  const int loadWIdx_x = threadIdx.x % (BK / 8);
  constexpr int strideX = BLOCK_SIZE / (BK / 8);
  constexpr int strideW = BLOCK_SIZE / (BK / 8);

  assert(blockDim.x == BLOCK_SIZE);

  // Hinh khoi phai khop GetpQkvTile - nguon su that dung chung voi launcher.
  static_assert(BM == GetpQkvTile::BM && BN == GetpQkvTile::BN &&
                    BK == GetpQkvTile::BK && WM == GetpQkvTile::WM &&
                    WN == GetpQkvTile::WN &&
                    BLOCK_SIZE == GetpQkvTile::THREADS,
                "than kernel va GetpQkvTile lech nhau");

  // Bo cuc LDS [k/4][hang][4] - xem GetpQkvLdsTile. Ten xs/ws bi XOA co y: moi
  // cho chua chuyen doi se la loi bien dich chu khong phai dia chi sai im lang.
  using LdsX = GetpQkvLdsTile<BM, BK, GETP_QKV_LDS_PAD_SLOTS>;
  using LdsW = GetpQkvLdsTile<BN, BK, GETP_QKV_LDS_PAD_SLOTS>;
  constexpr int KP = LdsX::KP;    // 4 short / o
  constexpr int SX = LdsX::S;     // o tren mot plane k0, tile A
  constexpr int SW = LdsW::S;     // o tren mot plane k0, tile B
  static_assert(LdsX::KSL == LdsW::KSL, "hai tile phai cung so plane k0");
  static_assert(LdsX::bytes + LdsW::bytes <= 64 * 1024, "vuot 64 KB LDS / CU");

  // Afloat/Bfloat va bf16x4 deu la vector 4 x unsigned short (8 byte) nen gan
  // thang duoc; neu mot ngay nao do lech kieu thi day thanh loi bien dich.
  static_assert(sizeof(bf16x4) == sizeof(Afloat) &&
                    sizeof(bf16x4) == sizeof(Bfloat),
                "toan hang A/B cua MFMA phai la vector 8 byte");

  // __align__(16) la bao hiem: bf16x4 chi can 8 byte, nhung goc tile can le 16
  // byte lam MOI dia chi o chac chan can le 8 byte duoi mat LLVM, thay vi chi
  // "tinh ra la dung". Do la dieu kien de sinh ds_read_b64 / ds_write_b64.
  __shared__ __align__(16) bf16x4 xss[LdsX::slots];
  __shared__ __align__(16) bf16x4 wss[LdsW::slots];

#ifdef DEBUG_LDS_POISON
  // Do day LDS bang bf16 qNaN: o nao loader khong ghi se thanh NaN. Dung de
  // chia doi loi khi token agreement khong ra dung 100.00 %.
  {
    unsigned short *raw_x = reinterpret_cast<unsigned short *>(xss);
    unsigned short *raw_w = reinterpret_cast<unsigned short *>(wss);
    for (int i = threadIdx.x; i < LdsX::slots * KP; i += blockDim.x)
      raw_x[i] = 0x7FC0;
    for (int i = threadIdx.x; i < LdsW::slots * KP; i += blockDim.x)
      raw_w[i] = 0x7FC0;
  }
  __syncthreads();
#endif

  Afloat x_local;
  Bfloat w_local;

  CDfloat acc_local[nIterWaveM * nIterWaveN] = {0};

  // Load x and w from global memory to shared memory
  #pragma unroll
  for (int i = 0; i < nbLoadsX; ++i) {
    int offset = i * strideX;
    int index_x = 0 + 8 * loadXIdx_x;
    int index_y = BM * blockIdx.y + loadXIdx_y + offset;
    uint4 tmp = index_x < K && index_y < M ?
      *reinterpret_cast<const uint4 *>(&x[index_y * K + index_x]) : make_uint4(0, 0, 0, 0);
    // 8 short lien tiep theo k cua MOT hang = 2 o ke nhau theo truc plane k0.
    // Thu tu nua-word giu y nguyen (tmp.x thap = k+0, tmp.x cao = k+1, ...)
    // nen day la phep doi dia chi thuan tuy: bit-exact.
    bf16x4 *dst = &xss[((8 * loadXIdx_x) / KP) * SX + (loadXIdx_y + offset)];
    dst[0]  = getp_qkv_pack2(tmp.x, tmp.y);   // k+0 .. k+3
    dst[SX] = getp_qkv_pack2(tmp.z, tmp.w);   // k+4 .. k+7
  }
  #pragma unroll
  for (int i = 0; i < nbLoadsW; ++i) {
    int offset = i * strideW;
    int index_x = 0 + 8 * loadWIdx_x;
    int index_y = BN * blockIdx.x + loadWIdx_y + offset;
    uint4 tmp = index_x < K && index_y < N ?
      *reinterpret_cast<const uint4 *>(&w[index_y * K + index_x]) : uint4(0, 0, 0, 0);
    bf16x4 *dst = &wss[((8 * loadWIdx_x) / KP) * SW + (loadWIdx_y + offset)];
    dst[0]  = getp_qkv_pack2(tmp.x, tmp.y);   // k+0 .. k+3
    dst[SW] = getp_qkv_pack2(tmp.z, tmp.w);   // k+4 .. k+7
  }
  __syncthreads();
  
  #pragma unroll 1
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Prefetch
    uint4 regX[nbLoadsX];
    uint4 regW[nbLoadsW];
    if (bkIdx < K - BK) {
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        int offset = i * strideX;
        int index_x = bkIdx + BK + 8 * loadXIdx_x;
        int index_y = BM * blockIdx.y + loadXIdx_y + offset;
        regX[i] = index_x < K && index_y < M ?
          *reinterpret_cast<const uint4 *>(&x[index_y * K + index_x]) : make_uint4(0, 0, 0, 0);
      }
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        int offset = i * strideW;
        int index_x = bkIdx + BK + 8 * loadWIdx_x;
        int index_y = BN * blockIdx.x + loadWIdx_y + offset;
        regW[i] = index_x < K && index_y < N ?
          *reinterpret_cast<const uint4 *>(&w[index_y * K + index_x]) : uint4(0, 0, 0, 0);
      }
    }

    // Computing
    #pragma unroll
    for (int k = 0; k < BK; k += MFMA_K) {
      #pragma unroll
      for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
        #pragma unroll
        for (int iterWaveN = 0; iterWaveN < nIterWaveN; ++iterWaveN) {
          // basek = k + 4*(lane>>4) luon chia het cho 4, nen 4 phan tu k cua
          // mot hang nam gon trong MOT o -> 1 x ds_read_b64 thay cho 4 x
          // ds_read_u16. Giu Ak/Bk de anh xa lane cua MFMA van hien ro.
          x_local = xss[((k + Ak) / KP) * SX +
                        (yInBlockTile * WM + MFMA_M * iterWaveM + Ai)];
          w_local = wss[((k + Bk) / KP) * SW +
                        (xInBlockTile * WN + MFMA_N * iterWaveN + Bj)];

          CDfloat y_local = acc_local[iterWaveM * nIterWaveN + iterWaveN]; 
          y_local  = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(x_local, w_local, y_local, 0, 0, 0);
          acc_local[iterWaveM * nIterWaveN + iterWaveN] = y_local;
        }
      }
    }
    __syncthreads();
    if (bkIdx < K - BK) {
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        int offset = i * strideX;
        bf16x4 *dst = &xss[((8 * loadXIdx_x) / KP) * SX + (loadXIdx_y + offset)];
        dst[0]  = getp_qkv_pack2(regX[i].x, regX[i].y);
        dst[SX] = getp_qkv_pack2(regX[i].z, regX[i].w);
      }
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        int offset = i * strideW;
        bf16x4 *dst = &wss[((8 * loadWIdx_x) / KP) * SW + (loadWIdx_y + offset)];
        dst[0]  = getp_qkv_pack2(regW[i].x, regW[i].y);
        dst[SW] = getp_qkv_pack2(regW[i].z, regW[i].w);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
    #pragma unroll
    for (int iterWaveN = 0; iterWaveN < nIterWaveN; ++iterWaveN) {
      int acc_index = iterWaveM * nIterWaveN + iterWaveN;
      #pragma unroll
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
    float *q, float *k, float *v,
    const __hip_bfloat16 *x_bf16,
    const __hip_bfloat16 *w_qkv_bf16,
    const __hip_bfloat16 *b_qkv_bf16,
    int n, int head_dim, int n_attn_heads, int n_kv_heads, int batch_size,
    hipStream_t stream) {
      
  const int q_len = head_dim * n_attn_heads;
  const int k_len = head_dim * n_kv_heads;
  const int v_len = head_dim * n_kv_heads;

  const int M = batch_size;
  const int N = q_len + k_len + v_len;
  const int K = n;

  // MOT nguon su that cho hinh khoi: lay tu chinh hang so than kernel dung.
  // Ban cu o day co "const int BK = 16;" trong khi than kernel dung BK = 32 -
  // bien chet nen vo hai, nhung static_assert trong than kernel gio khoa cap
  // BM/BN/BK/WM/WN/THREADS lai voi nhau.
  const int BN = GetpQkvTile::BN;
  const int BM = GetpQkvTile::BM;

  dim3 block(GetpQkvTile::THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  new_matmul_qkv_fused_kernel<<<grid, block, 0, stream>>>(
      q, k, v, x_bf16, w_qkv_bf16, b_qkv_bf16, n, q_len, k_len, v_len, batch_size);
  
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

  // Calculate number of loads per thread
    constexpr int X_VEC = 8;
    const int loadX_x8 = threadIdx.x % (BK / X_VEC);
    const int loadX_y8 = threadIdx.x / (BK / X_VEC);
    constexpr int strideX8 = THREADS / (BK / X_VEC);
  constexpr int nbLoadsX = (BM + strideX8 - 1) / strideX8;
  constexpr int nbLoadsW = (BNt + strideW - 1) / strideW;
    
  // Initial load into first buffer
    #pragma unroll
  for (int i = 0; i < nbLoadsX; ++i) {
    const int off = i * strideX8;
      if (loadX_y8 + off >= BM) break;
      const int row = rowBase + loadX_y8 + off;
    const int kk = X_VEC * loadX_x8;
      
      if (row < M && kk + 7 < K) {
        const float* p = x + (size_t)row * K + kk;
        const float4 v1 = *reinterpret_cast<const float4*>(p);
        const float4 v2 = *reinterpret_cast<const float4*>(p + 4);
        
      sx0[(X_VEC * loadX_x8 + 0) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v1.x);
      sx0[(X_VEC * loadX_x8 + 1) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v1.y);
      sx0[(X_VEC * loadX_x8 + 2) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v1.z);
      sx0[(X_VEC * loadX_x8 + 3) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v1.w);
      sx0[(X_VEC * loadX_x8 + 4) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v2.x);
      sx0[(X_VEC * loadX_x8 + 5) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v2.y);
      sx0[(X_VEC * loadX_x8 + 6) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v2.z);
      sx0[(X_VEC * loadX_x8 + 7) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v2.w);
      } else {
        #pragma unroll
      for (int t = 0; t < X_VEC; ++t) sx0[(X_VEC * loadX_x8 + t) * BM + (loadX_y8 + off)] = 0;
      }
    }
    
    #pragma unroll
  for (int i = 0; i < nbLoadsW; ++i) {
    const int off = i * strideW;
      if (loadW_y + off >= BNt) break;
    const int kk = 8 * loadW_x;
      const int col = colBase + loadW_y + off;
    unsigned short *dst = sw0 + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
      
      if (col < N && kk + 7 < K) {
        const uint4* p = reinterpret_cast<const uint4*>(w + (size_t)col * K + kk);
        const uint4 t = *p;
        
        dst[0 * BNt] = (unsigned short)(t.x & 0xFFFF);
        dst[1 * BNt] = (unsigned short)(t.x >> 16);
        dst[2 * BNt] = (unsigned short)(t.y & 0xFFFF);
        dst[3 * BNt] = (unsigned short)(t.y >> 16);
        dst[4 * BNt] = (unsigned short)(t.z & 0xFFFF);
        dst[5 * BNt] = (unsigned short)(t.z >> 16);
        dst[6 * BNt] = (unsigned short)(t.w & 0xFFFF);
        dst[7 * BNt] = (unsigned short)(t.w >> 16);
      } else {
        #pragma unroll
        for (int t = 0; t < 8; ++t) dst[t * BNt] = 0;
      }
    }
  __syncthreads();

  #pragma unroll 1
  for (int bk = 0; bk < K; bk += BK) {
    unsigned short *sx = (bk / BK) & 1 ? sx1 : sx0;
    unsigned short *sw = (bk / BK) & 1 ? sw1 : sw0;
    unsigned short *sx_next = (bk / BK) & 1 ? sx0 : sx1;
    unsigned short *sw_next = (bk / BK) & 1 ? sw0 : sw1;
    
    // Prefetch next tile into registers
    float4 regX_v1[nbLoadsX];
    float4 regX_v2[nbLoadsX];
    uint4 regW[nbLoadsW];
    
    if (bk + BK < K) {
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        const int off = i * strideX8;
        if (loadX_y8 + off >= BM) break;
        const int row = rowBase + loadX_y8 + off;
        const int kk = bk + BK + X_VEC * loadX_x8;
        
        if (row < M && kk + 7 < K) {
          const float* p = x + (size_t)row * K + kk;
          regX_v1[i] = *reinterpret_cast<const float4*>(p);
          regX_v2[i] = *reinterpret_cast<const float4*>(p + 4);
        } else {
          regX_v1[i] = make_float4(0, 0, 0, 0);
          regX_v2[i] = make_float4(0, 0, 0, 0);
        }
      }
      
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        const int off = i * strideW;
        if (loadW_y + off >= BNt) break;
        const int kk = bk + BK + 8 * loadW_x;
        const int col = colBase + loadW_y + off;
        
        if (col < N && kk + 7 < K) {
          const uint4* p = reinterpret_cast<const uint4*>(w + (size_t)col * K + kk);
          regW[i] = *p;
        } else {
          regW[i] = make_uint4(0, 0, 0, 0);
        }
      }
    }
    
    // MFMA compute from shared memory
    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
      const int basek = k + yMF * 4;
      const int row = yTile * WM + xMF;
      const int c = xTile * WN + xMF;
      
      const unsigned short* sx_ptr = &sx[basek * BM + row];
      bf16x4 av = {sx_ptr[0], sx_ptr[BM], sx_ptr[2*BM], sx_ptr[3*BM]};
      
      #pragma unroll
      for (int g = 0; g < GETP_BN_AGG; ++g) {
        const unsigned short* sw_ptr = &sw[basek * BNt + c + g * BN];
        bf16x4 bv = {sw_ptr[0], sw_ptr[BNt], sw_ptr[2*BNt], sw_ptr[3*BNt]};
        
        d_acc[g] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, d_acc[g], 0, 0, 0);
      }
    }
    __syncthreads();
    
    // Store prefetched data to shared memory
    if (bk + BK < K) {
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        const int off = i * strideX8;
        if (loadX_y8 + off >= BM) break;
        
        sx_next[(X_VEC * loadX_x8 + 0) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v1[i].x);
        sx_next[(X_VEC * loadX_x8 + 1) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v1[i].y);
        sx_next[(X_VEC * loadX_x8 + 2) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v1[i].z);
        sx_next[(X_VEC * loadX_x8 + 3) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v1[i].w);
        sx_next[(X_VEC * loadX_x8 + 4) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v2[i].x);
        sx_next[(X_VEC * loadX_x8 + 5) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v2[i].y);
        sx_next[(X_VEC * loadX_x8 + 6) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v2[i].z);
        sx_next[(X_VEC * loadX_x8 + 7) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v2[i].w);
      }
      
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        const int off = i * strideW;
        if (loadW_y + off >= BNt) break;
        unsigned short *dst = sw_next + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
        
        dst[0 * BNt] = (unsigned short)(regW[i].x & 0xFFFF);
        dst[1 * BNt] = (unsigned short)(regW[i].x >> 16);
        dst[2 * BNt] = (unsigned short)(regW[i].y & 0xFFFF);
        dst[3 * BNt] = (unsigned short)(regW[i].y >> 16);
        dst[4 * BNt] = (unsigned short)(regW[i].z & 0xFFFF);
        dst[5 * BNt] = (unsigned short)(regW[i].z >> 16);
        dst[6 * BNt] = (unsigned short)(regW[i].w & 0xFFFF);
        dst[7 * BNt] = (unsigned short)(regW[i].w >> 16);
      }
    }
    __syncthreads();
  }

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  
  // AGGRESSIVE output: fully unrolled with pre-computed base
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = rowBase + yTile * WM + (yD + i);
    if (row >= M) continue;
    
    float* output_row = xout + (size_t)row * N;
    
    #pragma unroll
    for (int g = 0; g < GETP_BN_AGG; ++g) {
      const int col = colBase + xTile * WN + xD + g * BN;
      if (col >= N) continue;
      
      float v = d_acc[g][i];
      if (b) v += __bfloat162float(b[col]);
      output_row[col] = v;
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

  // Calculate number of loads per thread
    constexpr int X_VEC = 8;
    const int loadX_x8 = threadIdx.x % (BK / X_VEC);
    const int loadX_y8 = threadIdx.x / (BK / X_VEC);
    constexpr int strideX8 = THREADS / (BK / X_VEC);
  constexpr int nbLoadsX = (BM + strideX8 - 1) / strideX8;
  constexpr int nbLoadsW = (BNt + strideW - 1) / strideW;
    
  // Initial load into first buffer
  if (kBeg < kEnd) {
    #pragma unroll
    for (int i = 0; i < nbLoadsX; ++i) {
      const int off = i * strideX8;
      if (loadX_y8 + off >= BM) break;
      const int row = rowBase + loadX_y8 + off;
      const int kk = kBeg + X_VEC * loadX_x8;
      
      if (row < M && kk + 7 < kEnd) {
        const float* p = x + (size_t)row * K + kk;
        const float4 v1 = *reinterpret_cast<const float4*>(p);
        const float4 v2 = *reinterpret_cast<const float4*>(p + 4);
        
        sx0[(X_VEC * loadX_x8 + 0) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v1.x);
        sx0[(X_VEC * loadX_x8 + 1) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v1.y);
        sx0[(X_VEC * loadX_x8 + 2) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v1.z);
        sx0[(X_VEC * loadX_x8 + 3) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v1.w);
        sx0[(X_VEC * loadX_x8 + 4) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v2.x);
        sx0[(X_VEC * loadX_x8 + 5) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v2.y);
        sx0[(X_VEC * loadX_x8 + 6) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v2.z);
        sx0[(X_VEC * loadX_x8 + 7) * BM + (loadX_y8 + off)] = f32_to_bf16bits(v2.w);
      } else {
        #pragma unroll
        for (int t = 0; t < X_VEC; ++t) sx0[(X_VEC * loadX_x8 + t) * BM + (loadX_y8 + off)] = 0;
      }
    }
    
    #pragma unroll
    for (int i = 0; i < nbLoadsW; ++i) {
      const int off = i * strideW;
      if (loadW_y + off >= BNt) break;
      const int kk = kBeg + 8 * loadW_x;
      const int col = colBase + loadW_y + off;
      unsigned short *dst = sw0 + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
      
      if (col < N && kk + 7 < kEnd) {
        const uint4* p = reinterpret_cast<const uint4*>(w + (size_t)col * K + kk);
        const uint4 t = *p;
        
        dst[0 * BNt] = (unsigned short)(t.x & 0xFFFF);
        dst[1 * BNt] = (unsigned short)(t.x >> 16);
        dst[2 * BNt] = (unsigned short)(t.y & 0xFFFF);
        dst[3 * BNt] = (unsigned short)(t.y >> 16);
        dst[4 * BNt] = (unsigned short)(t.z & 0xFFFF);
        dst[5 * BNt] = (unsigned short)(t.z >> 16);
        dst[6 * BNt] = (unsigned short)(t.w & 0xFFFF);
        dst[7 * BNt] = (unsigned short)(t.w >> 16);
      } else {
        #pragma unroll
        for (int t2 = 0; t2 < 8; ++t2) dst[t2 * BNt] = 0;
      }
    }
  }
  __syncthreads();

  #pragma unroll 1
  for (int bk = kBeg; bk < kEnd; bk += BK) {
    unsigned short *sx = ((bk - kBeg) / BK) & 1 ? sx1 : sx0;
    unsigned short *sw = ((bk - kBeg) / BK) & 1 ? sw1 : sw0;
    unsigned short *sx_next = ((bk - kBeg) / BK) & 1 ? sx0 : sx1;
    unsigned short *sw_next = ((bk - kBeg) / BK) & 1 ? sw0 : sw1;
    
    // Prefetch next tile into registers
    float4 regX_v1[nbLoadsX];
    float4 regX_v2[nbLoadsX];
    uint4 regW[nbLoadsW];
    
    if (bk + BK < kEnd) {
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        const int off = i * strideX8;
        if (loadX_y8 + off >= BM) break;
        const int row = rowBase + loadX_y8 + off;
        const int kk = bk + BK + X_VEC * loadX_x8;
        
        if (row < M && kk + 7 < kEnd) {
          const float* p = x + (size_t)row * K + kk;
          regX_v1[i] = *reinterpret_cast<const float4*>(p);
          regX_v2[i] = *reinterpret_cast<const float4*>(p + 4);
        } else {
          regX_v1[i] = make_float4(0, 0, 0, 0);
          regX_v2[i] = make_float4(0, 0, 0, 0);
        }
      }
      
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        const int off = i * strideW;
        if (loadW_y + off >= BNt) break;
        const int kk = bk + BK + 8 * loadW_x;
        const int col = colBase + loadW_y + off;
        
        if (col < N && kk + 7 < kEnd) {
          const uint4* p = reinterpret_cast<const uint4*>(w + (size_t)col * K + kk);
          regW[i] = *p;
        } else {
          regW[i] = make_uint4(0, 0, 0, 0);
        }
      }
    }

    // MFMA compute from shared memory
    #pragma unroll
    for (int k = 0; k < BK; k += MFMA_K) {
      const int basek = k + yMF * 4;
      const int row = yTile * WM + xMF;
      const int c = xTile * WN + xMF;
      
      const unsigned short* sx_ptr = &sx[basek * BM + row];
      bf16x4 av = {sx_ptr[0], sx_ptr[BM], sx_ptr[2*BM], sx_ptr[3*BM]};
      
      #pragma unroll
      for (int g = 0; g < BN_AGG; ++g) {
        const unsigned short* sw_ptr = &sw[basek * BNt + c + g * BN];
        bf16x4 bv = {sw_ptr[0], sw_ptr[BNt], sw_ptr[2*BNt], sw_ptr[3*BNt]};
        
        d_acc[g] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, d_acc[g], 0, 0, 0);
      }
    }
    __syncthreads();
    
    // Store prefetched data to shared memory
    if (bk + BK < kEnd) {
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        const int off = i * strideX8;
        if (loadX_y8 + off >= BM) break;
        
        sx_next[(X_VEC * loadX_x8 + 0) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v1[i].x);
        sx_next[(X_VEC * loadX_x8 + 1) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v1[i].y);
        sx_next[(X_VEC * loadX_x8 + 2) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v1[i].z);
        sx_next[(X_VEC * loadX_x8 + 3) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v1[i].w);
        sx_next[(X_VEC * loadX_x8 + 4) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v2[i].x);
        sx_next[(X_VEC * loadX_x8 + 5) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v2[i].y);
        sx_next[(X_VEC * loadX_x8 + 6) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v2[i].z);
        sx_next[(X_VEC * loadX_x8 + 7) * BM + (loadX_y8 + off)] = f32_to_bf16bits(regX_v2[i].w);
      }
      
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        const int off = i * strideW;
        if (loadW_y + off >= BNt) break;
        unsigned short *dst = sw_next + (size_t)(8 * loadW_x) * BNt + (loadW_y + off);
        
        dst[0 * BNt] = (unsigned short)(regW[i].x & 0xFFFF);
        dst[1 * BNt] = (unsigned short)(regW[i].x >> 16);
        dst[2 * BNt] = (unsigned short)(regW[i].y & 0xFFFF);
        dst[3 * BNt] = (unsigned short)(regW[i].y >> 16);
        dst[4 * BNt] = (unsigned short)(regW[i].z & 0xFFFF);
        dst[5 * BNt] = (unsigned short)(regW[i].z >> 16);
        dst[6 * BNt] = (unsigned short)(regW[i].w & 0xFFFF);
        dst[7 * BNt] = (unsigned short)(regW[i].w >> 16);
      }
    }
    __syncthreads();
  }

  const size_t slice_stride = (size_t)M * N;
  const size_t base_out = (size_t)blockIdx.z * slice_stride;

  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  
  // AGGRESSIVE split-K output: fully unrolled with base pointer optimization
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = rowBase + yTile * WM + (yD + i);
    if (row >= M) continue;
    
    float* output_row = partial + base_out + (size_t)row * N;
    
    #pragma unroll
    for (int g = 0; g < BN_AGG; ++g) {
      const int col = colBase + xTile * WN + xD + g * BN;
      if (col < N) output_row[col] = d_acc[g][i];
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
  
  dim3 blockDim(16, 16);
  dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x,
               (n_heads + blockDim.y - 1) / blockDim.y, batch_size);
  apply_rotary_emb_kernel<<<gridDim, blockDim, 0, stream>>>(x, cos, sin,
                                                            n_heads, head_dim);
  
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
  
  const int BLK = 1024;
  const dim3 block(BLK);
  const dim3 grid(batch_size);
  const size_t shmem = (size_t)n_experts * sizeof(float) +
                       (size_t)BLK * (sizeof(float) + sizeof(int)) +
                       GETP_ROUTER_TOPK_MAXK * (sizeof(float) + sizeof(int));
  router_topk_softmax_batch_kernel<<<grid, block, shmem, stream>>>(
      router_score, n_experts, experts_per_token, topk_v_out, topk_i_out);
  
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
  
  const int BLK = 128, GRD = (B + BLK - 1) / BLK;
  map_global_to_local_batch_kernel<<<GRD, BLK, 0, stream>>>(
      topk_i, topk_v, local_ids, local_wts, n_local, K, expert_start,
      expert_end, B);
  
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
  
  dim3 blockDim(1024);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x, batch_size);
  vecadd_kernel<<<gridDim, blockDim, 0, stream>>>(x, y, size);
  
}

// ============================================================================
// Attention giai ma dung mfma_f32_16x16x16bf16_1k.
//
// Ban cu dat truc K cua phep nhan vao 16 BLOCK cua mfma 4x4x4, nen sau moi lenh
// MFMA con phai gop 16 block lai bang 4 vong __shfl_down roi phat tan diem so
// bang 16 lan __shfl nua: 128 lenh cheo lane va 80 lenh __expf cho moi 16 key,
// doi lay dung 4 lenh MFMA. Phep thu dong chay (thay toan bo phan tinh bang mot
// phep cong duy nhat, giu nguyen duong nap HBM + LDS + barrier) do duoc 56%
// thoi gian kernel nam o phan tinh do - tuc no dang la cho ton, khong phai
// dong du lieu.
//
// Ban nay dat KEY tren truc M va HEAD tren truc N:
//
//     S^T = K   . Q^T    A = K  [16 key   x 16 chieu]  B = Q^T [16 chieu x 16 head]
//     O^T = V^T . P^T    A = V^T[16 chieu x 16 key]    B = P^T [16 key   x 16 head]
//
// Diem mau chot: bo cuc ket qua D cua MFMA (lane giu D[4*(lane>>4)+m][lane&15])
// TRUNG dung bo cuc toan hang B. Nen P^T di thang tu MFMA thu nhat sang MFMA
// thu hai - khong chuyen vi, khong mot lenh cheo lane nao. Ca tile 16 key chi
// con 2 lenh __shfl_xor (gop max theo nhom g) va 5 lenh exp.
//
// Gia phai tra: moi wave chi nam HPW head that trong 16 cot cua truc N, so cot
// con lai dien 0. HPW = 4 giu nguyen cach chia head cu (2 wave/block, 25% cot
// co ich); HPW = 8 cho mot wave om ca 8 head cua mot kv_head - mot nua so lenh
// MFMA nhung cung mot nua so thread nap du lieu. Chon bang co bien dich.
// ============================================================================
#ifndef FLASH_HEADS_PER_WAVE
#define FLASH_HEADS_PER_WAVE 8
#endif
#ifndef FLASH_LDS_PAD
#define FLASH_LDS_PAD 4
#endif
// Tach P thanh hai bf16 (phan cao + phan du) roi cong bang HAI lenh MFMA, cho
// tich PV co do chinh xac gan fp32 nhu ban cu. Dung de kiem chung gia thuyet
// "lam tron P la nguyen nhan lech"; chi giu neu that su la nguyen nhan, vi no
// them 4 lenh MFMA moi tile.
#ifndef FLASH_P_HILO
#define FLASH_P_HILO 0
#endif

template <int HPW, bool APPLY_MASK>
__global__ void __launch_bounds__(64 * (8 / HPW))
flash_attn_decode_mfma16_kernel(
  __hip_bfloat16 *__restrict__ tb_bf16, float *__restrict__ tb_f32,
  const __hip_bfloat16 *__restrict__ key_cache,
  const __hip_bfloat16 *__restrict__ value_cache, const float *__restrict__ q,
  const int *__restrict__ mask_on,
  const float *__restrict__ attn_sinks, int head_dim, int n_attn_heads,
  int n_kv_heads, int pos, int batch_size, int kv_dim, int kv_mul,
  int sliding_window, int cache_tcap, float inv_sqrt_d
) {
  constexpr int HEAD_DIM   = 64;
  constexpr int TILE       = 16;                 // = M = N = K cua MFMA
  constexpr int NCHUNK     = HEAD_DIM / TILE;    // 4 chunk 16 chieu
  constexpr int KV_MUL     = 8;
  constexpr int WAVES      = KV_MUL / HPW;
  constexpr int BLOCK_SIZE = 64 * WAVES;
  constexpr int PAD        = FLASH_LDS_PAD;

  // PAD = 4 -> hang dai 68 ushort = 136 byte = 34 bank; 34 mod 32 = 2 nen 16 lane
  // doc b64 phu kin dung mot lan 32 bank, khong xung dot. Doi lai hang chi con
  // can 8 byte, nen phan nap ghi hai lenh b64 thay vi mot lenh b128.
  static_assert(HPW == 4 || HPW == 8, "HPW chi ho tro 4 hoac 8");
  static_assert(PAD % 4 == 0, "PAD phai chia het 4 de hang con can 8 byte");
  static_assert(TILE * (HEAD_DIM / 8) % BLOCK_SIZE == 0, "nap KV khong chia deu");

  using bf16x4 = __attribute__((__vector_size__(8)))  unsigned short;
  using f32x4  = __attribute__((__vector_size__(16))) float;

  assert(head_dim == HEAD_DIM);
  assert(kv_mul == KV_MUL);
  assert(blockDim.x == BLOCK_SIZE);

  const int waveIdx = threadIdx.x / warpSize;
  const int lane    = threadIdx.x % warpSize;
  const int ln      = lane & 15;   // truc M cua A, truc N cua B va D
  const int g       = lane >> 4;   // truc K cua A va B, nhom hang cua D

  const int kv_h = blockIdx.x;
  const int b    = blockIdx.y;
  if (!mask_on[b]) return;

  const int t_start = APPLY_MASK ? MAX(0, pos - sliding_window + 1)
                                 : MAX(0, pos + 1 - cache_tcap);
  const int n_steps = pos + 1 - t_start;
  const int t_end   = t_start + n_steps;

  __shared__ __align__(16) unsigned short ks[TILE][HEAD_DIM + PAD];
  __shared__ __align__(16) unsigned short vs[TILE][HEAD_DIM + PAD];
  __shared__ float ot[WAVES][HPW][HEAD_DIM];

  // ---- Q^T lam toan hang B: lane giu B[4g+kk][ln] = q[head ln][chunk*16+4g+kk].
  // Cot ln >= HPW khong ung voi head nao nen dien 0; diem so cua chung bang 0
  // (huu han, khong sinh NaN) va khong bao gio duoc ghi ra. Chi so head duoc
  // kep lai de con tro khong bao gio ra ngoai mang q.
  const int hg_self = kv_h * KV_MUL + waveIdx * HPW + (ln < HPW ? ln : 0);
  bf16x4 qB[NCHUNK];
  {
    const float *qrow =
        q + (size_t)b * n_attn_heads * head_dim + (size_t)hg_self * head_dim;
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        qB[c][kk] = (ln < HPW)
            ? f32_to_bf16bits(inv_sqrt_d * qrow[c * TILE + 4 * g + kk])
            : (unsigned short)0;
    }
  }

  constexpr int nbLoadsKV = TILE * (HEAD_DIM / 8) / BLOCK_SIZE;
  constexpr int strideKV  = BLOCK_SIZE / (HEAD_DIM / 8);
  const int ly = threadIdx.x / (HEAD_DIM / 8);
  const int lx = threadIdx.x % (HEAD_DIM / 8);

  // Hang LDS chi can 8 byte (136 = 8 x 17) nen mot lenh b128 se lech can chinh;
  // hai lenh b64 chuyen du 16 byte ma van hop le.
  auto put16 = [](unsigned short *dst, uint4 v) {
    *reinterpret_cast<uint2 *>(dst)     = make_uint2(v.x, v.y);
    *reinterpret_cast<uint2 *>(dst + 4) = make_uint2(v.z, v.w);
  };

  for (int i = 0; i < nbLoadsKV; ++i) {
    const int offset  = i * strideKV;
    const int index_y = t_start + ly + offset;
    if (index_y < t_end) {
      const size_t off = (size_t)(index_y % cache_tcap) * batch_size * kv_dim +
                         (size_t)b * kv_dim + (size_t)kv_h * head_dim + 8 * lx;
      put16(&ks[ly + offset][lx * 8], *reinterpret_cast<const uint4 *>(key_cache + off));
      put16(&vs[ly + offset][lx * 8], *reinterpret_cast<const uint4 *>(value_cache + off));
    } else {
      put16(&ks[ly + offset][lx * 8], make_uint4(0, 0, 0, 0));
      put16(&vs[ly + offset][lx * 8], make_uint4(0, 0, 0, 0));
    }
  }
  __syncthreads();

  f32x4 acc_o[NCHUNK];
  #pragma unroll
  for (int c = 0; c < NCHUNK; ++c) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc_o[c][i] = 0.f;
  }
  float m_run = -INFINITY;
  float l_run = 0.f;

  for (int bt = t_start; bt < t_end; bt += TILE) {
    uint4 regK[nbLoadsKV];
    uint4 regV[nbLoadsKV];
    const bool has_next = (bt + TILE < t_end);
    if (has_next) {
      #pragma unroll
      for (int i = 0; i < nbLoadsKV; ++i) {
        const int index_y = bt + TILE + ly + i * strideKV;
        if (index_y < t_end) {
          const size_t off = (size_t)(index_y % cache_tcap) * batch_size * kv_dim +
                             (size_t)b * kv_dim + (size_t)kv_h * head_dim + 8 * lx;
          regK[i] = *reinterpret_cast<const uint4 *>(key_cache + off);
          regV[i] = *reinterpret_cast<const uint4 *>(value_cache + off);
        } else {
          regK[i] = regV[i] = make_uint4(0, 0, 0, 0);
        }
      }
    }

    // ---- S^T = K . Q^T : A[key][chieu] doc mot lenh b64 moi chunk
    f32x4 st;
    #pragma unroll
    for (int i = 0; i < 4; ++i) st[i] = 0.f;
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
      bf16x4 kA;
      *reinterpret_cast<uint2 *>(&kA) =
          *reinterpret_cast<const uint2 *>(&ks[ln][c * TILE + 4 * g]);
      st = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(kA, qB[c], st, 0, 0, 0);
    }

    // ---- mat na duoi. Hang ks ngoai pham vi bi dien 0 nen cho diem 0, ma
    // exp(0 - m) > 0 se lam hong tong -> phai ep ve -INFINITY.
    const int kbase = bt + 4 * g;
    #pragma unroll
    for (int m = 0; m < 4; ++m)
      if (kbase + m >= t_end) st[m] = -INFINITY;

    // ---- max cua tile, dung chung cho ca 4 nhom g cua cung mot head. Bat buoc
    // phai gop truoc khi tinh m_new: mot nhom g co the khong co key hop le nao
    // (m_tile = -INF) va se sinh NaN neu tu tinh lay. Sau khi gop thi moi tile
    // chac chan co it nhat mot key hop le nen m_tile huu han.
    float m_tile = fmaxf(fmaxf(st[0], st[1]), fmaxf(st[2], st[3]));
    m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 16));
    m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 32));

    const float m_new = fmaxf(m_run, m_tile);
    const float alpha = __expf(m_run - m_new);

    // ---- P^T = exp(S^T - m_new), lam tron bf16 de lam toan hang B ngay.
    bf16x4 pB;
#if FLASH_P_HILO
    bf16x4 pLo;
#endif
    float lsum = 0.f;
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float e = __expf(st[m] - m_new);   // = 0 khi st[m] = -INF
      lsum += e;
      pB[m] = f32_to_bf16bits(e);
#if FLASH_P_HILO
      pLo[m] = f32_to_bf16bits(e - __bfloat162float(bf16bits_to_bf16(pB[m])));
#endif
    }

    // l giu rieng theo nhom g; vi ca 4 nhom dung chung m_new nen cong lai mot
    // lan o phan ket la du - gop moi tile la thua.
    l_run = l_run * alpha + lsum;
    m_run = m_new;

    // ---- O^T += V^T . P^T
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
      bf16x4 vA;
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk) vA[kk] = vs[4 * g + kk][c * TILE + ln];
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc_o[c][i] *= alpha;
      acc_o[c] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(vA, pB, acc_o[c], 0, 0, 0);
#if FLASH_P_HILO
      acc_o[c] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(vA, pLo, acc_o[c], 0, 0, 0);
#endif
    }

    __syncthreads();
    if (has_next) {
      #pragma unroll
      for (int i = 0; i < nbLoadsKV; ++i) {
        put16(&ks[ly + i * strideKV][lx * 8], regK[i]);
        put16(&vs[ly + i * strideKV][lx * 8], regV[i]);
      }
    }
    __syncthreads();
  }

  // ---- gop l qua 4 nhom g (mot lan duy nhat)
  l_run += __shfl_xor(l_run, 16);
  l_run += __shfl_xor(l_run, 32);

  if (ln < HPW) {
    const float sink  = attn_sinks[hg_self];
    const float m_fin = fmaxf(m_run, sink);
    const float a     = __expf(m_run - m_fin);
    const float l     = l_run * a + __expf(sink - m_fin);
    const float scale = a / l;
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c)
      *reinterpret_cast<float4 *>(&ot[waveIdx][ln][c * TILE + 4 * g]) =
          make_float4(acc_o[c][0] * scale, acc_o[c][1] * scale,
                      acc_o[c][2] * scale, acc_o[c][3] * scale);
  }
  __syncthreads();

  // ---- chuyen vi qua LDS roi ghi lien mach: moi head 256 byte lien nhau,
  // thay cho 16 lane ghi rai rac sang cac hang head khac nhau.
  // Ghi THANG bf16: consumer duy nhat cua tb la matmul attn_o, va no chi doc
  // bf16. Viec lam tron nam TRONG lane - moi lane dang giu dung 4 float lien
  // tiep cua mot head - nen khong can them mot trao doi LDS nao. Cap so hoc
  // giong het fp32_to_bf16_kernel: no doc lai chinh 4 float nay tu HBM roi goi
  // f32_to_bf16bits. Mot vong doc-ghi HBM cua ca tensor bien mat.
  unsigned short *obase = reinterpret_cast<unsigned short *>(tb_bf16) +
                          (size_t)b * n_attn_heads * head_dim;
  #pragma unroll
  for (int idx = lane; idx < HPW * HEAD_DIM / 4; idx += 64) {
    const int hh = idx / (HEAD_DIM / 4);
    const int d4 = idx % (HEAD_DIM / 4);
    const float4 v = *reinterpret_cast<const float4 *>(&ot[waveIdx][hh][d4 * 4]);
    uint2 packed;
    packed.x = (unsigned)f32_to_bf16bits(v.x) |
               ((unsigned)f32_to_bf16bits(v.y) << 16);
    packed.y = (unsigned)f32_to_bf16bits(v.z) |
               ((unsigned)f32_to_bf16bits(v.w) << 16);
    const size_t off =
        (size_t)(kv_h * KV_MUL + waveIdx * HPW + hh) * head_dim + d4 * 4;
    *reinterpret_cast<uint2 *>(obase + off) = packed;
#if GETP_DUMP_ATTN
    // Duong go roi: van giu ban fp32 de getp_dump_attn so sanh duoc.
    *reinterpret_cast<float4 *>(
        tb_f32 + (size_t)b * n_attn_heads * head_dim + off) = v;
#endif
  }
#if !GETP_DUMP_ATTN
  (void)tb_f32;
#endif
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
  constexpr int MFMA_K = 4;
  constexpr int MFMA_BLOCK = 16;

  constexpr int GPRs_A = 2;
  constexpr int GPRs_B = 2;
  constexpr int GPRs_CD = 4;

  using Afloat  = __attribute__( (__vector_size__(GPRs_A  * sizeof(float)) )) unsigned short;
  using Bfloat  = __attribute__( (__vector_size__(GPRs_B  * sizeof(float)) )) unsigned short;
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

  const int nbLoadsKV = TILE_T * (HEAD_DIM / 8) / BLOCK_SIZE;

  const int kv_h = blockIdx.x;
  const int b = blockIdx.y;

  if (!mask_on[b]) return;

  const int loadKVIdx_y = threadIdx.x / (HEAD_DIM / 8);
  const int loadKVIdx_x = threadIdx.x % (HEAD_DIM / 8);
  constexpr int strideKV = BLOCK_SIZE / (HEAD_DIM / 8);

  const int t_start = MAX(0, pos - sliding_window + 1);
  const int n_steps = pos + 1 - t_start;

  __shared__ __align__(16) unsigned short ks[TILE_T][HEAD_DIM];
  __shared__ __align__(16) __hip_bfloat16 vs[TILE_T][HEAD_DIM];

  float out_values[MFMA_M] = {0.f};
  float m_values[MFMA_M] = {-INFINITY};
  float l_values[MFMA_M] = {0.f};

  const float *qptr =
      q + (size_t)b * n_attn_heads * head_dim;
  
  Afloat q_values[HEAD_DIM / (MFMA_BLOCK * MFMA_K)];
  for (int i = 0; i < HEAD_DIM / (MFMA_BLOCK * MFMA_K); ++i) {
    for (int j = 0; j < MFMA_K; ++j) {
      // inv_sqrt_d gap vao day thay vi nhan vao tung diem so sau __shfl,
      // noi gia tri dong nhat ca wave nen 63/64 lane bi phi. Bit-exact vi
      // head_dim == 64 -> inv_sqrt_d = 2^-3.
      q_values[i][j] = 
        f32_to_bf16bits(
          inv_sqrt_d *
          qptr[(Ai + MFMA_M * waveIdx + kv_h * kv_mul) * head_dim + 
                  i * (MFMA_BLOCK * MFMA_K) + 
                  Ablock * MFMA_K + j]);
    }
  }

  // Load data to shared memory
  for (int i = 0; i < nbLoadsKV; ++i) {
    int offset = i * strideKV;
    int index_x = 8 * loadKVIdx_x;
    int index_y = t_start + loadKVIdx_y + offset;
    if (index_y < t_start + n_steps) {
      const __hip_bfloat16 *kptr =
          key_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
          (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
      const __hip_bfloat16 *vptr =
          value_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
          (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;

      uint4 tmp_k = *reinterpret_cast<const uint4 *>(kptr);
      uint4 tmp_v = *reinterpret_cast<const uint4 *>(vptr);

      // Mot ghi LDS 128-bit thay cho 8 ghi 16-bit: 8 short nay lien nhau tren
      // truc thu hai va cot bat dau luon chia het 16 byte.
      *reinterpret_cast<uint4 *>(&ks[loadKVIdx_y + offset][loadKVIdx_x * 8]) = tmp_k;

      *reinterpret_cast<uint4 *>(&vs[loadKVIdx_y + offset][loadKVIdx_x * 8]) = tmp_v;
    }
    else {
      *reinterpret_cast<uint4 *>(&ks[loadKVIdx_y + offset][loadKVIdx_x * 8]) = make_uint4(0, 0, 0, 0);

      *reinterpret_cast<uint4 *>(&vs[loadKVIdx_y + offset][loadKVIdx_x * 8]) = make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();

  for (int btIdx = t_start; btIdx < t_start + n_steps; btIdx += TILE_T) {
    // Prefetch
    uint4 regK[nbLoadsKV];
    uint4 regV[nbLoadsKV];
    if (btIdx + TILE_T < t_start + n_steps) {
      for (int i = 0; i < nbLoadsKV; ++i) {
        int offset = i * strideKV;
        int index_x = 8 * loadKVIdx_x;
        int index_y = btIdx + TILE_T + loadKVIdx_y + offset;
        if (index_y < t_start + n_steps) {
          const __hip_bfloat16 *kptr =
              key_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
              (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
          const __hip_bfloat16 *vptr =
              value_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
              (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;

          uint4 tmp_k = *reinterpret_cast<const uint4 *>(kptr);
          uint4 tmp_v = *reinterpret_cast<const uint4 *>(vptr);

          regK[i] = tmp_k;
          regV[i] = tmp_v;
        }
        else {
          regK[i] = regV[i] = make_uint4(0, 0, 0, 0);
        }
      }
    }

    // Computing
    for (int t = 0; t < TILE_T; t += MFMA_N) {
      CDfloat acc = {0};
      for (int dimIdx = 0; dimIdx < HEAD_DIM; dimIdx += MFMA_BLOCK * MFMA_K) {
        // float k_value = __bfloat162float(ks[t + Bj][dimIdx + Ablock]);
        Bfloat k_value;
        for (int i = 0; i < MFMA_K; ++i) {
          k_value[i] = ks[t + Bj][dimIdx + Ablock * MFMA_K + i];
        }
        acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(q_values[dimIdx / (MFMA_BLOCK * MFMA_K)], k_value, acc, 0, 0, 0);
      }

      for (int width = warpSize / 2; width >= MFMA_N; width /= 2) {
        for (int i = 0; i < GPRs_CD; ++i) {
          acc[i] += __shfl_down(acc[i], width);
        }
      }

      // Softmax truc tuyen theo TILE thay vi theo tung key. Voi moi i: mot lan
      // rescale accumulator va 1 + MFMA_N lan exp, thay cho 4 rescale va 8 exp.
      int nvalid = 0;
      #pragma unroll
      for (int j = 0; j < MFMA_N; ++j)
        if (btIdx + t + j < t_start + n_steps) ++nvalid;

      if (nvalid > 0) {
        #pragma unroll
        for (int i = 0; i < MFMA_M; ++i) {
          float s[MFMA_N];
          float m_tile = -INFINITY;
          #pragma unroll
          for (int j = 0; j < MFMA_N; ++j) {
            // Hang ks ngoai pham vi bi zero-pad nen cho diem 0, ma exp(0) = 1
            // se lam hong tong -> phai ep ve -INFINITY.
            s[j] = (j < nvalid) ? (__shfl(acc[i], j)) : -INFINITY;
            m_tile = fmaxf(m_tile, s[j]);
          }
          const float m_new = fmaxf(m_values[i], m_tile);
          const float alpha = __expf(m_values[i] - m_new);
          float lsum = 0.f;
          float osum = 0.f;
          #pragma unroll
          for (int j = 0; j < MFMA_N; ++j) {
            const float e = __expf(s[j] - m_new);   // = 0 khi s[j] = -INF
            lsum += e;
            osum += e * __bfloat162float(vs[t + j][laneIdx]);
          }
          l_values[i] = l_values[i] * alpha + lsum;
          m_values[i] = m_new;
          out_values[i] = alpha * out_values[i] + osum;
        }
      }
    }

    __syncthreads();

    if (btIdx + TILE_T < t_start + n_steps) {
      for (int i = 0; i < nbLoadsKV; ++i) {
        int offset = i * strideKV;
        int index_x = 8 * loadKVIdx_x;
        int index_y = btIdx + TILE_T + loadKVIdx_y + offset;

        uint4 tmp_k = regK[i];
        uint4 tmp_v = regV[i];

        // Mot ghi LDS 128-bit thay cho 8 ghi 16-bit: 8 short nay lien nhau tren
        // truc thu hai va cot bat dau luon chia het 16 byte.
        *reinterpret_cast<uint4 *>(&ks[loadKVIdx_y + offset][loadKVIdx_x * 8]) = tmp_k;

        *reinterpret_cast<uint4 *>(&vs[loadKVIdx_y + offset][loadKVIdx_x * 8]) = tmp_v;
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
  constexpr int MFMA_K = 4;
  constexpr int MFMA_BLOCK = 16;

  constexpr int GPRs_A = 2;
  constexpr int GPRs_B = 2;
  constexpr int GPRs_CD = 4;

  using Afloat  = __attribute__( (__vector_size__(GPRs_A  * sizeof(float)) )) unsigned short;
  using Bfloat  = __attribute__( (__vector_size__(GPRs_B  * sizeof(float)) )) unsigned short;
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

  const int nbLoadsKV = TILE_T * (HEAD_DIM / 8) / BLOCK_SIZE;

  const int kv_h = blockIdx.x;
  const int b = blockIdx.y;

  if (!mask_on[b]) return;

  const int loadKVIdx_y = threadIdx.x / (HEAD_DIM / 8);
  const int loadKVIdx_x = threadIdx.x % (HEAD_DIM / 8);
  constexpr int strideKV = BLOCK_SIZE / (HEAD_DIM / 8);

  const int t_start = MAX(0, pos + 1 - cache_tcap);
  const int n_steps = pos + 1 - t_start;

  __shared__ __align__(16) unsigned short ks[TILE_T][HEAD_DIM];
  __shared__ __align__(16) __hip_bfloat16 vs[TILE_T][HEAD_DIM];

  float out_values[MFMA_M] = {0.f};
  float m_values[MFMA_M] = {-INFINITY};
  float l_values[MFMA_M] = {0.f};

  const float *qptr =
      q + (size_t)b * n_attn_heads * head_dim;
  
  Afloat q_values[HEAD_DIM / (MFMA_BLOCK * MFMA_K)];
  for (int i = 0; i < HEAD_DIM / (MFMA_BLOCK * MFMA_K); ++i) {
    for (int j = 0; j < MFMA_K; ++j) {
      // inv_sqrt_d gap vao day thay vi nhan vao tung diem so sau __shfl,
      // noi gia tri dong nhat ca wave nen 63/64 lane bi phi. Bit-exact vi
      // head_dim == 64 -> inv_sqrt_d = 2^-3.
      q_values[i][j] = 
        f32_to_bf16bits(
          inv_sqrt_d *
          qptr[(Ai + MFMA_M * waveIdx + kv_h * kv_mul) * head_dim + 
                  i * (MFMA_BLOCK * MFMA_K) + 
                  Ablock * MFMA_K + j]);
    }
  }

  // Load data to shared memory
  for (int i = 0; i < nbLoadsKV; ++i) {
    int offset = i * strideKV;
    int index_x = 8 * loadKVIdx_x;
    int index_y = t_start + loadKVIdx_y + offset;
    if (index_y < t_start + n_steps) {
      const __hip_bfloat16 *kptr =
          key_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
          (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
      const __hip_bfloat16 *vptr =
          value_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
          (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;

      uint4 tmp_k = *reinterpret_cast<const uint4 *>(kptr);
      uint4 tmp_v = *reinterpret_cast<const uint4 *>(vptr);

      // Mot ghi LDS 128-bit thay cho 8 ghi 16-bit: 8 short nay lien nhau tren
      // truc thu hai va cot bat dau luon chia het 16 byte.
      *reinterpret_cast<uint4 *>(&ks[loadKVIdx_y + offset][loadKVIdx_x * 8]) = tmp_k;

      *reinterpret_cast<uint4 *>(&vs[loadKVIdx_y + offset][loadKVIdx_x * 8]) = tmp_v;
    }
    else {
      *reinterpret_cast<uint4 *>(&ks[loadKVIdx_y + offset][loadKVIdx_x * 8]) = make_uint4(0, 0, 0, 0);

      *reinterpret_cast<uint4 *>(&vs[loadKVIdx_y + offset][loadKVIdx_x * 8]) = make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();

  for (int btIdx = t_start; btIdx < t_start + n_steps; btIdx += TILE_T) {
    // Prefetch
    uint4 regK[nbLoadsKV];
    uint4 regV[nbLoadsKV];
    if (btIdx + TILE_T < t_start + n_steps) {
      for (int i = 0; i < nbLoadsKV; ++i) {
        int offset = i * strideKV;
        int index_x = 8 * loadKVIdx_x;
        int index_y = btIdx + TILE_T + loadKVIdx_y + offset;
        if (index_y < t_start + n_steps) {
          const __hip_bfloat16 *kptr =
              key_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
              (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;
          const __hip_bfloat16 *vptr =
              value_cache + (index_y % cache_tcap) * (size_t)batch_size * (size_t)kv_dim +
              (size_t)b * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim + index_x;

          uint4 tmp_k = *reinterpret_cast<const uint4 *>(kptr);
          uint4 tmp_v = *reinterpret_cast<const uint4 *>(vptr);

          regK[i] = tmp_k;
          regV[i] = tmp_v;
        }
        else {
          regK[i] = regV[i] = make_uint4(0, 0, 0, 0);
        }
      }
    }

    // Computing
    for (int t = 0; t < TILE_T; t += MFMA_N) {
      CDfloat acc = {0};
      for (int dimIdx = 0; dimIdx < HEAD_DIM; dimIdx += MFMA_BLOCK * MFMA_K) {
        // float k_value = __bfloat162float(ks[t + Bj][dimIdx + Ablock]);
        Bfloat k_value;
        for (int i = 0; i < MFMA_K; ++i) {
          k_value[i] = ks[t + Bj][dimIdx + Ablock * MFMA_K + i];
        }
        acc = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(q_values[dimIdx / (MFMA_BLOCK * MFMA_K)], k_value, acc, 0, 0, 0);
      }

      for (int width = warpSize / 2; width >= MFMA_N; width /= 2) {
        for (int i = 0; i < GPRs_CD; ++i) {
          acc[i] += __shfl_down(acc[i], width);
        }
      }

      // Softmax truc tuyen theo TILE thay vi theo tung key. Voi moi i: mot lan
      // rescale accumulator va 1 + MFMA_N lan exp, thay cho 4 rescale va 8 exp.
      int nvalid = 0;
      #pragma unroll
      for (int j = 0; j < MFMA_N; ++j)
        if (btIdx + t + j < t_start + n_steps) ++nvalid;

      if (nvalid > 0) {
        #pragma unroll
        for (int i = 0; i < MFMA_M; ++i) {
          float s[MFMA_N];
          float m_tile = -INFINITY;
          #pragma unroll
          for (int j = 0; j < MFMA_N; ++j) {
            // Hang ks ngoai pham vi bi zero-pad nen cho diem 0, ma exp(0) = 1
            // se lam hong tong -> phai ep ve -INFINITY.
            s[j] = (j < nvalid) ? (__shfl(acc[i], j)) : -INFINITY;
            m_tile = fmaxf(m_tile, s[j]);
          }
          const float m_new = fmaxf(m_values[i], m_tile);
          const float alpha = __expf(m_values[i] - m_new);
          float lsum = 0.f;
          float osum = 0.f;
          #pragma unroll
          for (int j = 0; j < MFMA_N; ++j) {
            const float e = __expf(s[j] - m_new);   // = 0 khi s[j] = -INF
            lsum += e;
            osum += e * __bfloat162float(vs[t + j][laneIdx]);
          }
          l_values[i] = l_values[i] * alpha + lsum;
          m_values[i] = m_new;
          out_values[i] = alpha * out_values[i] + osum;
        }
      }
    }

    __syncthreads();

    if (btIdx + TILE_T < t_start + n_steps) {
      for (int i = 0; i < nbLoadsKV; ++i) {
        int offset = i * strideKV;
        int index_x = 8 * loadKVIdx_x;
        int index_y = btIdx + TILE_T + loadKVIdx_y + offset;

        uint4 tmp_k = regK[i];
        uint4 tmp_v = regV[i];

        // Mot ghi LDS 128-bit thay cho 8 ghi 16-bit: 8 short nay lien nhau tren
        // truc thu hai va cot bat dau luon chia het 16 byte.
        *reinterpret_cast<uint4 *>(&ks[loadKVIdx_y + offset][loadKVIdx_x * 8]) = tmp_k;

        *reinterpret_cast<uint4 *>(&vs[loadKVIdx_y + offset][loadKVIdx_x * 8]) = tmp_v;
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
// 1 = kernel attention mfma 16x16x16 (xem chu thich o dinh nghia),
// 0 = duong cu mfma 4x4x4. Giu ca hai de con A/B duoc.
#ifndef FLASH_MFMA16
#define FLASH_MFMA16 1
#endif
// Bat rieng tung lop de chia doi loi: neu chi mot phia sai thi loi nam o
// duong co mat na truot, khong phai o phep nhan.
#ifndef FLASH_MFMA16_EVEN
#define FLASH_MFMA16_EVEN FLASH_MFMA16
#endif
#ifndef FLASH_MFMA16_ODD
#define FLASH_MFMA16_ODD  FLASH_MFMA16
#endif
#if FLASH_MFMA16_EVEN || FLASH_MFMA16_ODD
static_assert(FLASH_DECODE_TILE_T == 16,
              "kernel mfma 16x16x16 buoc tile phai la 16");
#endif
// Ghi tb ra file de so so hoc giua hai ban kernel. Chi bat khi GETP_DUMP_ATTN=1
// luc bien dich VA co bien moi truong GETP_DUMP_FILE luc chay.
#if GETP_DUMP_ATTN
#include <cstdlib>
static inline void getp_dump_attn(const float *tb, int batch_size,
                                  int n_attn_heads, int head_dim, int layer_id,
                                  int pos, hipStream_t stream) {
  const char *base = getenv("GETP_DUMP_FILE");
  if (!base) return;
  const char *el = getenv("GETP_DUMP_LAYER");
  const char *ep = getenv("GETP_DUMP_POS");
  if (el && layer_id != atoi(el)) return;
  if (ep && pos != atoi(ep)) return;
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  const size_t n = (size_t)batch_size * n_attn_heads * head_dim;
  float *h = (float *)malloc(n * sizeof(float));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(h, tb, n * sizeof(float), hipMemcpyDeviceToHost));
  char path[512];
  snprintf(path, sizeof path, "%s.L%d.P%d.D%d.bin", base, layer_id, pos, dev);
  FILE *f = fopen(path, "wb");
  if (f) { fwrite(h, sizeof(float), n, f); fclose(f); }
  free(h);
}
#endif

static inline void getp_flash_attn_decode_bf16(
    __hip_bfloat16 *tb_bf16, float *tb, const __hip_bfloat16 *key_cache_layer,
    const __hip_bfloat16 *value_cache_layer, const float *q, const int *mask_on,
    const float *attn_sinks_layer, int head_dim, int n_attn_heads,
    int n_kv_heads, int pos, int seq_len, int sliding_window, int layer_id,
    int batch_size, int cache_tcap, hipStream_t stream) {
  
  const int kv_dim = head_dim * n_kv_heads;
  const int kv_mul = n_attn_heads / n_kv_heads;
  const float invsd = 1.0f / sqrtf((float)head_dim);
  const int apply_mask = (sliding_window > 0 && ((layer_id & 1) == 0)) ? 1 : 0;

  if (apply_mask) {
#if FLASH_MFMA16_EVEN
    dim3 block(64 * (8 / FLASH_HEADS_PER_WAVE));
    dim3 grid(n_kv_heads, batch_size);
    flash_attn_decode_mfma16_kernel<FLASH_HEADS_PER_WAVE, true><<<grid, block, 0, stream>>>
      (tb_bf16, tb, key_cache_layer,
       value_cache_layer, q,
       mask_on,
       attn_sinks_layer, head_dim, n_attn_heads,
       n_kv_heads, pos, batch_size, kv_dim, kv_mul,
       sliding_window, cache_tcap, invsd);
#else
    dim3 block(128);
    dim3 grid(n_kv_heads, batch_size);
    flash_attn_decode_even_layer_bf16_matrix_core_kernel<FLASH_DECODE_TILE_T><<<grid, block, 0, stream>>>
      (tb, key_cache_layer, 
       value_cache_layer, q, 
       mask_on, 
       attn_sinks_layer, head_dim, n_attn_heads, 
       n_kv_heads, pos, batch_size, kv_dim, kv_mul, 
       sliding_window, cache_tcap, invsd);
    tensor_fp32_to_bf16(tb_bf16, tb,
                        batch_size * n_attn_heads * head_dim, stream);
#endif
  }
  else {
#if FLASH_MFMA16_ODD
    dim3 block(64 * (8 / FLASH_HEADS_PER_WAVE));
    dim3 grid(n_kv_heads, batch_size);
    flash_attn_decode_mfma16_kernel<FLASH_HEADS_PER_WAVE, false><<<grid, block, 0, stream>>>
      (tb_bf16, tb, key_cache_layer,
      value_cache_layer, q,
      mask_on,
      attn_sinks_layer, head_dim, n_attn_heads,
      n_kv_heads, pos, batch_size, kv_dim, kv_mul,
      sliding_window, cache_tcap, invsd);
#else
    dim3 block(128);
    dim3 grid(n_kv_heads, batch_size);
    flash_attn_decode_odd_layer_bf16_matrix_core_kernel<FLASH_DECODE_TILE_T><<<grid, block, 0, stream>>>
      (tb, key_cache_layer, 
      value_cache_layer, q, 
      mask_on,
      attn_sinks_layer, head_dim, n_attn_heads, 
      n_kv_heads, pos, batch_size, kv_dim, kv_mul, 
      cache_tcap, invsd);
    tensor_fp32_to_bf16(tb_bf16, tb,
                        batch_size * n_attn_heads * head_dim, stream);
#endif
    }

  

#if GETP_DUMP_ATTN
  getp_dump_attn(tb, batch_size, n_attn_heads, head_dim, layer_id, pos, stream);
#endif
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
  
  dim3 block(256), grid(B);
  gather_embedding_bf16_kernel<<<grid, block, 0, stream>>>(x, table, tok, H);
  
}

static inline void getp_matmul_router_bf16(float *xout, float *x, __hip_bfloat16 *w, __hip_bfloat16 *b, int n, int d, int batch_size, hipStream_t stream) {
  
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
  
}

#ifndef MATMUL_ATTN_O_BLOCK_ROWS
#define MATMUL_ATTN_O_BLOCK_ROWS 128
#endif
#ifndef MATMUL_ATTN_O_BLOCK_COLS
#define MATMUL_ATTN_O_BLOCK_COLS 128
#endif
#ifndef MATMUL_ATTN_O_BLOCK_DEPTH
#define MATMUL_ATTN_O_BLOCK_DEPTH 32
#endif
#ifndef MATMUL_ATTN_O_WARP_TILE_M
#define MATMUL_ATTN_O_WARP_TILE_M 64
#endif
#ifndef MATMUL_ATTN_O_WARP_TILE_N
#define MATMUL_ATTN_O_WARP_TILE_N 64
#endif
#ifndef MATMUL_ATTN_O_WAVES_PER_BLOCK
// Suy tu tile: than kernel dung TILE_N = 64 * (BM/WM)*(BN/WN) thread.
// Ghi de bang -D chi duoc phep neu van bang bieu thuc nay (co static_assert).
#define MATMUL_ATTN_O_WAVES_PER_BLOCK (((MATMUL_ATTN_O_BLOCK_ROWS) / (MATMUL_ATTN_O_WARP_TILE_M)) * ((MATMUL_ATTN_O_BLOCK_COLS) / (MATMUL_ATTN_O_WARP_TILE_N)))
#endif

// ===========================================================================
// LDS cua attn_o: tu k-major [BK][BM+GETP_LDS_PAD] sang layout O (slot) 8 byte.
//
//   O (k0, row) chua 4 short k = 4*k0 .. 4*k0+3 cua CUNG mot hang.
//   * Mot manh MFMA (A[i][k] hoac B[k][j]) = dung 4 short k lien tiep cua mot
//     hang  ->  MOT ds_read_b64 thay cho BON ds_read_u16.
//   * Mot uint4 nap tu HBM = 8 short k lien tiep = hai o  ->  HAI ds_write_b64
//     thay cho TAM ds_write_b16.
//
// Bank (32 bank x 4 byte = 128 byte/chu ky; b64 phuc vu theo pha 16 lane, moi
// lane chiem MOT cap bank chan/le, khong xung dot <=> 16 lane phu kin 16 cap):
//
//   DOC   16 lane cua mot pha co cung k0 (k0 chi phu thuoc lane>>4) va 16 hang
//         lien tiep (Ai = Bj = lane&15):
//             byte = C + 8*(lane&15)  ->  128 byte lien tuc  ->  32 bank, moi
//         bank dung mot lan. DUNG VOI MOI PADS - doc khong rang buoc pad.
//
//   GHI   16 lane cua mot pha co a = lane % NKC (chunk k, truc nhanh loader) va
//         b = lane / NKC (hang), NKC = BK/8. O dich la k0 = 2a (va 2a+1), nen
//             dword = 2*(k0*S + row),  S = BM + PADS (hoac BN + PADS)
//             bank  = (4*S*a + 2*b) mod 32.
//         Pad phai lam 16 lane phu kin 16 bank chan:
//             BK=16 (NKC=2): 4S == 16 (mod 32) -> S == 4 (mod 8) -> PADS 4, 12
//             BK=32 (NKC=4): 4S == 8 hay 24    -> S == 2 hay 6   -> PADS 2, 6, 10, 14
//             BK=64 (NKC=8): 4S == 4,12,20,28  -> S le           -> PADS 1, 3, 5, 7
//         KHONG co hang so dung cho moi BK, nen PADS phai TU SUY theo BK
//         (mac dinh cua kernel nay la BK=16, co thang la BK=32, sweep co BK=64).
//
// Kich thuoc LDS KHONG tang: BK*(BM+BN+2*PADS)*2 byte, bang hoac nho hon ban cu
// BK*(BM+BN+2*GETP_LDS_PAD)*2 vi PADS <= 4.
//     BK=16 PADS=4:  8448 B  (cu  8448 B)
//     BK=32 PADS=2: 16640 B  (cu 16896 B)
//     BK=64 PADS=1: 33024 B  (cu 33792 B)
// ===========================================================================
constexpr bool attn_o_lds_write_conflict_free(int S, int NKC) {
  if (NKC <= 0 || NKC > 16 || 16 % NKC != 0) return false;
  int seen = 0;
  for (int l = 0; l < 16; ++l) {
    const int a = l % NKC;                 // chunk k cua loader
    const int b = l / NKC;                 // hang trong pha
    int bank = (4 * S * a + 2 * b) % 32;
    if (bank < 0) bank += 32;
    if (bank & 1) return false;            // o phai can le 8 byte
    const int bit = 1 << (bank >> 1);
    if (seen & bit) return false;          // hai lane cung cap bank
    seen |= bit;
  }
  return seen == 0xFFFF;                   // phu kin 16 cap bank
}

// PADS nho nhat hop le cho CA HAI tile (X dung S = BM+PADS, W dung S = BN+PADS).
constexpr int attn_o_lds_pad_slots(int BM_, int BN_, int NKC) {
  for (int p = 0; p < 16; ++p)
    if (attn_o_lds_write_conflict_free(BM_ + p, NKC) &&
        attn_o_lds_write_conflict_free(BN_ + p, NKC))
      return p;
  return -1;
}

// Pad RIENG cua attn_o - KHONG dung chung GETP_LDS_PAD (macro do van thuoc ve
// qkv:1392-1393 va logits:3656-3657). Hai macro dem hai thu khac nhau:
//   GETP_LDS_PAD                : so SHORT dem tren truc HANG cua mot mat k;
//                                 nhan gia tri tuy y, chi doi hieu nang
//                                 (autotune.sh:38-40 va run2.sh:35-37 quet
//                                 2 / 8 / 16).
//   MATMUL_ATTN_O_LDS_PAD_SLOTS : so O 8-BYTE dem tren truc hang cua mot mat o;
//                                 co rang buoc CUNG theo BK (bang tren). Dua
//                                 2 / 8 / 16 vao day se hoac loi bien dich hoac
//                                 thanh xung dot ghi 2 chieu bi do nham la nhieu.
// -1 = tu suy theo (BM, BN, BK). Chi dat tay khi muon quet 6/10/14 o BK=32.
#ifndef MATMUL_ATTN_O_LDS_PAD_SLOTS
#define MATMUL_ATTN_O_LDS_PAD_SLOTS (-1)
#endif

// Gop hai dword lien tiep cua uint4 thanh mot o 4 x bf16. Thuan tuy doi kieu:
// hai VGPR da nam dung cho (ket qua global_load_dwordx4) nen KHONG sinh lenh
// VALU nao - khac han 4 lenh trich nua-word (& 0xFFFF / >> 16) cua ban cu.
// __builtin_bit_cast thay cho union punning: cung codegen, dung chuan.
__device__ __forceinline__ bf16x4 attn_o_pack2(unsigned int lo, unsigned int hi) {
  struct Pair { unsigned int a, b; };
  static_assert(sizeof(Pair) == sizeof(bf16x4), "pack2: kich thuoc lech");
  Pair p; p.a = lo; p.b = hi;
  return __builtin_bit_cast(bf16x4, p);
}

template<
  int BM, int BN, int BK,
  int WM, int WN,
  int WARPS_PER_BLOCK,
  bool ACC = false,
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
  // SUA LOI TIEM AN (cung lop voi forward.hip:1371-1372 cua qkv): xInBlockTile
  // chay tren truc N nen phai chia theo BN/WN; yInBlockTile chay tren truc M nen
  // la thuong cua cung mau so do. Ban cu dung (BM/WM) cho ca hai - vo hai khi
  // BM/WM == BN/WN (moi cau hinh dang chay, ke ca -DMATMUL_ATTN_O_BLOCK_DEPTH=32)
  // nhung SAI voi attno_cfg.txt dong 5 (bm256) va dong 8 (bk32_bm256): BM/WM=4,
  // BN/WN=2 -> xInBlockTile toi 3 -> col = 3*64+63 = 255 >= BN=128, doc ngoai
  // tile W va ghi sai cot, dong thoi chi mot nua so hang duoc tinh. Hai dong nay
  // sinh MA Y HET ban cu o moi cau hinh co BM/WM == BN/WN.
  const int xInBlockTile = waveIdx % (BN / WN);
  const int yInBlockTile = waveIdx / (BN / WN);

  const int Ai = laneIdx & 15;
  const int Ak = 4 * (laneIdx >> 4);
  const int Bj = laneIdx & 15;
  const int Bk = 4 * (laneIdx >> 4);

  constexpr int nIterWaveM = WM / MFMA_M;
  constexpr int nIterWaveN = WN / MFMA_N;

  constexpr int nbLoadsW = BN * (BK / 8) / THREADS;
  const int loadWIdx_y = threadIdx.x / (BK / 8);
  const int loadWIdx_x = threadIdx.x % (BK / 8);
  const int loadW_k0   = 2 * loadWIdx_x;      // o dau cua uint4 trong tile W
  constexpr int strideW = THREADS / (BK / 8);
  // Da xoa nbLoadsX / loadXIdx_x / loadXIdx_y / strideX: chung KHONG BAO GIO
  // duoc dung (duong nap X di qua bo bien *_x8 ben duoi). Giu lai chung dung la
  // lop "stride cu song sot qua mot lan sua".

  constexpr int KP   = 4;             // short / o = 8 byte = 1 ds_read_b64
  constexpr int KSL  = BK / KP;       // so mat o (k0)
  constexpr int NKC  = BK / 8;        // so chunk k cua loader (uint4 = 8 short)
  constexpr int PADS = (MATMUL_ATTN_O_LDS_PAD_SLOTS >= 0)
                         ? MATMUL_ATTN_O_LDS_PAD_SLOTS
                         : attn_o_lds_pad_slots(BM, BN, NKC);
  static_assert(BK % 8 == 0, "loader nap uint4 = 8 short theo truc k");
  static_assert(BK % KP == 0, "BK phai chia het cho KP = 4");
  static_assert(PADS >= 0,
                "khong tim duoc PADS nao lam ds_write_b64 het xung dot bank; "
                "xem attn_o_lds_pad_slots()");
  static_assert(attn_o_lds_write_conflict_free(BM + PADS, NKC),
                "MATMUL_ATTN_O_LDS_PAD_SLOTS sai: ds_write_b64 tile X xung dot "
                "bank (BK=16 -> 4, BK=32 -> 2, BK=64 -> 1)");
  static_assert(attn_o_lds_write_conflict_free(BN + PADS, NKC),
                "MATMUL_ATTN_O_LDS_PAD_SLOTS sai: ds_write_b64 tile W xung dot "
                "bank (BK=16 -> 4, BK=32 -> 2, BK=64 -> 1)");
  static_assert(sizeof(bf16x4) == 8 && sizeof(Afloat) == 8 &&
                sizeof(Bfloat) == 8, "manh MFMA phai la 8 byte");

  // Ten moi (xsT/wsT) co chu dich: moi vi tri xs[..][..] chua doi se la LOI BIEN
  // DICH chu khong phai mot dia chi sai im lang.
  // Kieu phan tu la bf16x4 (alignment 8) nen MOI truy cap la mot ds_read_b64 /
  // ds_write_b64 - khong can reinterpret_cast o bat ky cho nao. __align__(16)
  // bao dam goc mang: mang __shared__ kieu unsigned short chi duoc canh 2 byte.
  __shared__ __align__(16) bf16x4 xsT[KSL][BM + PADS];
  __shared__ __align__(16) bf16x4 wsT[KSL][BN + PADS];
  static_assert(sizeof(xsT) + sizeof(wsT) <= 64 * 1024, "LDS > 64 KB moi CU");

  CDfloat acc_local[nIterWaveM * nIterWaveN] = {0};

  // AGGRESSIVE X load: uint4 for 8 bf16 elements (16 bytes)
  constexpr int X_LOAD_VEC = 8;
  const int loadX_x8 = threadIdx.x % (BK / X_LOAD_VEC);
  const int loadX_y8 = threadIdx.x / (BK / X_LOAD_VEC);
  const int loadX_k0 = 2 * loadX_x8;          // o dau cua uint4 trong tile X
  constexpr int strideX8 = THREADS / (BK / X_LOAD_VEC);
  constexpr int nbLoadsX8 = BM * (BK / X_LOAD_VEC) / THREADS;
  // Loader phai phu kin tile. Khong co hai assert nay thi wm32/wn32 o BK=16 cho
  // nbLoadsX8 = 128*2/512 = 0, va bm256 o BK=16 cho nbLoadsW = 128*2/512 = 0:
  // tile khong duoc nap va ket qua sai IM LANG (va nhanh hon). Ba dong cau hinh
  // do trong attno_cfg.txt phai di kem -DMATMUL_ATTN_O_BLOCK_DEPTH=32 tro len.
  static_assert(nbLoadsX8 >= 1 && (BM * (BK / X_LOAD_VEC)) % THREADS == 0,
                "loader X khong phu kin tile: BM*(BK/8) phai chia het THREADS");
  static_assert(nbLoadsW >= 1 && (BN * (BK / 8)) % THREADS == 0,
                "loader W khong phu kin tile: BN*(BK/8) phai chia het THREADS");
  
  #pragma unroll
  for (int i = 0; i < nbLoadsX8; ++i) {
    const int off = i * strideX8;
    const int row_tile = loadX_y8 + off;
    if (row_tile >= BM) break;
    const int rg = BM * blockIdx.y + row_tile;
    const int kk = X_LOAD_VEC * loadX_x8;
    
    if (rg < M && kk + 7 < K) {
      // Fast path: uint4 load (8 bf16 = 16 bytes)
      const uint4* p = reinterpret_cast<const uint4*>(x + (size_t)rg * K + kk);
      const uint4 v = *p;

      // 8 short k lien tiep = hai o -> hai ds_write_b64 (truoc: 8 ds_write_b16).
      // Thu tu nua-word giu NGUYEN: v.x thap = k+0, v.x cao = k+1, v.y thap =
      // k+2, ... -> cung anh byte -> bit-exact.
      xsT[loadX_k0 + 0][row_tile] = attn_o_pack2(v.x, v.y);
      xsT[loadX_k0 + 1][row_tile] = attn_o_pack2(v.z, v.w);
    } else {
      const bf16x4 zero4 = attn_o_pack2(0u, 0u);
      xsT[loadX_k0 + 0][row_tile] = zero4;
      xsT[loadX_k0 + 1][row_tile] = zero4;
    }
  }
  // AGGRESSIVE W load: optimized uint4 load with full unroll
  #pragma unroll
  for (int i = 0; i < nbLoadsW; ++i) {
    const int off = i * strideW;
    const int kk = 8 * loadWIdx_x;
    const int cy = BN * blockIdx.x + loadWIdx_y + off;
    
    if (kk + 7 < K && cy < N) {
      // Fast path: vectorized uint4 load
      const uint4* p = reinterpret_cast<const uint4*>(&w[(size_t)cy * K + kk]);
      const uint4 tmp = *p;

      wsT[loadW_k0 + 0][loadWIdx_y + off] = attn_o_pack2(tmp.x, tmp.y);
      wsT[loadW_k0 + 1][loadWIdx_y + off] = attn_o_pack2(tmp.z, tmp.w);
    } else {
      const bf16x4 zero4 = attn_o_pack2(0u, 0u);
      wsT[loadW_k0 + 0][loadWIdx_y + off] = zero4;
      wsT[loadW_k0 + 1][loadWIdx_y + off] = zero4;
    }
  }
  __syncthreads();

  #pragma unroll 1
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    uint4 regX8[nbLoadsX8];
    uint4 regW[nbLoadsW];
    
    // AGGRESSIVE prefetching: uint4 loads
    if (bkIdx < K - BK) {
      #pragma unroll
      for (int i = 0; i < nbLoadsX8; ++i) {
        const int off = i * strideX8;
        const int row_tile = loadX_y8 + off;
        const int rg = BM * blockIdx.y + row_tile;
        const int kk = bkIdx + BK + X_LOAD_VEC * loadX_x8;
        
        if (row_tile < BM && rg < M && kk + 7 < K) {
          // Fast path: uint4 load (8 bf16)
          const uint4* p = reinterpret_cast<const uint4*>(x + (size_t)rg * K + kk);
          regX8[i] = *p;
        } else {
          regX8[i] = make_uint4(0,0,0,0);
        }
      }
      
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        const int off = i * strideW;
        const int kk = bkIdx + BK + 8 * loadWIdx_x;
        const int cy = BN * blockIdx.x + loadWIdx_y + off;
        
        if (kk + 7 < K && cy < N) {
          const uint4* p = reinterpret_cast<const uint4*>(&w[(size_t)cy * K + kk]);
          regW[i] = *p;
        } else {
          regW[i] = make_uint4(0,0,0,0);
        }
      }
    }

    // MFMA compute loop - fully unrolled for maximum ILP
    #pragma unroll
    for (int k = 0; k < BK; k += MFMA_K) {
      // base_k = k + Ak la boi cua 4 (k boi cua 16, Ak = 4*(lane>>4)) nen ca 4
      // short k nam GON trong mot o -> chi so o = base_k >> 2. Chi so giu 32-bit:
      // dia chi LDS la 32-bit, khong dua (size_t) vao vong nong.
      const int ak0 = (k + Ak) >> 2;
      const int bk0 = (k + Bk) >> 2;

      #pragma unroll
      for (int im = 0; im < nIterWaveM; ++im) {
        const int row = yInBlockTile * WM + MFMA_M * im + Ai;
        // MOT ds_read_b64 (truoc: 4 ds_read_u16 buoc BM+GETP_LDS_PAD).
        const Afloat av = xsT[ak0][row];

        #pragma unroll
        for (int in = 0; in < nIterWaveN; ++in) {
          const int col = xInBlockTile * WN + MFMA_N * in + Bj;

          // MOT ds_read_b64 (truoc: 4 ds_read_u16 buoc BN+GETP_LDS_PAD).
          const Bfloat bv = wsT[bk0][col];

          const int idx = im * nIterWaveN + in;
          acc_local[idx] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc_local[idx], 0, 0, 0);
        }
      }
    }
    __syncthreads();

    // Store prefetched data: uint4 register → shared memory
    if (bkIdx < K - BK) {
      #pragma unroll
      for (int i = 0; i < nbLoadsX8; ++i) {
        const int off = i * strideX8;
        const int row_tile = loadX_y8 + off;
        if (row_tile >= BM) break;
        
        const uint4 v = regX8[i];
        xsT[loadX_k0 + 0][row_tile] = attn_o_pack2(v.x, v.y);
        xsT[loadX_k0 + 1][row_tile] = attn_o_pack2(v.z, v.w);
      }
      
      #pragma unroll
      for (int i = 0; i < nbLoadsW; ++i) {
        const int off = i * strideW;
        const uint4 w_val = regW[i];
        wsT[loadW_k0 + 0][loadWIdx_y + off] = attn_o_pack2(w_val.x, w_val.y);
        wsT[loadW_k0 + 1][loadWIdx_y + off] = attn_o_pack2(w_val.z, w_val.w);
      }
    }
    __syncthreads();
  }

  // AGGRESSIVE output: fully unrolled with optimized address calculation
  #pragma unroll
  for (int im = 0; im < nIterWaveM; ++im) {
    #pragma unroll
    for (int in = 0; in < nIterWaveN; ++in) {
      const int acc_index = im * nIterWaveN + in;
      
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int CDi = 4 * (laneIdx >> 4) + (i & 3);
        const int CDj = laneIdx & 15;
        const int gx = blockIdx.x * BN + WN * xInBlockTile + in * MFMA_N + CDj;
        const int gy = blockIdx.y * BM + WM * yInBlockTile + im * MFMA_M + CDi;
        
        if (gx < N && gy < M) {
          float v = acc_local[acc_index][i];
          if (b) v += __bfloat162float(b[gx]);
          // ACC: cong thang vao dong du (dev_x) thay vi ghi ra tb2 roi de
          // vecadd doc lai. Moi o (gy,gx) do DUNG MOT thread sinh ra - khong
          // co split-k trong kernel nay - nen doc-sua-ghi la an toan, khong
          // can atomic. Bit-exact: v la dung so fp32 ma tb2 se chua.
          if constexpr (ACC) xout[(size_t)gy * N + gx] += v;
          else               xout[(size_t)gy * N + gx]  = v;
        }
      }
    }
  }
}

static inline void getp_matmul_attn_o_bf16(
    float *xout, const __hip_bfloat16 *x_bf16,
    const __hip_bfloat16 *w, const __hip_bfloat16 *b,
    int n, int d, int batch_size, hipStream_t stream) {
      
  const int M = batch_size, N = d, K = n;

  constexpr int BM = MATMUL_ATTN_O_BLOCK_ROWS;
  constexpr int BN = MATMUL_ATTN_O_BLOCK_COLS;
  constexpr int BK = MATMUL_ATTN_O_BLOCK_DEPTH;
  constexpr int WM = MATMUL_ATTN_O_WARP_TILE_M;
  constexpr int WN = MATMUL_ATTN_O_WARP_TILE_N;
  constexpr int WARPS = MATMUL_ATTN_O_WAVES_PER_BLOCK;

  // Ghi chu __launch_bounds__: doi so THU HAI cua HIP la SO WAVE TOI THIEU MOI
  // SIMD, khong phai so block moi CU. Kernel nay chi truyen mot doi so (THREADS)
  // nen khong bi anh huong; neu ai do them doi so thu hai thi so workgroup duoc
  // hua la 4*CTA/WARPS chu khong phai CTA.
  dim3 block(64 * WARPS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  matmul_attn_o_bf16_kernel_tuned<BM,BN,BK,WM,WN,WARPS,false>
      <<<grid, block, 0, stream>>>(xout, x_bf16, w, b, M, N, K);

}

// Nhu tren nhung CONG vao xout (dong du fp32) thay vi ghi de. Bo hoan toan
// tensor tb2: mot lan ghi 17,7 MB va mot lan doc 17,7 MB moi lop.
static inline void getp_matmul_attn_o_bf16_acc(
    float *x_resid, const __hip_bfloat16 *x_bf16,
    const __hip_bfloat16 *w, const __hip_bfloat16 *b,
    int n, int d, int batch_size, hipStream_t stream) {

  const int M = batch_size, N = d, K = n;

  constexpr int BM = MATMUL_ATTN_O_BLOCK_ROWS;
  constexpr int BN = MATMUL_ATTN_O_BLOCK_COLS;
  constexpr int BK = MATMUL_ATTN_O_BLOCK_DEPTH;
  constexpr int WM = MATMUL_ATTN_O_WARP_TILE_M;
  constexpr int WN = MATMUL_ATTN_O_WARP_TILE_N;
  constexpr int WARPS = MATMUL_ATTN_O_WAVES_PER_BLOCK;

  dim3 block(64 * WARPS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  matmul_attn_o_bf16_kernel_tuned<BM,BN,BK,WM,WN,WARPS,true>
      <<<grid, block, 0, stream>>>(x_resid, x_bf16, w, b, M, N, K);
}


// Anh xa float -> uint32 giu nguyen thu tu, de so sanh so thuc tro thanh so
// sanh so nguyen khong dau: so khong am duoc dua len nua tren, so am bi lat het.
__device__ inline unsigned int order_preserving_bits(float v) {
  unsigned int b = __float_as_uint(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ inline float order_preserving_bits_inv(unsigned int k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
// Khoa don dieu theo (gia tri, -idx): gia tri lon thang, hoa thi idx NHO thang.
// Nho vay argmax chi con la mot lenh atomicMax, khong con vong CAS va khong con
// phu thuoc thu tu block den.
__device__ inline unsigned long long pack_val_idx(float v, unsigned int idx) {
  return (static_cast<unsigned long long>(order_preserving_bits(v)) << 32) |
         static_cast<unsigned long long>(0xffffffffu - idx);
}
__device__ inline float unpack_val(unsigned long long p) {
  return order_preserving_bits_inv(static_cast<unsigned int>(p >> 32));
}
__device__ inline unsigned int unpack_idx(unsigned long long p) {
  return 0xffffffffu - static_cast<unsigned int>(p & 0xffffffffu);
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
  // Truoc day day la mot vong CAS voi phep so sanh dung nguong TUONG DOI
  // (1e-6 * max(|v|,|ov|)), tuc mot quan he KHONG BAC CAU: A ~ B va B ~ C khong
  // suy ra A ~ C. Voi hang nghin block cot cung tranh mot o, ke thang phu thuoc
  // thu tu den, va ket qua doi giua hai lan chay cung mot binary. Khoa don dieu
  // o tren bien viec nay thanh mot atomicMax duy nhat, co thu tu toan phan chat.
  atomicMax(addr, pack_val_idx(v, idx));
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
#define MATMUL_LOGITS_BLOCK_DEPTH 32
#endif
#ifndef MATMUL_LOGITS_WARP_TILE_M
#define MATMUL_LOGITS_WARP_TILE_M 64
#endif
#ifndef MATMUL_LOGITS_WARP_TILE_N
#define MATMUL_LOGITS_WARP_TILE_N 64
#endif
#ifndef MATMUL_LOGITS_WAVES_PER_BLOCK
// Suy tu tile: than kernel dung TILE_N = 64 * (BM/WM)*(BN/WN) thread.
// Ghi de bang -D chi duoc phep neu van bang bieu thuc nay (co static_assert).
#define MATMUL_LOGITS_WAVES_PER_BLOCK (((MATMUL_LOGITS_BLOCK_ROWS) / (MATMUL_LOGITS_WARP_TILE_M)) * ((MATMUL_LOGITS_BLOCK_COLS) / (MATMUL_LOGITS_WARP_TILE_N)))
#endif

// ---------------------------------------------------------------------------
// Hinh hoc LDS cua kernel logits (LM head).
//
// Tile duoc xep theo [row][k] - k LIEN TIEP trong mot hang - nen "padding" o
// day la pad cua MOT HANG k, KHONG phai pad cua mot mat k nhu GETP_LDS_PAD.
// Hai dai luong nay cung ten nhung khac nghia, khac dieu kien hop le va khac
// don vi, nen kernel nay phai co macro pad RIENG. GETP_LDS_PAD giu nguyen
// nghia cu va van phuc vu hai kernel con lai o bo [k][row]: qkv (:1392-1393)
// va attn_o (:3324-3325).
//
//   Buoc mot hang  = (BK+PAD) short = 2*(BK+PAD) byte = (BK+PAD)/2 bank.
//   Mot fragment MFMA = 4 short = 8 byte = 2 bank = 1 "double-bank".
//   16 lane cua mot phase ds_read_b64 chuyen 16*8 = 128 byte = dung 32 bank.
//   Trong mot phase, L/16 la HANG SO nen basek co dinh va row = row0 + (L&15)
//   chay tren 16 hang lien tiep, tuc db(i) = (S*i + c) mod 16 voi S = (BK+PAD)/4.
//   Trai het 32 bank <=> gcd(S,16) = 1 <=> S LE.
//   BK boi 16 (static_assert san co) => BK/4 chan => S = BK/4 + PAD/4 le <=>
//   PAD/4 le. Vay PAD = 4 dung cho MOI BK hop le (BK=16 -> S=5; 32 -> S=9;
//   64 -> S=17) va la gia tri RE NHAT; 12/20/28 cung hop le nhung chi ton them
//   LDS ma khong bot duoc mot xung dot nao.
//
// Vi du BK=32 (co trong co build dang thang): KPAD = 36 short = 72 byte
// = 18 bank = 9 double-bank. Bank DAU cua 16 lane trong mot phase (row0=0,
// basek=0) la 18*i mod 32 = 18,4,22,8,26,12,30,16,2,20,6,24,10,28,14,0 - doi
// mot khac nhau, moi lane chiem cap bank {b, b+1} -> phu dung 32 bank mot lan,
// KHONG xung dot. Duong GHI van con xung dot 2-way va khong pad nao chua duoc:
// ghi X can {0,S} + {0..7} phu Z16 => S = 8 (mod 16) => S CHAN; ghi W can
// {0,S,2S,3S} = {a,a+8,b,b+8} => 2S = 8 (mod 16) => S CHAN; ca hai deu mau
// thuan voi dieu kien DOC (S le). Da quet can PAD 0..64.
#ifndef MATMUL_LOGITS_LDS_KPAD
#define MATMUL_LOGITS_LDS_KPAD 4
#endif

// Mot nguon su that duy nhat cho hinh hoc tile logits.
constexpr int logits_lds_kpad(int BK, int PAD) { return BK + PAD; }
constexpr size_t logits_lds_shorts(int BM, int BN, int BK, int PAD) {
  // xsT[BM][KPAD] + wsT[BN][KPAD]
  return (size_t)(BM + BN) * (size_t)logits_lds_kpad(BK, PAD);
}

// 8 bf16 lien tiep theo k (mot uint4 tu HBM) -> 2 x ds_write_b64.
// Sao chep NGUYEN dword: trong mot dword, nua thap la k+0 va nua cao la k+1,
// dung y het ban cu, nen ket qua bit-exact VA khong ton mot lenh VALU nao
// (khong con & 0xFFFF / >> 16). Cung khuon mau voi put16 cua flash attn (:2418).
__device__ __forceinline__ void logits_lds_store8(unsigned short* dst, const uint4 v) {
  *reinterpret_cast<uint2*>(dst + 0) = make_uint2(v.x, v.y);
  *reinterpret_cast<uint2*>(dst + 4) = make_uint2(v.z, v.w);
}

// 4 float -> 4 bf16 lien tiep theo k -> 1 x ds_write_b64. f32_to_bf16bits duoc
// goi tren dung nhung gia tri cu, theo dung thu tu cu; chi dia chi LDS doi.
__device__ __forceinline__ void logits_lds_store4(unsigned short* dst, const float4 v) {
  const bf16x4 p = { f32_to_bf16bits(v.x), f32_to_bf16bits(v.y),
                     f32_to_bf16bits(v.z), f32_to_bf16bits(v.w) };
  *reinterpret_cast<bf16x4*>(dst) = p;
}

template<
  int BM, int BN, int BK,
  int WM, int WN,
  int WARPS_PER_BLOCK,
  int THREADS = 64 * WARPS_PER_BLOCK
>
// __launch_bounds__ chi truyen MOT doi so o day. Neu sau nay them doi so thu
// hai thi luu y: trong HIP do la so wave TOI THIEU tren moi SIMD, khong phai
// so block tren moi CU; so workgroup/CU tuong ung la 4*CTA/WARPS_PER_BLOCK.
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
  // Wave -> (tile COT, tile HANG). xInBlockTile danh chi so cot nen mau so phai
  // la BN/WN. Ban cu chia BM/WM: bang nhau o cau hinh dang ship (2 == 2) nen
  // thay doi nay BIT-EXACT, nhung sai im lang - va doc ngoai tile LDS - voi cac
  // dong sweep bm256 / bn256 / wm32 / wn32 / bk32_wn32 trong logits_cfg.txt.
  // Cung LOP LOI voi :1371-1372 cua qkv.
  constexpr int splitN = BN / WN;
  const int xInBlockTile = waveIdx % splitN;
  const int yInBlockTile = waveIdx / splitN;

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

  // Tile LDS theo [row][k]: 4 phan tu k ma MOT lane can cho mfma nam lien tiep
  // -> mot ds_read_b64 thay vi 4 ds_read_u16, va 8 short nap tu HBM thanh
  // 2 ds_write_b64 thay vi 8 ds_write_b16. Doi ten xs/ws -> xsT/wsT co chu dich:
  // moi cho con doc theo bo cu [k][row] se la LOI BIEN DICH, khong phai mot dia
  // chi sai im lang. Chi so trong vong lap giu 32-bit (dia chi LDS la 32-bit).
  constexpr int PAD  = MATMUL_LOGITS_LDS_KPAD;
  constexpr int KPAD = logits_lds_kpad(BK, PAD);
  static_assert(KPAD % 4 == 0,
                "KPAD phai boi 4 short: ds_read_b64/ds_write_b64 can align 8 byte.");
  static_assert((KPAD / 4) % 2 == 1,
                "(BK+PAD)/4 phai LE thi 16 lane cua mot phase b64 moi trai het 32 bank.");
  static_assert(logits_lds_shorts(BM, BN, BK, PAD) * sizeof(unsigned short)
                    + sizeof(float) * BM + sizeof(int) * BM <= 64 * 1024,
                "Tile LDS + smax/sidx vuot 64 KB LDS moi CU.");

  __shared__ __align__(16) unsigned short xsT[BM][KPAD];
  __shared__ __align__(16) unsigned short wsT[BN][KPAD];
  __shared__ float smax[BM];
  __shared__ int sidx[BM];

#ifdef DEBUG_LDS_POISON
  // Kiem launch: than kernel gia dinh dung THREADS thread.
  if (blockDim.x != THREADS) {
    if (threadIdx.x == 0)
      printf("BAD LAUNCH: blockDim=%d nhung THREADS=%d\n", (int)blockDim.x, (int)THREADS);
    return;
  }
  // Do day tile bang bf16 qNaN: o nao loader khong ghi se thanh NaN.
  {
    unsigned short* raw = &xsT[0][0];
    for (int i = threadIdx.x; i < BM * KPAD; i += blockDim.x) raw[i] = 0x7FC0;
    raw = &wsT[0][0];
    for (int i = threadIdx.x; i < BN * KPAD; i += blockDim.x) raw[i] = 0x7FC0;
  }
  __syncthreads();
#endif

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
    // dia chi = 2*(ry*KPAD + 4*loadXIdx_x) byte: KPAD boi 4 va chunk boi 4
    // => boi 8 => hop le cho ds_write_b64.
    logits_lds_store4(&xsT[ry][4 * loadXIdx_x], tmp);
  }
  for (int i = 0; i < nbLoadsW; ++i) {
    int off = i * strideW;
    int kk = 0 + 8 * loadWIdx_x;
    int cy = BN * blockIdx.x + loadWIdx_y + off;
    uint4 t = (kk < K && cy < N) ? *reinterpret_cast<const uint4 *>(&w[(size_t)cy * K + kk]) : make_uint4(0,0,0,0);
    logits_lds_store8(&wsT[loadWIdx_y + off][8 * loadWIdx_x], t);
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
          // Lane L can A[i = L%16][k = 4*(L/16)+r], r = 0..3: BON phan tu k lien
          // tiep cua CUNG mot hang -> mot ds_read_b64 (dia chi boi 8 vi KPAD boi
          // 4, k boi 16 va Ak/Bk boi 4). Afloat/Bfloat la vector 8 byte nen cast
          // nay chinh la thu bao cho LLVM duoc phat mot lenh duy nhat.
          const Afloat av = *reinterpret_cast<const Afloat*>(&xsT[row][k + Ak]);
          const Bfloat bv = *reinterpret_cast<const Bfloat*>(&wsT[col][k + Bk]);
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
        logits_lds_store4(&xsT[ry][4 * loadXIdx_x], v);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;
        logits_lds_store8(&wsT[loadWIdx_y + off][8 * loadWIdx_x], regW[i]);
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
      
  const int M = batch_size, N = n, K = d;

  constexpr int BM = MATMUL_LOGITS_BLOCK_ROWS;
  constexpr int BN = MATMUL_LOGITS_BLOCK_COLS;
  constexpr int BK = MATMUL_LOGITS_BLOCK_DEPTH;
  constexpr int WM = MATMUL_LOGITS_WARP_TILE_M;
  constexpr int WN = MATMUL_LOGITS_WARP_TILE_N;
  constexpr int WARPS = MATMUL_LOGITS_WAVES_PER_BLOCK;

  dim3 block(64 * WARPS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  // hipFree dong bo ca thiet bi, nen cap phat/giai phong moi buoc la mot lan
  // dung hoan toan moi buoc cho mot buffer 12 KB co kich thuoc khong doi.
  static thread_local unsigned long long *pairs = nullptr;
  static thread_local int pairs_cap = 0;
  if (pairs_cap < M) {
    if (pairs) HIP_CHECK(hipFree(pairs));
    HIP_CHECK(hipMalloc(&pairs, sizeof(unsigned long long) * M));
    pairs_cap = M;
  }
  dim3 gi((M + 255) / 256), bi(256);
  init_argmax_pairs<<<gi, bi, 0, stream>>>(pairs, M, N);

  matmul_logits_argmax_bf16_kernel_tuned<BM,BN,BK,WM,WN,WARPS>
      <<<grid, block, 0, stream>>>(x, w, M, N, K, pairs);

  extract_argmax_pairs<<<gi, bi, 0, stream>>>(pairs, out, M);
  // (pairs duoc giu lai cho buoc sau)
  
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
  
  dim3 grid(B), block(1024);
  argmax_rows_kernel<<<grid, block, 0, stream>>>(logits, V, out);
  
  
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
                       Barrier &sync_point, int thread_idx_offset, int token[],
                       int pos, int *mask_on) {
  Config *p = &dev_transformers[thread_idx_offset]->config;
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int n_experts = p->n_experts;

  const int base_device_index = workers->device_index;

  GPUWorker *worker = &workers[thread_idx_offset];
  int device_index = worker->device_index;
  // assert(device_index == thread_idx_offset);

  DeviceTransformerWeights *dev_w = &dev_transformers[device_index]->weights;
  RunState *dev_s = &dev_transformers[device_index]->state;
  RunStateExt *ext = ext_get(device_index);
  hipStream_t compute_stream = dev_transformers[device_index]->compute_stream;
  hipStream_t memory_stream  = dev_transformers[device_index]->memory_stream;

  float *dev_x = dev_s->x;

  HIP_CHECK(hipSetDevice(device_index));

  // Truoc day 5 event duoc tao MOI BUOC va khong bao gio huy. Gio tao mot lan
  // cho moi thread (moi GPUWorker mot thread, moi thread mot thiet bi).
  static thread_local hipEvent_t event_rmsnorm = nullptr;
  static thread_local hipEvent_t event_router_topk = nullptr;
  static thread_local hipEvent_t events_e_agg[MAXIMUM_GPU] = {};
  if (!event_rmsnorm) {
    HIP_CHECK(hipEventCreateWithFlags(&event_rmsnorm, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&event_router_topk, hipEventDisableTiming));
    for (int i = 0; i < EXPERT_PARALLELISM; ++i)
      HIP_CHECK(hipEventCreateWithFlags(&events_e_agg[i], hipEventDisableTiming));
  }

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
  int odd_tcap = p->seq_len / 2;
  if (odd_tcap < 1) odd_tcap = 1;
  const size_t even_stride = (size_t)even_tcap;
  const size_t odd_stride = (size_t)odd_tcap;

  const int own_slice = device_index - base_device_index;

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    // rmsnorm ghi THANG bf16 vao pre_qkv_bf16: ban fp32 (dev_s->t) khong con
    // consumer nao trong duong 120b.
    getp_rmsnorm_bf16(ext->pre_qkv_bf16, dev_x,
                      dev_w->rms_attn_w + 1ll * l * hidden_dim,
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
          ext->attn_o_bf16, dev_s->tb,
          k_layer, v_layer, q_ptr, ext->mask_on, attn_sinks_layer, head_dim,
          p->n_attn_heads, p->n_kv_heads, pos, p->seq_len, p->sliding_window,
          (int)l, BATCH_SIZE, cache_tcap, compute_stream);
    }

    __hip_bfloat16 *dev_w_o =
        dev_w->w_o_bf16 + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    __hip_bfloat16 *dev_b_o = dev_w->b_o_bf16 + 1ll * l * hidden_dim;
    // attention da ghi bf16 vao attn_o_bf16; matmul cong thang vao dev_x.
    getp_matmul_attn_o_bf16_acc(dev_x, ext->attn_o_bf16, dev_w_o, dev_b_o,
                                head_dim * p->n_attn_heads, hidden_dim,
                                BATCH_SIZE, compute_stream);

    // Router doc ext_t fp32, peer copy doc ext_t_bf16: sinh ca hai trong mot
    // luot, thay cho mot kernel doi kieu rieng tren memory_stream.
    getp_rmsnorm_f32_bf16(
        ext->ext_t + (size_t)own_slice * BATCH_SIZE * hidden_dim,
        ext->ext_t_bf16 + (size_t)own_slice * BATCH_SIZE * hidden_dim,
        dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim,
        BATCH_SIZE, hidden_dim, compute_stream);
    HIP_CHECK(hipEventRecord(event_rmsnorm, compute_stream));

    __hip_bfloat16 *dev_w_router =
        dev_w->w_router_bf16 + 1ll * l * hidden_dim * n_experts;
    __hip_bfloat16 *dev_b_router = dev_w->b_router_bf16 + 1ll * l * n_experts;
    getp_matmul_router_bf16(
      dev_s->router_score,
      ext->ext_t + (size_t)(device_index - base_device_index) * BATCH_SIZE * hidden_dim,
      dev_w_router, dev_b_router, hidden_dim, n_experts,
      BATCH_SIZE, compute_stream);

    getp_router_topk_softmax_batch(
        dev_s->router_score, n_experts, p->experts_per_token,
        ext->ext_topk_v + (size_t)(device_index - base_device_index) * BATCH_SIZE * p->experts_per_token,
        ext->ext_topk_i + (size_t)(device_index - base_device_index) * BATCH_SIZE * p->experts_per_token,
        BATCH_SIZE, compute_stream);
    HIP_CHECK(hipEventRecord(event_router_topk, compute_stream));

    // ext_t_bf16 da duoc rmsnorm ghi trong compute_stream truoc event_rmsnorm.
    HIP_CHECK(hipStreamWaitEvent(memory_stream, event_rmsnorm));

    for (int delta = 1; delta < EXPERT_PARALLELISM; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      GPUWorker *peer_worker = &workers[i];
      int peer_device_index = peer_worker->device_index;
      RunStateExt *peer_ext = ext_get(peer_device_index);
      if (peer_device_index != device_index) {
        HIP_CHECK(hipMemcpyPeerAsync(
            peer_ext->ext_t_bf16 + (size_t)(device_index - base_device_index) * BATCH_SIZE * hidden_dim,
            peer_device_index,
            ext->ext_t_bf16 + (size_t)(device_index - base_device_index) * BATCH_SIZE * hidden_dim,
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
            peer_ext->ext_topk_i + (size_t)(device_index - base_device_index) * BATCH_SIZE * p->experts_per_token,
            peer_device_index,
            ext->ext_topk_i + (size_t)(device_index - base_device_index) * BATCH_SIZE * p->experts_per_token,
            device_index,
            sizeof(int) * (size_t)BATCH_SIZE * (size_t)p->experts_per_token,
            memory_stream));
        HIP_CHECK(hipMemcpyPeerAsync(
            peer_ext->ext_topk_v + (size_t)(device_index - base_device_index) * BATCH_SIZE * p->experts_per_token,
            peer_device_index,
            ext->ext_topk_v + (size_t)(device_index - base_device_index) * BATCH_SIZE * p->experts_per_token,
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

    int total_blocks_mlp1 = build_moe_block_schedule(
      ext->e_offsets, experts_per_device, MATMUL_MLP1_BLOCK_ROWS,
      ext->blk_counts, ext->blk_offsets, compute_stream);
  
  moe_scatter_acts_to_expert_frombf16(
      ext->a_in,
      ext->ext_t_bf16,
      ext->e_dev,
      total_pairs, hidden_dim, compute_stream);
  
  // (memset 35,4 MB da bo: moe_gather_pairs gio ghi de toan bo vung nay)
  
  __hip_bfloat16 *w1_base = dev_w->w_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim * hidden_dim;
  __hip_bfloat16 *b1_base = dev_w->b_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim;
  
  launch_mlp1_swiglu_bf16_bucketed_outbf16(
      ext->gate_up_bf16, ext->a_in, w1_base, b1_base, ext->e_offsets,
      ext->blk_offsets, hidden_dim, p->intermediate_dim, experts_per_device,
      total_blocks_mlp1, p->swiglu_limit, compute_stream);
  
  __hip_bfloat16 *w2_base = dev_w->w_mlp2 + 1ll * l * experts_per_device * hidden_dim * p->intermediate_dim;
  __hip_bfloat16 *b2_base = dev_w->b_mlp2 + 1ll * l * experts_per_device * hidden_dim;
  
#if MATMUL_MLP2_BLOCK_ROWS != MATMUL_MLP1_BLOCK_ROWS
  // mlp2 giai ma blk_offsets bang BM cua rieng no, nen khi hai BM khac nhau thi
  // phai co lich rieng. Ghi de cung buffer la an toan: moi thu tren compute_stream,
  // nen kernel nay khong the chay truoc khi mlp1 doc xong.
  int total_blocks_mlp2 = build_moe_block_schedule(
      ext->e_offsets, experts_per_device, MATMUL_MLP2_BLOCK_ROWS,
      ext->blk_counts, ext->blk_offsets, compute_stream);
#else
  const int total_blocks_mlp2 = total_blocks_mlp1;
#endif
  
  launch_mlp2_partial_bf16_bucketed_frombf16(
      ext->z_partial, ext->gate_up_bf16, w2_base, b2_base, ext->w_dev,
      ext->e_offsets, ext->blk_offsets, ext->e_dev, p->intermediate_dim,
      hidden_dim, experts_per_device, total_blocks_mlp2, compute_stream);
  
    // Lat cat own_slice di thang vao dev_x; cac lat cat khac van ra ext_e_agg
    // vi peer se keo chung. Bo han mot vecadd (doc 17,7 MB + ghi 17,7 MB).
    moe_gather_pairs_acc(ext->ext_e_agg, dev_x,
                         own_slice * BATCH_SIZE,
                         own_slice * BATCH_SIZE + BATCH_SIZE,
                         ext->z_partial, ext->pair_pos, hidden_dim,
                         p->experts_per_token, EXPERT_PARALLELISM * BATCH_SIZE,
                         compute_stream);

    sync_workers(compute_stream, sync_point);

    for (int delta = 1, j = 0; delta < EXPERT_PARALLELISM; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      GPUWorker *peer_worker = &workers[i];
      int peer_device_index = peer_worker->device_index;
      RunStateExt *peer_ext = ext_get(peer_device_index);
      if (peer_device_index != device_index) {
        HIP_CHECK(hipMemcpyPeerAsync(
            ext->peer_e_agg + (size_t)j * BATCH_SIZE * hidden_dim, device_index,
            peer_ext->ext_e_agg + (size_t)(device_index - base_device_index) * BATCH_SIZE * hidden_dim,
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

  // Use pinned memory for faster device-to-host transfer
  int *next_host = nullptr;
  HIP_CHECK(hipHostMalloc(&next_host, sizeof(int) * BATCH_SIZE, hipHostMallocDefault));
  HIP_CHECK(hipMemcpyAsync(next_host, dev_s->topk_i, sizeof(int) * BATCH_SIZE,
                           hipMemcpyDeviceToHost, compute_stream));
  HIP_CHECK(hipStreamSynchronize(compute_stream));
  return next_host;
}


int *getp_forward_20b(Transformer * /*transformer*/,
                      DeviceTransformer **dev_transformeres, GPUWorker *worker,
                      int token[], int pos, int *mask_on, int batch_size) {
  
  
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
  
  hipStream_t h2d_stream = dev_transformeres[device_index]->h2d_stream;

  // Use async copies for better performance
  HIP_CHECK(hipMemcpyAsync(ext->mask_on, mask_on, sizeof(int) * batch_size, 
                           hipMemcpyHostToDevice, h2d_stream));
  HIP_CHECK(hipMemcpyAsync(dev_s->topk_i, token, sizeof(int) * batch_size,
                           hipMemcpyHostToDevice, h2d_stream));
  HIP_CHECK(hipStreamSynchronize(h2d_stream));
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
  int odd_tcap = p->seq_len / 2;
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
          ext->attn_o_bf16, dev_s->tb,
          k_layer, v_layer, q_ptr, ext->mask_on, attn_sinks_layer, head_dim,
          p->n_attn_heads, p->n_kv_heads, pos, p->seq_len, p->sliding_window,
          (int)l, batch_size, cache_tcap, nullptr);
    }

    __hip_bfloat16 *dev_w_o =
        dev_w->w_o_bf16 + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    __hip_bfloat16 *dev_b_o = dev_w->b_o_bf16 + 1ll * l * hidden_dim;
    // getp_new_matmul(dev_s->tb2, dev_s->tb, dev_w_o, dev_b_o,
    //                     head_dim * p->n_attn_heads, hidden_dim,
    //                     batch_size, nullptr);
    getp_matmul_attn_o_bf16_acc(dev_x, ext->attn_o_bf16, dev_w_o, dev_b_o,
                                head_dim * p->n_attn_heads, hidden_dim,
                                batch_size, nullptr);

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

  // Use pinned memory for faster device-to-host transfer
  int *next_host = nullptr;
  HIP_CHECK(hipHostMalloc(&next_host, sizeof(int) * batch_size, hipHostMallocDefault));
  hipStream_t d2h_stream = dev_transformeres[device_index]->d2h_stream;
  HIP_CHECK(hipMemcpyAsync(next_host, dev_s->topk_i, sizeof(int) * batch_size,
                           hipMemcpyDeviceToHost, d2h_stream));
  HIP_CHECK(hipStreamSynchronize(d2h_stream));
                      
  return next_host;
}
