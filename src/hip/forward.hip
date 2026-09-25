#include <chrono>
#include <algorithm>
#include <vector>
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

// Padding on the fast axis of the LDS tile. At BM=BN=128 the stride between two k planes is
// 64 dwords, a multiple of the 32 banks, so every group of 16 lanes reads and writes the same
// bank. Four extra elements make the stride 66 dwords, which spreads it across the banks.
#ifndef GETP_LDS_PAD
#define GETP_LDS_PAD 4
#endif
// GETP_LDS_PAD above still pads the ROW axis of the [BK][BM+PAD] layout, and is still used
// verbatim by matmul_attn_o_bf16_kernel_tuned and
// matmul_logits_argmax_bf16_kernel_tuned. The QKV kernel has moved to the slot layout
// [k/4][row][4]: its padding counts in 8-byte SLOTS rather than in shorts, and the validity
// condition depends on BK through NKC = BK/8, which differs per kernel. That is why QKV has
// its OWN macro; reusing one macro for two different UNITS is a reliable way to get it wrong.
//   -1 = derive the smallest bank-conflict-free value (see GetpQkvLdsTile).
#ifndef GETP_QKV_LDS_PAD_SLOTS
#define GETP_QKV_LDS_PAD_SLOTS (-1)
#endif

// Fold the RoPE rotation into the QKV kernel epilogue. See the long note just above the
// epilogue: the partner at dim+head_dim/2 lands in the same lane, so no shuffle is needed.
#ifndef GETP_ROPE_FUSED
#define GETP_ROPE_FUSED 1   // on by default (measured end to end: -0,73 ms/step); set 0 to A/B
#endif
// One step further: write the KV cache slot directly as bf16 from the epilogue and drop
// kv_store_pair. A separate flag, because it trades store efficiency (2 bytes/lane) for 12,6 MB/layer.
#ifndef GETP_ROPE_FUSED_KV
#define GETP_ROPE_FUSED_KV 1   // on by default (the same rotation, applied to K); set 0 to A/B
#endif
#if GETP_ROPE_FUSED_KV && !GETP_ROPE_FUSED
#error "GETP_ROPE_FUSED_KV=1 requires GETP_ROPE_FUSED=1"
#endif

// --- smallfuse: two independent switches, OFF by default ---------------------------
// Write q already in bf16 and already scaled by 1/sqrt(d): attention rounds it to bf16 at its
// first use anyway, and the factor is 2^-3 so the multiply leaves the mantissa alone. See qbf16.
#ifndef GETP_Q_BF16
#define GETP_Q_BF16 1   // on by default (attention 445 -> 355 us/call); set 0 to A/B
#endif
#if GETP_Q_BF16 && !GETP_ROPE_FUSED
#error "GETP_Q_BF16 requires GETP_ROPE_FUSED: the non-rope epilogue writes q elsewhere"
#endif

// Send only the non-zero rows of the e_agg block between GPUs. On the 120B, 57 % of the rows
// are zero (measured). See the rowskip notes. Only meaningful at EXPERT_PARALLELISM > 2.
// Merge the fourteen row-list launches per layer into two: one BLOCK per slice instead of one
// LAUNCH per slice. Each of the old launches cost 8 us but sat 60 us apart, so almost all of
// the cost was launching them in sequence rather than the work itself. Same arithmetic, only a
// different launch count, so it is bit-exact. Measured on the 120B: 22.807 -> 23.136 tok/s
// (+1,4 %), md5 efe1096ff64c unchanged. Set 0 to go back to one launch per slice.
#ifndef GETP_RS_MERGE
#define GETP_RS_MERGE 1
#endif

// Sync point 2 orders the exchange with one EVENT per peer instead of a drain plus a barrier.
// The old sequence was: GPU finishes gather_pairs -> the host wakes out of hipStreamSynchronize
// -> eight threads meet at a barrier -> the host issues the pulls -> only then does the GPU
// pull. That whole round trip through the host sat in a 1,57 ms gap per layer. With events the
// pulls are already queued and start the instant the peer has written. A HOST barrier stays
// here, but WITHOUT the drain: it only guarantees that every thread has ISSUED its
// hipEventRecord before anyone waits, since waiting on an event that has not been recorded yet
// returns immediately. REQUIRES GETP_AGSKIP: it puts the row counts in g_ag_cnt before the sp1
// barrier, which is what lets the blocking hipMemcpy here be deleted. That read is exactly what
// made the older event path pick up garbage and report hipErrorInvalidValue. Sync point 1 KEEPS
// its drain, deliberately: it is cheap (0,20 ms against 0,86 ms for sp2) and it stops two ranks
// drifting more than one layer apart, so an event slot is never reused under a waiter. The
// older attempt converted BOTH sync points, so the host ran arbitrarily far ahead - 15 % slower.
// Measured on the 120B: 23.910 -> 24.341 tok/s (+1,8 %), output hash unchanged.
// Emit moe_gather_pairs one destination slice at a time, so the peer that pulls from us first
// can start as soon as ITS slice is ready instead of waiting for all eight. Rank D pulls from
// D+1 first, and D+1 owes it slice D, so each rank emits slice (own-1) first and records an
// early event for it. Splitting the b range touches no arithmetic: every e_agg cell is still
// written exactly once and each token's sum keeps its order, so this is bit-exact.
// Measured on the 120B: 24.313 -> 24.685 tok/s (+1,5 %), output hash unchanged.
#ifndef GETP_GPSLICE
#define GETP_GPSLICE 1
#endif

#ifndef GETP_EVAGG
#define GETP_EVAGG 1
#endif

// At EXPERT_PARALLELISM = 2, moe_gather_pairs writes the peer's slice first, then the device-side
// sync point, then our own rows into the residual - so the peer's pull overlaps our own rows.
// Bit-exact. Set 0 for one launch before the sync point.
#ifndef GETP_GP_OWN_LAST
#define GETP_GP_OWN_LAST 1
#endif

// The all-gather sends only the rows each peer actually needs, mirroring GETP_ROWSKIP, which
// trimmed the reduce-scatter leg alone. The old code sent all BATCH_SIZE rows to every peer,
// 31,0 MB per layer, while each rank pair uses only 326/768 = 42,4 % of them - nothing reads
// the rest, because ext_t_bf16 is only ever fetched through the pair list (tok_idx/e_dev).
// So dropping them is bit-exact. 31,0 -> 13,2 MB per layer. Measured on the 120B: 23.022 ->
// 23.910 tok/s (+3,9 %), output hash unchanged. Set 0 to go back to sending every row.
// Send the reduce-scatter leg as bf16 instead of fp32. After GETP_AGSKIP the outbound leg is
// already bf16, which left the return leg as the larger of the two purely because of its element
// type: 26,6 MB against 13,2 MB per layer. Halving it costs no extra kernel and no extra launch.
// This is the one change here that is NOT bit-exact. The rounding applies only to what crosses
// the wire - a rank's own slice still adds into dev_x in fp32 - and it is measured, not argued:
// METEOR 0,5608 -> 0,5603 and BERTScore 0,9814 -> 0,9805, against thresholds of 0,3 and 0,9.
// It is gated on rs_on, so at EXPERT_PARALLELISM = 2 the 20B path never takes it and keeps its
// exact-output gate (720709b86d36) as a cheap check for everything else.
// Measured on the 120B: 24.902 -> 26.011 tok/s (+4,5 %).
#ifndef GETP_EAGG_BF16
#define GETP_EAGG_BF16 1
#endif

#ifndef GETP_AGSKIP
#define GETP_AGSKIP 1
#endif

#ifndef GETP_ROWSKIP
#define GETP_ROWSKIP 1   // on by default; turns itself off when EP <= 2. Set 0 to A/B
#endif

// The expert-output gather and the row scatter-add move 16 bytes per thread instead of 2-4: the
// same additions in the same order on every element, so bit-exact. Set 0 for the scalar kernels.
#ifndef GETP_SMALL_VEC
#define GETP_SMALL_VEC 1
#endif

#ifndef GETP_SMALL_TOPK
#define GETP_SMALL_TOPK 1   // on by default (measured end to end: -2,09 ms/step); set 0 to A/B
#endif
#ifndef GETP_SMALL_NOSCATTER
#define GETP_SMALL_NOSCATTER 0   // MEASURED and LOST: dropping the scatter kernel (-1,42 ms/step)
                                 // but mlp1 has to gather rows itself, so +1,66 ms/step.
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

// rmsnorm has moved BELOW the bf16 conversion helpers (look for rmsnorm_kernel_out), because
// it needs f32_to_bf16bits to write bf16 directly.

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
// rmsnorm with a selectable output: fp32, bf16, or BOTH.
//
// The fp32 value is computed EXACTLY as before - weight * (ss * x), two multiplies with no
// add, so the compiler cannot fuse them into an FMA and change the result.
// Since writing fp32 to HBM and reading it back is BIT-EXACT, the bf16 output written here is
// exactly the bits a separate fp32_to_bf16_kernel would have produced. Dropping that kernel is
// therefore a BIT-EXACT change, not an approximation.
//
// Why no LDS exchange is needed: the rounding runs on the SAME thread that produced the value,
// on that same element. The bf16 output is 2 bytes per lane, so the stride
// is 2 bytes -> one wave writes 128 contiguous bytes.
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


// RMSNorm with one wave per row, for size == 2880 (any other size keeps the block kernel).
// rmsnorm_kernel_out gives each row a 1024-thread block whose sum of squares goes through ten LDS
// levels, each behind a barrier: 65 / 57 us per call on the 20B, 0.4 TB/s. The wave keeps the same
// 1024 partials, p[j] = x[j]^2 + x[j+1024]^2 (+ x[j+2048]^2) accumulated in that order, and the same
// pairwise tree over them, p[j] += p[j+s] for s = 512 ... 1: lane l holds p[4l + 256m + c], levels 512
// and 256 pair registers of the lane, 128 ... 4 are shuffles, 2 and 1 are in the lane again. So the
// same bits; the row stays in registers for the output pass.
#ifndef GETP_RMSNORM_WAVE
#define GETP_RMSNORM_WAVE 1
#endif
// Weights prefetched before the reduction (WPF) below this many rows: with few waves per SIMD the
// weight load's latency is exposed (120B, 768 rows: 48.3 -> 41.1 us for both norms), with more it
// only costs registers (20B, 1536 rows: 72.4 -> 74.5 us).
#ifndef GETP_RMSNORM_WPF_ROWS
#define GETP_RMSNORM_WPF_ROWS 1024
#endif
#if GETP_RMSNORM_WAVE
// x is not __restrict__: the final norm runs in place (o == x). Each row is read whole before it
// is written, by one wave, so that is safe.
template <bool WRITE_F32, bool WRITE_BF16, bool WPF>
__global__ __launch_bounds__(256) void rmsnorm_wave_kernel(float *o, __hip_bfloat16 *ob,
                                                          const float *x,
                                                          const float *__restrict__ weight,
                                                          int rows, int size) {
  constexpr int SIZE = 2880, NK = (SIZE + 255) / 256;   // 12 float4 slots per lane, the last partial
  const int lane = threadIdx.x & 63;
  const int r = blockIdx.x * 4 + (threadIdx.x >> 6);
  if (r >= rows) return;
  const size_t row = (size_t)r * SIZE;
  const float4 *xr = reinterpret_cast<const float4 *>(x + row);
  // slot k = m + 4q holds elements 4*lane + 256*k + (0..3), i.e. partial j = 4*lane + 256*m + c, term q
  float4 v[NK];
  #pragma unroll
  for (int k = 0; k < NK; ++k)
    if (4 * lane + 256 * k < SIZE) v[k] = xr[lane + 64 * k];
  const float4 *wr = reinterpret_cast<const float4 *>(weight);
  float4 gw[NK];
  if constexpr (WPF) {
    #pragma unroll
    for (int k = 0; k < NK; ++k)
      if (4 * lane + 256 * k < SIZE) gw[k] = wr[lane + 64 * k];
  }
  float p[4][4];   // p[m][c]
  #pragma unroll
  for (int m = 0; m < 4; ++m) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    #pragma unroll
    for (int q = 0; q < 3; ++q) {
      const int k = m + 4 * q;
      if (k < NK && 4 * lane + 256 * k < SIZE) {
        // Explicit fma, as the block kernel's rolled loop compiles (v_fmac acc, t, t). Written as
        // acc += t * t and unrolled, the backend fuses the FIRST square instead of the new one.
        const float4 t = v[k];
        a0 = __builtin_fmaf(t.x, t.x, a0); a1 = __builtin_fmaf(t.y, t.y, a1);
        a2 = __builtin_fmaf(t.z, t.z, a2); a3 = __builtin_fmaf(t.w, t.w, a3);
      }
    }
    p[m][0] = a0; p[m][1] = a1; p[m][2] = a2; p[m][3] = a3;
  }
  float w[4];
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float l512a = p[0][c] + p[2][c], l512b = p[1][c] + p[3][c];   // level 512
    w[c] = l512a + l512b;                                               // level 256
  }
  #pragma unroll
  for (int d = 32; d >= 1; d >>= 1)                                     // levels 128 ... 4
    #pragma unroll
    for (int c = 0; c < 4; ++c) w[c] += __shfl_down(w[c], d, 64);
  float ss = (w[0] + w[2]) + (w[1] + w[3]);                             // levels 2, 1
  ss = __shfl(ss, 0, 64);
  ss /= size;   // the runtime divisor, as in the block kernel
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  #pragma unroll
  for (int k = 0; k < NK; ++k) {
    if (4 * lane + 256 * k < SIZE) {
      const float4 t = v[k], g = WPF ? gw[k] : wr[lane + 64 * k];
      float4 y;
      y.x = g.x * (ss * t.x); y.y = g.y * (ss * t.y); y.z = g.z * (ss * t.z); y.w = g.w * (ss * t.w);
      if constexpr (WRITE_F32) reinterpret_cast<float4 *>(o + row)[lane + 64 * k] = y;
      if constexpr (WRITE_BF16) {
        uint2 u;
        u.x = (unsigned)f32_to_bf16bits(y.x) | ((unsigned)f32_to_bf16bits(y.y) << 16);
        u.y = (unsigned)f32_to_bf16bits(y.z) | ((unsigned)f32_to_bf16bits(y.w) << 16);
        reinterpret_cast<uint2 *>(reinterpret_cast<unsigned short *>(ob) + row)[lane + 64 * k] = u;
      }
    }
  }
}
#define GETP_RMSNORM_LAUNCH(WF, WB, o, ob, x, w, rows, dim, st)                                   \
  if ((dim) == 2880) {                                                                          \
    if ((rows) <= GETP_RMSNORM_WPF_ROWS)                                                        \
      rmsnorm_wave_kernel<WF, WB, true><<<(unsigned)(((rows) + 3) / 4), 256, 0, (st)>>>(         \
          o, ob, x, w, rows, dim);                                                              \
    else                                                                                        \
      rmsnorm_wave_kernel<WF, WB, false><<<(unsigned)(((rows) + 3) / 4), 256, 0, (st)>>>(        \
          o, ob, x, w, rows, dim);                                                              \
    return;                                                                                     \
  }
#else
#define GETP_RMSNORM_LAUNCH(WF, WB, o, ob, x, w, rows, dim, st)
#endif

void getp_rmsnorm(float *o, float *x, float *weight, int batch_size, int dim,
                  hipStream_t stream) {
  GETP_RMSNORM_LAUNCH(true, false, o, nullptr, x, weight, batch_size, dim, stream)
  dim3 blockDim(1024);
  dim3 gridDim(1, batch_size);
  rmsnorm_kernel_out<true, false>
      <<<gridDim, blockDim, sizeof(float) * (blockDim.x + 1), stream>>>(
          o, nullptr, x, weight, dim);
}

// bf16 only: for the case where nothing reads the fp32 form any more (the pre-qkv path).
void getp_rmsnorm_bf16(__hip_bfloat16 *ob, float *x, float *weight,
                       int batch_size, int dim, hipStream_t stream) {
  GETP_RMSNORM_LAUNCH(false, true, nullptr, ob, x, weight, batch_size, dim, stream)
  dim3 blockDim(1024);
  dim3 gridDim(1, batch_size);
  rmsnorm_kernel_out<false, true>
      <<<gridDim, blockDim, sizeof(float) * (blockDim.x + 1), stream>>>(
          nullptr, ob, x, weight, dim);
}

// Both: for the case where a consumer still reads fp32 (the router reads ext_t).
void getp_rmsnorm_f32_bf16(float *o, __hip_bfloat16 *ob, float *x,
                           float *weight, int batch_size, int dim,
                           hipStream_t stream) {
  GETP_RMSNORM_LAUNCH(true, true, o, ob, x, weight, batch_size, dim, stream)
  dim3 blockDim(1024);
  dim3 gridDim(1, batch_size);
  rmsnorm_kernel_out<true, true>
      <<<gridDim, blockDim, sizeof(float) * (blockDim.x + 1), stream>>>(
          o, ob, x, weight, dim);
}

// GETP_KV_HEADMAJOR (default 1): a layer's KV cache is [batch][kv_head][t][64] instead of
// [t][batch][kv_dim], each (sequence, kv_head) region padded by GETP_KV_PAD elements. In the old
// layout every key an attention workgroup reads is its own 128-byte piece, batch x kv_dim x 2 bytes
// from the next one (1.5 MB on the 20B), and decode attention ran at 0.8 TB/s with the L1 waiting on
// L2 88 % of the time. Head-major makes a 16-key tile one 2 KB block - but unpadded it was SLOWER on
// full-context layers: each region is CAP x 128 bytes, so every region starts on the same memory
// channel while the workgroups advance through t together. The pad staggers them. Op test (us, old
// layout / head-major / head-major + 384 pad): 20B full T=1024 4051 / 4359 / 2651, T=512 2070 /
// 1924 / 1358, 20B sliding 653 / 395 / 388, 120B full T=1024 2098 / 2919 / 1343 (pad 64 bad,
// 128-256 good, 384 best, 512 worse). Only addresses change: bit-exact.
// GETP_KV_OFF(t, b, col, B, KVD, CAP) is the
// element offset of (ring slot t, batch row b, column col = kv_head * 64 + d) inside one layer, for B
// batch rows, KVD = kv_dim columns and a ring of CAP slots; it is additive in t and (b, col), so a
// slot pointer (t, 0, 0) plus the offset (0, b, col) addresses the element. head_dim is 64.
// transformer.cpp (included before this file) sizes the cache with the same macros.
#ifndef GETP_KV_HEADMAJOR
#define GETP_KV_HEADMAJOR 1
#endif
// GETP_KV8_K / GETP_KV8_V: that half of the KV cache is stored as int8 (see the note below).
#ifndef GETP_KV8_K
#define GETP_KV8_K 0
#endif
#ifndef GETP_KV8_V
#define GETP_KV8_V 1
#endif
// The pad is counted in elements, so an int8 tensor needs twice the element count for the same
// byte stagger between regions.
#ifndef GETP_KV_PAD
#if GETP_KV8_K || GETP_KV8_V
#define GETP_KV_PAD 768
#else
#define GETP_KV_PAD 384
#endif
#endif
#if GETP_KV_HEADMAJOR
#define GETP_KV_OFF(t, b, col, B, KVD, CAP)                                                    \
  (((size_t)(b) * (size_t)((KVD) >> 6) + (size_t)((col) >> 6)) *                               \
       ((size_t)(CAP) * 64 + GETP_KV_PAD) + (size_t)(t) * 64 + (size_t)((col) & 63))
// Elements a layer occupies beyond CAP * B * KVD: its padding.
#define GETP_KV_LAYER_PAD(B, KVD) ((size_t)(B) * (size_t)((KVD) >> 6) * GETP_KV_PAD)
#else
#define GETP_KV_OFF(t, b, col, B, KVD, CAP) \
  ((size_t)(t) * (size_t)(B) * (size_t)(KVD) + (size_t)(b) * (size_t)(KVD) + (size_t)(col))
#define GETP_KV_LAYER_PAD(B, KVD) ((size_t)0)
#endif

// int8 KV cache (GETP_KV8_K, GETP_KV8_V; only V by default). NOT bit-exact; 0 keeps bf16.
// Decode attention reads the whole cache every step and is bound by those bytes: a 64-dim head
// vector costs 128 bytes in bf16, 68 as int8 codes plus one fp32 scale per (position, sequence,
// kv_head) vector.
// Layout, no new allocation: code e sits at byte e of the bf16-sized allocation (the same
// GETP_KV_OFF offsets, counted in bytes) and the scale of the vector starting at element e is float
// e/64 of the allocation's upper half (getp_kv8_scales), so a 16-key tile has 16 consecutive scales.
// Writer: the qkv epilogue. A row's 64 dims sit on the 16 lanes of a lane group x 4 values, so amax
// takes four xor-shuffles; the code is round(x * 127 / amax) + 128.
// Reader: one 16-byte load per lane per tile, dequantized while the LDS tile is staged: v_perm_b32
// against 0x64646464 makes the fp16 value 1024 + code exactly, v_pk_add_f16 subtracts 1152 (exact)
// and v_pk_mul_f16 applies the scale (one fp16 rounding). That tensor's MFMA then runs in fp16 on
// real values; for V the other operand is P = e * 2^15 in fp16 (normal down to e ~ 2e-9), undone in
// the final normalisation. Keeping the 1152 offset inside the MFMA and subtracting 1152 * sum(P)
// afterwards looks exact but is not: the MFMA flushes fp16 denormals, so small P vanish from the
// product but not from the correction (output error 290 % once the scores are peaky).
// Why only V: gpt-oss keys carry a large component shared across positions and very unequal
// channels (the largest dim's rms up to 23x the median), so per-vector int8 costs K far more than V.
// 20B METEOR, bf16 0.534: V int8 0.519, K and V int8 0.342.
#define GETP_KV8_ANY (GETP_KV8_K || GETP_KV8_V)
#if GETP_KV8_ANY && (!GETP_KV_HEADMAJOR)
#error "GETP_KV8_* needs GETP_KV_HEADMAJOR (one scale per contiguous 64-element vector)"
#endif
// Element `off` of a cache allocation: a byte offset for an int8 tensor, an element one for bf16.
#define GETP_KV_PTR8(cache, off) \
  reinterpret_cast<__hip_bfloat16 *>(reinterpret_cast<signed char *>(cache) + (size_t)(off))
#define GETP_KV_PTR16(cache, off) (reinterpret_cast<__hip_bfloat16 *>(cache) + (size_t)(off))
#if GETP_KV8_K
#define GETP_KV_PTR_K GETP_KV_PTR8
#else
#define GETP_KV_PTR_K GETP_KV_PTR16
#endif
#if GETP_KV8_V
#define GETP_KV_PTR_V GETP_KV_PTR8
#else
#define GETP_KV_PTR_V GETP_KV_PTR16
#endif
// The scale of the vector whose first element is at offset `off` (a multiple of 64).
#define GETP_KV_SIDX(off) ((size_t)(off) >> 6)
// The stored code of q in [-127, 127]: q + 128 (the reader stages it as fp16 1024 + code).
#define GETP_KV8_CODE(q) ((signed char)((q) + 128))

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

// Return an UPPER BOUND on the block count, computed on the host from constants, instead of
// reading blk_offsets[E] back (one stream drain per layer). Each expert rounds up
// by at most 1 block, so (max_pairs + BM - 1)/BM + E is always enough. Both MoE kernels
// return at once at `if (base_all >= n) return;` for the blocks past the real total.
static inline int build_moe_block_schedule(const int *offsets, int E, int BM,
                                           int *blk_counts, int *blk_offsets,
                                           int max_pairs, hipStream_t stream) {

  dim3 b(256), g((E + b.x - 1) / b.x);
  moe_blocks_per_expert_kernel<<<g, b, 0, stream>>>(offsets, E, BM, blk_counts);
  exclusive_scan_small_kernel<<<dim3(1), dim3(256), sizeof(int) * 256,
                                stream>>>(blk_counts, blk_offsets, E);
    return (max_pairs + BM - 1) / BM + E;
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
    // Do not read e_offsets back to the host any more: the only value it served (cap_pairs)
  // is unused, and every read was one drain of compute_stream in the middle of a layer.
  HIP_CHECK(hipMemsetAsync(e_counts, 0, sizeof(int) * E, stream));
  moe_fill_local_pos_kernel<<<dim3(B), dim3(128), 0, stream>>>(
      local_ids, local_wts, n_local, B, K, E, e_offsets, e_counts, tok_idx_out,
      w_out, pair_pos);
  return 0;
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
  const int *tok_idx, const int *total_pairs_dev, int H) {
int pos = blockIdx.x;
if (pos >= *total_pairs_dev) return;   // read from the device, no host drain
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
  const int *total_pairs_dev, int max_pairs, int H, hipStream_t stream) {
dim3 block(256), grid(max_pairs);
moe_scatter_acts_to_expert_frombf16_kernel<<<grid, block, 0, stream>>>(
    a_in, x_bf16, tok_idx, total_pairs_dev, H);
}

// Fast-axis padding of the LDS tile in the two MoE kernels. This constant 8 used to be written
// out in BOTH the kernel body and the launcher; letting them disagree means allocating
// too little shared memory and silently corrupting the tile.
#ifndef MATMUL_MLP1_LDS_PAD
#define MATMUL_MLP1_LDS_PAD 4
#endif
// mlp1 has moved to the [k/4][row][4] LDS layout: padding counts in SLOTS (4 shorts = 8 bytes),
// not in shorts. Two slots = 16 bytes per k0 plane; see Mlp1LdsPlan.
#ifndef MATMUL_MLP1_LDS_PAD_SLOTS
#define MATMUL_MLP1_LDS_PAD_SLOTS 2
#endif
// mlp2 has also moved to the [k/4][row][4] LDS layout: padding counts in SLOTS (4 shorts =
// 8 bytes) rather than in shorts, so it needs its OWN macro. The old MATMUL_MLP2_LDS_PAD no
// longer means anything; fail loudly if anyone still passes the old -D, rather than letting it
// become a silent no-op that later reads as measurement noise.
// (GETP_LDS_PAD for qkv/attn_o/logits and MATMUL_MLP1_LDS_PAD are UNCHANGED.)
#ifdef MATMUL_MLP2_LDS_PAD
#error "MATMUL_MLP2_LDS_PAD no longer has any effect: use MATMUL_MLP2_LDS_PAD_SLOTS (see Mlp2LdsPlan) and stop passing the old one."
#endif
// -1 = derive the smallest bank-clean padding from (BM, BNt, BK). BK=32 -> 2, BK=64 -> 1.
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
// Derived from the tile: the kernel body uses TILE_N = 64 * (BM/WM)*(BN/WN) threads.
// Overriding with -D is only allowed if it still equals this expression (there is a static_assert).
#define MATMUL_MLP1_WAVES_PER_BLOCK (((MATMUL_MLP1_BLOCK_ROWS) / (MATMUL_MLP1_WARP_TILE_M)) * ((MATMUL_MLP1_BLOCK_COLS) / (MATMUL_MLP1_WARP_TILE_N)))
#endif
#ifndef NUM_CTA_MLP1
#define NUM_CTA_MLP1 2
#endif

#ifndef MATMUL_MLP2_BLOCK_ROWS
// 128, not 96. At the old 32 x 32 wave tile mlp2 packed 1.78x less MFMA into
// each LDS read than mlp1 (1.50 against 2.67 per ds_read_b64) - the whole of
// why it reached 34% of MFMA peak where mlp1 reaches 55%. The 64 x 32 wave tile
// that BM=128 allows lifts that to 2.40 (4 A + 6 B reads per 24 MFMA). 128 was
// long believed to give WRONG output; it does not - the difference was split-K
// (see MATMUL_MLP2_MAX_SPLITS). With that pinned, 128 is bit-exact with 96 and
// 3.4% faster end to end (62.9k against 60.8k tok/s, 8 GPU, 1024 steps).
#define MATMUL_MLP2_BLOCK_ROWS 128
#endif
#ifndef MATMUL_MLP2_BLOCK_COLS
#define MATMUL_MLP2_BLOCK_COLS 64
#endif
#ifndef MATMUL_MLP2_BLOCK_DEPTH
#define MATMUL_MLP2_BLOCK_DEPTH 32
#endif
#ifndef MATMUL_MLP2_WARP_TILE_M
// 64 with BM=128 keeps 4 waves per block (2x2), so TILE_N stays 256 and the
// loader strides do not change; each wave then owns 4 MFMA row tiles (nIterM).
#define MATMUL_MLP2_WARP_TILE_M 64
#endif
#ifndef MATMUL_MLP2_WARP_TILE_N
#define MATMUL_MLP2_WARP_TILE_N 32
#endif
#ifndef MATMUL_MLP2_WAVES_PER_BLOCK
// Derived from the tile: the kernel body uses TILE_N = 64 * (BM/WM)*(BN/WN) threads.
// Overriding with -D is only allowed if it still equals this expression (there is a static_assert).
#define MATMUL_MLP2_WAVES_PER_BLOCK (((MATMUL_MLP2_BLOCK_ROWS) / (MATMUL_MLP2_WARP_TILE_M)) * ((MATMUL_MLP2_BLOCK_COLS) / (MATMUL_MLP2_WARP_TILE_N)))
#endif
#ifndef NUM_CTA_MLP2
// SECOND argument of __launch_bounds__ = minimum WAVES PER SIMD, not blocks
// per CU. 2 leaves 512/2 = 256 VGPRs per lane for the 128x192 tile, whose
// 3x4x2 f32x4 accumulators alone take 96. 3 (170 VGPRs) measured the same
// at 128 steps; 2 is what the 8 GPU / 1024 step number was measured with.
#define NUM_CTA_MLP2 2
#endif
// Maximum number of split-K slices for mlp2. 1 = never split. At BM=128, 4 of 360 launches
// (the layers with few pairs) fall below the target_cta threshold and would split; the
// z_partial accumulation is then an atomicAdd, so the addition order follows the block
// schedule, and that rounding difference amplifies through bf16 until it changes routing at
// layer 23. Measured throughput was the same either way, so not splitting is the choice:
// the output is bit-exact against the BM=96 build, and deterministic.
#ifndef MATMUL_MLP2_MAX_SPLITS
#define MATMUL_MLP2_MAX_SPLITS 1
#endif


// The router GEMM as one kernel over the bf16 copy of its input; see getp_matmul_router_fused.
// Bit-exact. Set 0 for the split-K kernels on the fp32 input.
#ifndef GETP_ROUTER_FUSED
#define GETP_ROUTER_FUSED 1
#endif
// The fused router's loads run this many 16-k steps ahead of its MFMAs.
#ifndef GETP_ROUTER_G
#define GETP_ROUTER_G 8
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
// Derived from the tile: the kernel body uses TILE_N = 64 * (BM/WM)*(BN/WN) threads.
// Overriding with -D is only allowed if it still equals this expression (there is a static_assert).
#define MATMUL_ROUTER_WAVES_PER_BLOCK (((MATMUL_ROUTER_BLOCK_ROWS) / (MATMUL_ROUTER_WARP_TILE_M)) * ((MATMUL_ROUTER_BLOCK_COLS) / (MATMUL_ROUTER_WARP_TILE_N)))
#endif
#ifndef MATMUL_ROUTER_THREADS
#define MATMUL_ROUTER_THREADS (64 * MATMUL_ROUTER_WAVES_PER_BLOCK)
#endif

// ===========================================================================
// mlp1 LDS layout: KPack = 4  ->  tile[k/4][row][4]
//
// One slot of 4 shorts = 8 bytes is exactly the 4 CONSECUTIVE k elements a lane needs for
// v_mfma_f32_16x16x16bf16_1k (A[i][k]: lane L register r -> i = L%16,
// k = 4*(L/16)+r; B[k][j] likewise). So:
//   read : 1 x ds_read_b64   instead of 4 x ds_read_u16
//   write: 2 x ds_write_b64 instead of 8 x ds_write_b16 (one uint4 from HBM)
//
// Bank (32 bank x 4 byte = 128 byte/chu ky):
//   READ   the 16 lanes of one phase share k0 and take consecutive rows:
//          byte = C + 8*(L%16) -> 128 contiguous bytes = covering the 32 banks exactly
//          once. True for EVERY pad value.
//   WRITE  the 16 lanes of one phase have a = lane%NKC (k chunk, the loader's fast axis)
//          and b = lane/NKC (row): dword offset = 4*S*a + 2*b, where S is the number of slots
//          per k0 plane. The pad has to make the 16 lanes cover the 16 even banks - checked below.
//        S = BM + 2 = 130: 4*130 = 520 == 8 (mod 32) -> bank = 8a + 2b with
//          a in [0,4), b in [0,4) -> {0,2,...,30}, each bank used exactly once.
//        S = BM + 0 = 128: 4*128 = 512 == 0 (mod 32) -> bank = 2b -> a 4-way conflict.
// ===========================================================================
constexpr bool mlp1_lds_write_conflict_free(int S, int NKC) {
  if (NKC <= 0 || NKC > 16 || 16 % NKC != 0) return false;
  int seen = 0;
  for (int l = 0; l < 16; ++l) {
    const int a = l % NKC;
    const int b = l / NKC;
    int bank = (4 * S * a + 2 * b) % 32;
    if (bank < 0) bank += 32;
    if (bank & 1) return false;            // a slot must be 8-byte aligned
    const int bit = 1 << (bank >> 1);
    if (seen & bit) return false;          // two lanes on the same bank
    seen |= bit;
  }
  return seen == 0xFFFF;                   // covers all 16 even banks
}

// One LDS plan SHARED by the kernel body and the launcher: a single copy of the padding
// arithmetic, so the two cannot disagree (the old silent failure mode).
template <int BM_, int BNt_, int BK_, int PADS_>
struct Mlp1LdsPlan {
  static constexpr int KP   = 4;                 // short / o = 8 byte = 1 ds_read_b64
  static constexpr int KSL  = BK_ / KP;          // so plane k0
  static constexpr int PADS = PADS_;             // pad counted in SLOTS
  static constexpr int SM   = BM_  + PADS;       // slots per plane, tile A
  static constexpr int SN   = BNt_ + PADS;       // slots per plane, tile W
  static constexpr size_t bytes =
      (size_t)KSL * ((size_t)SM + 2 * (size_t)SN) * KP * sizeof(unsigned short);

  static_assert(BK_ % KP == 0, "BK must be a multiple of KPack = 4");
  static_assert(BK_ % 8 == 0,  "the loader loads a uint4 = 8 shorts along k");
  static_assert(mlp1_lds_write_conflict_free(SM, BK_ / 8),
                "MATMUL_MLP1_LDS_PAD_SLOTS is wrong: ds_write_b64 bank conflict on tile A");
  static_assert(mlp1_lds_write_conflict_free(SN, BK_ / 8),
                "MATMUL_MLP1_LDS_PAD_SLOTS is wrong: ds_write_b64 bank conflict on tile W");
  static_assert(((size_t)KSL * SM * KP * sizeof(unsigned short)) % 16 == 0,
                "the gate tile base must be 16-byte aligned");
  static_assert(((size_t)KSL * SN * KP * sizeof(unsigned short)) % 16 == 0,
                "the up tile base must be 16-byte aligned");
};

// Pack two dwords into one slot of 4 x bf16. A pure retype: both VGPRs are already in place,
// so it emits no VALU instruction at all (unlike the old version's 4 half-word extracts).
__device__ __forceinline__ bf16x4 mlp1_pack2(unsigned int lo, unsigned int hi) {
  struct Pair { unsigned int a, b; };
  union { Pair p; bf16x4 h; } cvt;
  cvt.p.a = lo;
  cvt.p.b = hi;
  return cvt.h;
}

// GETP_MLP1_AHEAD2 (default 1): mlp1's HBM loads go two k tiles ahead, branch-free, between the
// MFMAs. The one-ahead loop issued tile t+1 at the top of step t and needed it at the end of the
// same step, from basic blocks of its own (`if (r < tcnt && kk + 7 < H)`, zero fills), so all six
// loads went out in one burst; removing the loads altogether made mlp1 10-13 % faster while serving
// the weights from L2 saved only 1-6 %, i.e. the loads cost latency, not bandwidth. Now step t loads
// tile t+2 into register set t&1 while it writes tile t+1 from set (t+1)&1 to LDS, and a row past
// tcnt / a column past I / a tile past the end read a clamped valid address instead of branching: an
// MFMA output row depends only on its own A row (a column only on its own B column), and the epilogue
// never stores those. Op test (one rank, us, 120B typical / 120B hot / 20B): 1250 / 2296 / 1969 ->
// 1199 / 2234 / 1889. Bit-exact. Needs H % (2 * BK) == 0; 0 restores the one-ahead loop.
#ifndef GETP_MLP1_AHEAD2
#define GETP_MLP1_AHEAD2 1
#endif
// MFMAs between two consecutive HBM loads of the two-ahead loop (6 loads over 64 MFMAs; 7..10 are
// equal within noise, 4 is 1 % slower).
#ifndef GETP_MLP1_LOAD_GAP
#define GETP_MLP1_LOAD_GAP 10
#endif
template <int V> struct Mlp1Parity { static constexpr int value = V; };
// GATHER: when enabled, A_in is no longer a pre-copied [pair][H] table but ext_t_bf16
// [token][H] itself; the tile's row comes from tok_idx[pos]. See the note where `arow` is
// computed. GATHER = false generates exactly the old code (tok_idx is optimised away).
template<
  int BM, int BN, int BK,
  int WM, int WN,
  int BN_AGG, int CTA, bool GATHER,
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
  int H, int I, int E, float swiglu_limit,
  const int* __restrict__ tok_idx
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

  // LDS layout [k/4][row][4] - see Mlp1LdsPlan.
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
  // Launch check: the kernel body assumes exactly TILE_N threads.
  if (blockDim.x != TILE_N) {
    if (threadIdx.x == 0)
      printf("BAD LAUNCH: blockDim=%d but TILE_N=%d\n", (int)blockDim.x, (int)TILE_N);
    return;
  }
  // Fill LDS with bf16 qNaN: any slot the loader fails to write shows up as NaN.
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

  // --- Rows of tile A ------------------------------------------------------
  // The old code: a separate kernel copied a_in[pos][:] = ext_t_bf16[tok_idx[pos]][:]
  // for all 6144 pairs (20b, EP=2) and then mlp1 read it back. That was 35,4 MB read + 35,4 MB
  // written per layer just to reorder rows - and a_in had no consumer other than mlp1. Now mlp1
  // gathers for itself: A_in points straight at ext_t_bf16 and the row is tok_idx[pos]. One
  // more gain: the source table is 17,7 MB instead of 35,4 MB (each token appears ONCE rather
  // than once per local expert of its own), so the 23 column blocks share L2 better.
  // This index does not depend on k, so it is computed ONCE here (nbLoadsX = 2 at BM = 128,
  // strideX8 = 64) rather than inside the k loop: it costs +2 VGPRs but
  // live across the whole k loop - see the note on occupancy.
  // `base` is NOT dead on the GATHER path: it is still the base of tok_idx[base + rr] and
  // still the output row of the epilogue (`pos = base + row` writes gate_up_bf16 by PAIR
  // index, not by token) - which is why the epilogue needs no change at all.
  // Rows r >= tcnt are never loaded (all three load sites are guarded by `r < tcnt`), so 0 is
  // used there simply to have a valid value: it neither reads past the end of e_dev (allocated
  // for exactly max_pairs elements) nor adds a branch.
  // The array is declared with a full nbLoadsX elements on BOTH paths: with GATHER = false the
  // the load does not run and `arow[i]` sits in the dead arm of a ternary - but if the array
  // has only 1 element, that dead arm is a constant index OUT OF BOUNDS (i = 1), which is
  // enough for the compiler to have to reason about it.
  int arow[nbLoadsX];
  #pragma unroll
  for (int i = 0; i < (GATHER ? nbLoadsX : 0); ++i) {
    const int rr = loadX_y8 + i * strideX8;
    arow[i] = (rr < tcnt) ? tok_idx[base + rr] : 0;
  }

  // Fast/Slow Path Split optimization: Separate boundary-safe and boundary-check loads
  constexpr int safe_iters = (BM >= strideX8) ? (BM - strideX8) / strideX8 : 0;

  // Fast path: No boundary checks (all safe iterations)
  #pragma unroll
  for (int i = 0; i < safe_iters; ++i) {
    const int off = i * strideX8;
    const int kk = 0 + LOAD_VEC * loadX_x8;
    const int r = loadX_y8 + off;
    const int pos = GATHER ? arow[i] : (base + r);

    // 8 consecutive shorts along k of one row = 2 adjacent slots along the k0 plane axis.
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
    const int pos = GATHER ? arow[i] : (base + r);

    // 8 consecutive shorts along k of one row = 2 adjacent slots along the k0 plane axis.
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

    static_assert(LOAD_VEC == 2 * KP, "one uint4 from HBM must cover exactly 2 KPack slots");
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

#if GETP_MLP1_AHEAD2
  // HBM loads two k tiles ahead (see GETP_MLP1_AHEAD2): step t loads tile t+2 into set t&1 and
  // writes tile t+1 from set (t+1)&1 to LDS. Unrolled by two so each set has a fixed register name.
  uint4 sX[2][nbLoadsX], sWG[2][nbLoadsW], sWU[2][nbLoadsW];
  // Branch-free: clamped row, column and k tile; `kt` is the k offset of the tile.
  auto load_set = [&](uint4 (&rX)[nbLoadsX], uint4 (&rG)[nbLoadsW], uint4 (&rU)[nbLoadsW], int kt) {
    const int kc = min(kt, H - BK);
    #pragma unroll
    for (int i = 0; i < nbLoadsX; ++i) {
      const int r = min(loadX_y8 + i * strideX8, tcnt - 1);
      const int pos = GATHER ? arow[i] : (base + r);
      rX[i] = *reinterpret_cast<const uint4*>(A_in + (size_t)pos * H + kc + 8 * loadX_x8);
    }
    #pragma unroll
    for (int i = 0; i < nbLoadsW; ++i) {
      const int col = min(colBase + loadW_y + i * strideW, I - 1);
      rG[i] = *reinterpret_cast<const uint4*>(W1 + w1_e + (size_t)(2 * col + 0) * H + kc + 8 * loadW_x);
      rU[i] = *reinterpret_cast<const uint4*>(W1 + w1_e + (size_t)(2 * col + 1) * H + kc + 8 * loadW_x);
    }
  };
  load_set(sX[1], sWG[1], sWU[1], BK);
  auto kstep = [&](auto S_, int bk) {
    constexpr int S = decltype(S_)::value;
    load_set(sX[S], sWG[S], sWU[S], bk + 2 * BK);
    // MFMA ILP optimization: Pre-load all vectors to improve instruction-level parallelism
    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
      // basek = k + 4*yMF is always a multiple of 4 -> plane k0 = basek/4 = (k>>2) + yMF.
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

    // Pinned schedule (mask 0x0100 = LDS read, 0x0008 = MFMA, 0x0020 = VMEM read): the LDS reads
    // first, since the MFMAs wait on them, then one HBM load every GETP_MLP1_LOAD_GAP MFMAs. The
    // loads are for two steps later, so nothing waits on them here. (Splitting the LDS reads - the
    // second 16-deep half's reads spread over the first half's MFMAs - measured 1-3 % slower.)
    __builtin_amdgcn_sched_group_barrier(0x0100, 20, 0);
    #pragma unroll
    for (int i = 0; i < nbLoadsX + 2 * nbLoadsW; ++i) {
      __builtin_amdgcn_sched_group_barrier(0x0008, GETP_MLP1_LOAD_GAP, 0);
      __builtin_amdgcn_sched_group_barrier(0x0020, 1, 0);
    }
    __builtin_amdgcn_sched_group_barrier(0x0008, 64 - (nbLoadsX + 2 * nbLoadsW) * GETP_MLP1_LOAD_GAP, 0);
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();
    {
      const uint4 (&regX)[nbLoadsX] = sX[1 - S];
      const uint4 (&regWG)[nbLoadsW] = sWG[1 - S];
      const uint4 (&regWU)[nbLoadsW] = sWU[1 - S];
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
  };
  #pragma unroll 1
  for (int bk = 0; bk < H; bk += 2 * BK) {     // H % (2 * BK) == 0: the launcher checks
    kstep(Mlp1Parity<0>{}, bk);
    kstep(Mlp1Parity<1>{}, bk + BK);
  }
#else
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
        const int pos = GATHER ? arow[i] : (base + r);

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
      // basek = k + 4*yMF is always a multiple of 4 -> plane k0 = basek/4 = (k>>2) + yMF.
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

    // Pin the schedule for mlp1's compute region, including the HBM loads: mask 0x0020 = VMEM read,
    // 0x0100 = LDS read, 0x0008 = MFMA. Issue every load first, then the LDS reads, then the MFMA.
    // Earlier pinning rounds only pinned MFMA and LDS reads, so the compiler still scattered the 12
    // loads through the MFMA chain, each one dragging an s_waitcnt vmcnt with it. Measured on 8 GPUs
    // over 1024 steps, two interleaved pairs: 64.505 / 64.512 against 64.299 / 64.248 (+0,4 %), bit-exact.
    // The 'one LDS read every 4 MFMA' pattern was tried and was SLOWER (mlp1 1.797 against 1.688 us/call).
    __builtin_amdgcn_sched_group_barrier(0x0020, 12, 0);
    __builtin_amdgcn_sched_group_barrier(0x0100, 20, 0);
    __builtin_amdgcn_sched_group_barrier(0x0008, 64, 0);
    __builtin_amdgcn_sched_barrier(0);
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
#endif

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
// mlp2 LDS layout: KPack = 4  ->  tile[k/4][row][4]   (was: tile[k][row])
//
// One slot of 4 shorts = 8 bytes is exactly the 4 CONSECUTIVE k elements a lane needs for
// v_mfma_f32_16x16x16bf16_1k (A[i][k]: lane L register r -> i = L%16,
// k = 4*(L/16)+r; B[k][j] likewise). So:
//   read : 1 x ds_read_b64   instead of 4 x ds_read_u16
//   write: 2 x ds_write_b64 instead of 8 x ds_write_b16 (one uint4 from HBM)
//
// Bank (32 banks x 4 bytes = 128 bytes/cycle), numbers for the default mlp2 configuration
// BM=64, BN=64, BN_AGG=3 -> BNt=192, BK=32, WM=WN=32, 256 thread:
//
//   READ   one ds_read_b64 phase = 16 lanes with the same yMF (hence the SAME k0 plane), xMF=0..15
//          run over 16 consecutive ROWS:  byte(L) = C + 8*L  ->  128 contiguous bytes,
//          i.e. 32 banks, each used once  => CLEAN FOR EVERY PAD.
//        (The old version: 16 lanes read 16 consecutive shorts = 8 dwords -> 8 banks; two lanes
//          sharing one dword is a broadcast and so 'conflict-free', but each cycle moves only
//          64 useful bytes = half the bandwidth, and costs 4x the instructions.)
//
//   WRITE  the loader keeps the thread -> (row, k chunk) mapping UNCHANGED:
//          a = tid % NKC (a chunk of 8 shorts along k), b = tid / NKC (row), NKC=BK/8.
//          One uint4 = 8 shorts along k = 2 ADJACENT slots along the k0 plane axis:
//          dword(a,b) = 4*S*a + 2*b   (S = the number of slots per k0 plane)
//        S = BM  + 2 =  66 : 4*66  = 264 == 8 (mod 32) -> bank = 8a + 2b with
//                           a in [0,4), b in [0,4) -> {0,2,...,30}, each bank
//                           used exactly once  => CLEAN WRITE.
//        S = BNt + 2 = 194 : 4*194 = 776 == 8 (mod 32) -> same as above => CLEAN.
//        The old version (PAD=8): chunk stride 8*BM_PAD/2 = 8*72/2 = 288 dwords == 0 (mod
//          32), and 8*BNt_PAD/2 = 800 == 0 (mod 32) -> ALL FOUR chunks land on the same
//          bank group: ds_write_b16 had a 4-WAY conflict on both tiles.
//
// Why not the flat [row][k] layout (KPAD = BK + 4 shorts): there the read is clean only when
// (BK+PAD)/4 is odd, while the WRITE stays two-dimensional for EVERY pad; this layout writes
// CLEANLY for real. It also uses less LDS: 16640 B against 18432 B for [row][k], and
// 17408 B for the layout it replaced.
// ===========================================================================
constexpr bool mlp2_lds_write_conflict_free(int S, int NKC) {
  if (NKC <= 0 || NKC > 16 || 16 % NKC != 0) return false;
  int seen = 0;
  for (int l = 0; l < 16; ++l) {         // one ds_write_b64 phase = 16 lanes
    const int a = l % NKC;               // k chunk (the loader's fast axis)
    const int b = l / NKC;               // row
    int bank = (4 * S * a + 2 * b) % 32;
    if (bank < 0) bank += 32;
    if (bank & 1) return false;          // an aligned 8-byte slot -> the bank must be even
    const int bit = 1 << (bank >> 1);
    if (seen & bit) return false;        // two lanes on the same bank
    seen |= bit;
  }
  return seen == 0xFFFF;                 // covers all 16 even banks
}

// The smallest valid pad for BOTH tiles, derived from (BM, BNt, BK) rather than a constant
// hard-wired: BK=32 -> 2 (2, 6, 10 all work); BK=64 -> 1 (1,3,5,7); BK=16 -> 4.
// This keeps every existing mlp2 sweep line (bm128 / bn128 / 128x128 / bk64) compiling without
// having to re-derive the padding by hand.
constexpr int mlp2_lds_pad_slots_auto(int BM_, int BNt_, int NKC) {
  for (int p = 0; p <= 16; ++p)
    if (mlp2_lds_write_conflict_free(BM_ + p, NKC) &&
        mlp2_lds_write_conflict_free(BNt_ + p, NKC))
      return p;
  return -1;
}

// One LDS plan SHARED by the kernel body and the launcher: a single copy of the padding
// arithmetic, so the two cannot disagree (the old silent failure mode: the launcher allocates
// too little dynamic LDS and the tile is corrupted with nobody reporting it).
template <int BM_, int BNt_, int BK_, int PADS_REQ_>
struct Mlp2LdsPlan {
  static constexpr int KP   = 4;                 // short / o = 8 byte = 1 ds_read_b64
  static constexpr int KSL  = BK_ / KP;          // so plane k0
  static constexpr int NKC  = BK_ / 8;           // number of 8-short chunks in one row
  static constexpr int PADS = (PADS_REQ_ >= 0)
                                ? PADS_REQ_
                                : mlp2_lds_pad_slots_auto(BM_, BNt_, NKC);
  static constexpr int SM   = BM_  + PADS;       // slots per plane, tile A
  static constexpr int SN   = BNt_ + PADS;       // slots per plane, tile W
  static constexpr size_t bytes =
      (size_t)KSL * ((size_t)SM + (size_t)SN) * KP * sizeof(unsigned short);

  static_assert(BK_ % KP == 0, "BK must be a multiple of KPack = 4");
  static_assert(BK_ % 8 == 0,  "the loader loads a uint4 = 8 shorts along k");
  static_assert(PADS >= 0,
                "no bank-clean pad exists for this (BM,BNt,BK) - see mlp2_lds_pad_slots_auto");
  static_assert(mlp2_lds_write_conflict_free(SM, NKC),
                "MATMUL_MLP2_LDS_PAD_SLOTS is wrong: ds_write_b64 bank conflict on tile A"
                " (BK=32 takes 2, BK=64 takes 1, BK=16 takes 4)");
  static_assert(mlp2_lds_write_conflict_free(SN, NKC),
                "MATMUL_MLP2_LDS_PAD_SLOTS is wrong: ds_write_b64 bank conflict on tile W"
                " (BK=32 takes 2, BK=64 takes 1, BK=16 takes 4)");
  static_assert(((size_t)KSL * SM * KP * sizeof(unsigned short)) % 16 == 0,
                "the W tile base must be 16-byte aligned");
  static_assert(bytes <= 64u * 1024u, "the mlp2 LDS tile exceeds 64 KB/CU");
};

// GETP_MLP2_AHEAD2 (default 1): mlp2's HBM loads two k tiles ahead, branch-free, between the MFMAs
// - the same change as GETP_MLP1_AHEAD2 (clamped row tcnt-1, column H-1, last k tile; those outputs
// are never stored). Op test (one rank, us, 120B typical / 120B hot / 20B): 662 / 1298 / 1158 ->
// 625 / 1204 / 1056. Bit-exact. Needs the unsplit kernel (MATMUL_MLP2_MAX_SPLITS = 1) and
// I % (2 * BK) == 0; 0 restores the one-ahead loop.
#ifndef GETP_MLP2_AHEAD2
#define GETP_MLP2_AHEAD2 1
#endif
// MFMAs between two consecutive HBM loads (5 loads in the 30 MFMAs after the LDS reads; 2-6 equal).
#ifndef GETP_MLP2_LOAD_GAP
#define GETP_MLP2_LOAD_GAP 4
#endif
template <int V> struct Mlp2Parity { static constexpr int value = V; };

// Pack two dwords into one slot of 4 x bf16. A pure retype: both VGPRs are already in place,
// so it emits no VALU instruction (unlike the old version's 4 half-word extracts).
// __builtin_bit_cast is the standard spelling, not punning through a union.
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

  // LDS layout [k/4][row][4] - see Mlp2LdsPlan. The geometry is computed ONCE
  // here, and the launcher reuses that same plan.
  using LdsPlan = Mlp2LdsPlan<BM, BNt, BK, MATMUL_MLP2_LDS_PAD_SLOTS>;
  constexpr int KP  = LdsPlan::KP;    // 4 short / o
  constexpr int KSL = LdsPlan::KSL;   // so plane k0 = BK/4
  constexpr int SM  = LdsPlan::SM;    // o / plane, tile A
  constexpr int SN  = LdsPlan::SN;    // o / plane, tile W

  // __align__(16): the base of the dynamic LDS region is GUARANTEED 16-byte aligned, so the
  // cast to bf16x4 (8-byte aligned) is correct by type, not correct by accident.
  // The name mlp2_sm is local so its align attribute cannot collide with another kernel's `sm`.
  extern __shared__ __align__(16) unsigned short mlp2_sm[];
  bf16x4* const sxs = reinterpret_cast<bf16x4*>(mlp2_sm);
  bf16x4* const sws = sxs + KSL * SM;

#ifdef DEBUG_LDS_POISON
  // Launch check: the kernel body assumes exactly TILE_N threads.
  if (blockDim.x != TILE_N) {
    if (threadIdx.x == 0)
      printf("BAD LAUNCH: blockDim=%d but TILE_N=%d\n", (int)blockDim.x, (int)TILE_N);
    return;
  }
  // Fill LDS with bf16 qNaN: any slot the loader fails to write shows up as NaN.
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

    // 8 consecutive shorts along k of ONE row = 2 adjacent slots along the k0 plane axis.
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

    // 8 consecutive shorts along k of ONE row = 2 adjacent slots along the k0 plane axis.
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

    static_assert(LOAD_VEC == 2 * KP, "one uint4 from HBM must cover exactly 2 KPack slots");
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

#if GETP_MLP2_AHEAD2
  // Branch-free HBM loads two k tiles ahead (see GETP_MLP2_AHEAD2): step t loads tile t+2 into set
  // t&1 and writes tile t+1 from set (t+1)&1 to LDS. Unrolled by two so each set has a fixed name.
  uint4 sX[2][nbLoadsX], sW[2][nbLoadsW];
  auto load_set = [&](uint4 (&rX)[nbLoadsX], uint4 (&rW)[nbLoadsW], int kt) {
    const int kc = min(kt, kEnd - BK);
    #pragma unroll
    for (int i = 0; i < nbLoadsX; ++i) {
      const int r = min(loadX_y8 + i * strideX8, tcnt - 1);
      rX[i] = *reinterpret_cast<const uint4*>(gate_up_bf16 + (size_t)(base + r) * (size_t)I + kc + LOAD_VEC * loadX_x8);
    }
    #pragma unroll
    for (int i = 0; i < nbLoadsW; ++i) {
      const int col = min(colBase + loadW_y + i * strideW, H - 1);
      rW[i] = *reinterpret_cast<const uint4*>(W2 + w2_e + (size_t)col * (size_t)I + kc + 8 * loadW_x);
    }
  };
  load_set(sX[1], sW[1], kBeg + BK);
  auto kstep = [&](auto S_, int bk) {
    constexpr int S = decltype(S_)::value;
    load_set(sX[S], sW[S], bk + 2 * BK);
    // ---- Compute: read substep k+16's LDS fragments BEFORE substep k's MFMA chain ----
    // See the note of the same name in mlp1: the ds_read latency -> s_waitcnt -> MFMA runs
    // back to back for most of the kernel's time; two fragment buffers in registers let the
    // next substep's ds_read issue ahead of the previous substep's MFMA. Bit-exact.
    bf16x4 fa[2][nIterM];
    bf16x4 fb[2][BN_AGG][nIterN];
    auto load_frags = [&](int k, int buf) {
      const int kslot = (k >> 2) + yMF;
      #pragma unroll
      for (int im = 0; im < nIterM; ++im)
        fa[buf][im] = sxs[kslot * SM + yTile * WM + im * 16 + xMF];          // ds_read_b64
      #pragma unroll
      for (int g = 0; g < BN_AGG; ++g) {
        #pragma unroll
        for (int in = 0; in < nIterN; ++in)
          fb[buf][g][in] = sws[kslot * SN + xTile * WN + xMF + g * BN + in * 16]; // ds_read_b64
      }
    };
    load_frags(0, 0);
    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
      const int cur = (k / 16) & 1;
      if (k + 16 < BK) load_frags(k + 16, cur ^ 1);
      #pragma unroll
      for (int im = 0; im < nIterM; ++im) {
        #pragma unroll
        for (int g = 0; g < BN_AGG; ++g) {
          #pragma unroll
          for (int in = 0; in < nIterN; ++in)
            d_acc[g][im][in] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(fa[cur][im], fb[cur][g][in], d_acc[g][im][in], 0, 0, 0);
        }
      }
    }
    // The one-ahead loop's pinned order (substep 0's reads, then one read of substep 1 per 3
    // MFMAs), with the HBM loads spread over the trailing MFMAs.
    __builtin_amdgcn_sched_group_barrier(0x0100, 6, 0);
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
      __builtin_amdgcn_sched_group_barrier(0x0008, 3, 0);
      __builtin_amdgcn_sched_group_barrier(0x0100, 1, 0);
    }
    #pragma unroll
    for (int i = 0; i < nbLoadsX + nbLoadsW; ++i) {
      __builtin_amdgcn_sched_group_barrier(0x0008, GETP_MLP2_LOAD_GAP, 0);
      __builtin_amdgcn_sched_group_barrier(0x0020, 1, 0);
    }
    __builtin_amdgcn_sched_group_barrier(0x0008, 30 - (nbLoadsX + nbLoadsW) * GETP_MLP2_LOAD_GAP, 0);
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();
    {
      const uint4 (&regX)[nbLoadsX] = sX[1 - S];
      const uint4 (&regW)[nbLoadsW] = sW[1 - S];
      // Store activation with padding
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX8;
        const int r = loadX_y8 + off;
        const int pos = base + r;

        // 8 consecutive shorts along k of ONE row = 2 adjacent slots along the k0 plane axis.
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

        static_assert(LOAD_VEC == 2 * KP, "one uint4 from HBM must cover exactly 2 KPack slots");
        bf16x4* dst = &sws[((8 * loadW_x) / KP) * SN + (loadW_y + off)];

        const uint4 wb = regW[i];

        dst[0]  = mlp2_pack2(wb.x, wb.y);   // k+0 .. k+3   ds_write_b64
        dst[SN] = mlp2_pack2(wb.z, wb.w);   // k+4 .. k+7   ds_write_b64
      }
    }
    __syncthreads();
  };
  #pragma unroll 1
  for (int bk = kBeg; bk < kEnd; bk += 2 * BK) {   // (kEnd - kBeg) % (2 * BK) == 0: the launcher checks
    kstep(Mlp2Parity<0>{}, bk);
    kstep(Mlp2Parity<1>{}, bk + BK);
  }
#else
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

    // ---- Compute: read substep k+16's LDS fragments BEFORE substep k's MFMA chain ----
    // See the note of the same name in mlp1: the ds_read latency -> s_waitcnt -> MFMA runs
    // back to back for most of the kernel's time; two fragment buffers in registers let the
    // next substep's ds_read issue ahead of the previous substep's MFMA. Bit-exact.
    bf16x4 fa[2][nIterM];
    bf16x4 fb[2][BN_AGG][nIterN];
    auto load_frags = [&](int k, int buf) {
      const int kslot = (k >> 2) + yMF;
      #pragma unroll
      for (int im = 0; im < nIterM; ++im)
        fa[buf][im] = sxs[kslot * SM + yTile * WM + im * 16 + xMF];          // ds_read_b64
      #pragma unroll
      for (int g = 0; g < BN_AGG; ++g) {
        #pragma unroll
        for (int in = 0; in < nIterN; ++in)
          fb[buf][g][in] = sws[kslot * SN + xTile * WN + xMF + g * BN + in * 16]; // ds_read_b64
      }
    };
    load_frags(0, 0);
    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
      const int cur = (k / 16) & 1;
      if (k + 16 < BK) load_frags(k + 16, cur ^ 1);
      #pragma unroll
      for (int im = 0; im < nIterM; ++im) {
        #pragma unroll
        for (int g = 0; g < BN_AGG; ++g) {
          #pragma unroll
          for (int in = 0; in < nIterN; ++in)
            d_acc[g][im][in] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(fa[cur][im], fb[cur][g][in], d_acc[g][im][in], 0, 0, 0);
        }
      }
    }
    // Force a schedule on this region: read substep 0's 6 fragments first, then one ds_read of
    // substep 1 every 3 MFMA, with the remaining MFMA at the end. Without it the compiler pulls
    // the ds_read right up against its use, so every following MFMA waits a full LDS latency
    // (ISA: W M W M ...). mlp2 measured 1.272 -> 1.172 us/call (-7,9 %), bit-exact. The trailing fence keeps MFMA from sinking past the barrier.
    __builtin_amdgcn_sched_group_barrier(0x0100, 6, 0);
    __builtin_amdgcn_sched_group_barrier(0x0008, 3, 0);
    __builtin_amdgcn_sched_group_barrier(0x0100, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x0008, 3, 0);
    __builtin_amdgcn_sched_group_barrier(0x0100, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x0008, 3, 0);
    __builtin_amdgcn_sched_group_barrier(0x0100, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x0008, 3, 0);
    __builtin_amdgcn_sched_group_barrier(0x0100, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x0008, 3, 0);
    __builtin_amdgcn_sched_group_barrier(0x0100, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x0008, 3, 0);
    __builtin_amdgcn_sched_group_barrier(0x0100, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x0008, 30, 0);
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();

    if (bk + BK < kEnd) {
      // Store activation with padding
      #pragma unroll
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX8;
        const int r = loadX_y8 + off;
        const int pos = base + r;

        // 8 consecutive shorts along k of ONE row = 2 adjacent slots along the k0 plane axis.
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

        static_assert(LOAD_VEC == 2 * KP, "one uint4 from HBM must cover exactly 2 KPack slots");
        bf16x4* dst = &sws[((8 * loadW_x) / KP) * SN + (loadW_y + off)];

        const uint4 wb = regW[i];

        dst[0]  = mlp2_pack2(wb.x, wb.y);   // k+0 .. k+3   ds_write_b64
        dst[SN] = mlp2_pack2(wb.z, wb.w);   // k+4 .. k+7   ds_write_b64
      }
    }

    __syncthreads();
  }
#endif

  // ---- Epilogue -------------------------------------------------------------
  // The old version loaded the bias and w_by_bucket INSIDE the store loop, after the per-row
  // `continue` branches: the ISA came out as 208 global_load_dword + 192 global_load_ushort +
  // 784 serial s_waitcnt, plus 192 CAS loops of the float atomicAdd for the split-K branch
  // - 8.102 instructions running once per block, longer than the main loop itself. Here each
  // lane loads its 6 biases and 16 row weights ONCE, and what is left is arithmetic and stores.
  // The arithmetic order is unchanged ((acc + bias) * w), so the result is bit-exact.
  const int xD = lane & 15;
  const int yD = 4 * (lane >> 4);
  const int colBaseLane = blockIdx.x * (BN * BN_AGG) + xTile * WN + xD;

  float bia[BN_AGG][nIterN];
  #pragma unroll
  for (int g = 0; g < BN_AGG; ++g) {
    #pragma unroll
    for (int in = 0; in < nIterN; ++in) {
      const int col = colBaseLane + in * 16 + g * BN;
      bia[g][in] = (B2 && col < H) ? __bfloat162float(B2[b2_e + col]) : 0.f;
    }
  }
  float wrow[nIterM][4];
  #pragma unroll
  for (int im = 0; im < nIterM; ++im) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = yTile * WM + im * 16 + (yD + i);
      wrow[im][i] = (row < tcnt) ? w_by_bucket[base + row] : 0.f;
    }
  }

#if MATMUL_MLP2_MAX_SPLITS > 1
  if (splits > 1) {
    // Split-K: accumulate atomically; the bias is added only on slice z = 0.
    #pragma unroll
    for (int im = 0; im < nIterM; ++im) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = yTile * WM + im * 16 + (yD + i);
        if (row >= tcnt) continue;
        float* output_row = z_partial + (size_t)(base + row) * (size_t)H;
        const float w = wrow[im][i];
        #pragma unroll
        for (int g = 0; g < BN_AGG; ++g) {
          #pragma unroll
          for (int in = 0; in < nIterN; ++in) {
            const int col = colBaseLane + in * 16 + g * BN;
            if (col >= H) continue;
            const float b = (blockIdx.z == 0) ? bia[g][in] : 0.f;
            atomicAdd(output_row + col, (d_acc[g][im][in][i] + b) * w);
          }
        }
      }
    }
    return;
  }
#else
  (void)splits;
#endif

  #pragma unroll
  for (int im = 0; im < nIterM; ++im) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = yTile * WM + im * 16 + (yD + i);
      if (row >= tcnt) continue;
      float* output_row = z_partial + (size_t)(base + row) * (size_t)H;
      const float w = wrow[im][i];
      #pragma unroll
      for (int g = 0; g < BN_AGG; ++g) {
        #pragma unroll
        for (int in = 0; in < nIterN; ++in) {
          const int col = colBaseLane + in * 16 + g * BN;
          if (col >= H) continue;
          output_row[col] = (d_acc[g][im][in][i] + bia[g][in]) * w;
        }
      }
    }
  }
}


static inline void launch_mlp1_swiglu_bf16_bucketed_outbf16(
  __hip_bfloat16* gate_up_bf16, const __hip_bfloat16* a_in,
  const __hip_bfloat16* w1_layer, const __hip_bfloat16* b1_layer,
  const int* offsets, const int* blk_offs, int H, int I, int E,
  int total_blocks, float swiglu_limit, hipStream_t stream,
  // tok_idx != nullptr: a_in is ext_t_bf16 [token][H], and the row comes from tok_idx.
  // nullptr (the default): a_in is the pre-copied [pair][H] table - the old path.
  const int* tok_idx = nullptr
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
                "WAVES_PER_BLOCK must equal (BM/WM)*(BN/WN): the kernel body allocates"
                " TILE_N = 64*(BM/WM)*(BN/WN) threads, the launcher 64*WARPS.");
  // BN_AGG is ONE constant for both BNt and the kernel's template argument: the 2 used to be
  // written in two places, and letting them disagree silently under-allocates LDS.
  constexpr int BN_AGG = 2;
  constexpr int BNt = BN * BN_AGG;
  using LdsPlan = Mlp1LdsPlan<BM, BNt, BK, MATMUL_MLP1_LDS_PAD_SLOTS>;
#if GETP_MLP1_AHEAD2
  if (H % (2 * BK) != 0) {
    fprintf(stderr, "mlp1: GETP_MLP1_AHEAD2 needs H %% (2 * BK) == 0 (H = %d)\n", H);
    abort();
  }
#endif
  dim3 block(THREADS);
  dim3 grid((I + BNt - 1) / BNt, total_blocks);
  // ONE source of truth for the LDS size - the same plan the kernel body uses.
  size_t shmem = LdsPlan::bytes;
#if GETP_SMALL_NOSCATTER
  if (tok_idx) {
    mlp1_swiglu_bf16_bucketed_kernel_outbf16_tuned<BM,BN,BK,WM,WN,BN_AGG,CTA,true>
      <<<grid, block, shmem, stream>>>(gate_up_bf16, a_in, w1_layer, b1_layer, offsets, blk_offs, H, I, E, swiglu_limit, tok_idx);
    return;
  }
#else
  (void)tok_idx;
#endif
  mlp1_swiglu_bf16_bucketed_kernel_outbf16_tuned<BM,BN,BK,WM,WN,BN_AGG,CTA,false>
    <<<grid, block, shmem, stream>>>(gate_up_bf16, a_in, w1_layer, b1_layer, offsets, blk_offs, H, I, E, swiglu_limit, nullptr);

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
                "WAVES_PER_BLOCK must equal (BM/WM)*(BN/WN): the kernel body allocates"
                " TILE_N = 64*(BM/WM)*(BN/WN) threads, the launcher 64*WARPS.");
  // BN_AGG is ONE constant for both BNt and the kernel's template argument:
  // the 3 used to be written in two places, and letting them disagree silently corrupts the tile.
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
  splits = min(splits, MATMUL_MLP2_MAX_SPLITS);
#if GETP_MLP2_AHEAD2
  // The two-ahead loop runs the whole k range in pairs of tiles: no split, I % (2 * BK) == 0.
  if (splits != 1 || I % (2 * BK) != 0) {
    fprintf(stderr, "mlp2: GETP_MLP2_AHEAD2 needs splits == 1 and I %% (2 * BK) == 0 (I = %d)\n", I);
    abort();
  }
#endif

  if (splits > 1) {
    int total_pairs = 0;
    // A SYNCHRONOUS hipMemcpy, not Async: total_pairs is used on the host on the very next line
    // to size the memset. The old code used Async and only worked because ROCm copies
    // synchronously when the host memory is pageable - relying on that is wrong.
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(&total_pairs, offsets + E, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemsetAsync(z_partial, 0, (size_t)total_pairs * (size_t)H * sizeof(float), stream));
  }

  // ONE source of truth for the LDS geometry: the kernel body's own plan.
  using LdsPlan = Mlp2LdsPlan<BM, BNt, BK, MATMUL_MLP2_LDS_PAD_SLOTS>;
  dim3 block(THREADS);
  dim3 grid(max(1, gx), max(1, gy), splits);
  // NUM_CTA_MLP2 is the minimum number of WAVES per SIMD (the second argument of
  // __launch_bounds__), NOT blocks per CU: 2 -> 256 VGPRs/lane for the 128x192 tile.
  // The LDS size comes from LdsPlan::bytes, so launcher and kernel cannot disagree.
  size_t shmem = LdsPlan::bytes;

  mlp2_partial_bf16_bucketed_splitk_kernel_inbf16_tuned<BM,BN,BK,WM,WN,BN_AGG,NUM_CTA_MLP2>
    <<<grid, block, shmem, stream>>>(z_partial, gate_up_bf16, w2_layer, b2_layer, w_by_bucket,
                                     offsets, blk_offs, tok_idx, I, H, E, splits);

}


// The [own_lo, own_hi) slice of e_agg is the part belonging to THIS DEVICE'S OWN TOKENS: no peer
// reads it (a peer pulls the slice carrying ITS OWN index out of our buffer). So instead of
// writing it out for vecadd to read back, add it straight into the residual.
// The branch is on blockIdx.y, so it is uniform across the block and does not diverge.
#if GETP_ROWSKIP
// One block, one slice. Scan B rows, mark the non-zero ones, prefix-scan to get the compacted
// position, then write BOTH directions of the mapping: rank[b] (row -> compacted position, -1
// if the row is zero) and idx[r] (the inverse).
//
// `in_slice` is the slice READ FROM, while `out_base`/`cnt_slot` say where to WRITE - the two
// must stay separate: the receiving side reads ITS OWN slice but writes the result into each
// PEER's bucket, and deriving the write position from the read slice makes seven peers collide.
template <bool FROM_NLOCAL>
__global__ __launch_bounds__(256) void getp_rowskip_build_kernel(
    const int *__restrict__ n_local,     // FROM_NLOCAL: [EP*B]
    const int *__restrict__ topk_i,      // !FROM_NLOCAL: [EP*B, K] global expert ids
    int K, int B, int in_slice, int e_lo, int e_hi,
    int *__restrict__ rank, int *__restrict__ idx, int *__restrict__ cnt,
    int out_base, int cnt_slot) {
  __shared__ int sacc;
  __shared__ int sc[256];
  if (threadIdx.x == 0) sacc = 0;
  __syncthreads();
  const int in_base = in_slice * B;
  for (int b0 = 0; b0 < B; b0 += 256) {
    const int b = b0 + threadIdx.x;
    int nz = 0;
    if (b < B) {
      if (FROM_NLOCAL) {
        nz = (n_local[in_base + b] > 0) ? 1 : 0;
      } else {
        for (int k = 0; k < K; ++k) {
          const int eg = topk_i[(size_t)(in_base + b) * K + k];
          if (eg >= e_lo && eg < e_hi) { nz = 1; break; }
        }
      }
    }
    sc[threadIdx.x] = nz;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
      const int v = (threadIdx.x >= off) ? sc[threadIdx.x - off] : 0;
      __syncthreads();
      sc[threadIdx.x] += v;
      __syncthreads();
    }
    const int excl = sc[threadIdx.x] - nz;        // EXCLUSIVE prefix within this pass
    const int tail = sc[255];
    if (b < B) {
      const int r = nz ? (sacc + excl) : -1;
      rank[out_base + b] = r;
      if (r >= 0) idx[out_base + r] = b;
    }
    __syncthreads();
    if (threadIdx.x == 0) sacc += tail;
    __syncthreads();
  }
  if (threadIdx.x == 0) cnt[cnt_slot] = sacc;
}

#if GETP_RS_MERGE
// Identical to getp_rowskip_build_kernel except that one BLOCK handles one slice, so all seven
// go in a single launch. `skip_slice` is this rank's own slice - that block returns immediately.
// blockIdx.x runs over destination slice (sender side) or peer index (receiver side).
template <bool FROM_NLOCAL>
__global__ __launch_bounds__(256) void getp_rowskip_build_all_kernel(
    const int *__restrict__ n_local, const int *__restrict__ topk_i,
    int K, int B, int in_slice_fixed, int skip_slice, int ep,
    const int *__restrict__ e_lo_arr, const int *__restrict__ e_hi_arr,
    int *__restrict__ rank, int *__restrict__ idx, int *__restrict__ cnt) {
  const int sl = blockIdx.x;
  if (sl >= ep || sl == skip_slice) return;
  // The sender reads slice sl; the receiver always reads its own slice.
  const int in_base = (FROM_NLOCAL ? sl : in_slice_fixed) * B;
  const int e_lo = FROM_NLOCAL ? 0 : e_lo_arr[sl];
  const int e_hi = FROM_NLOCAL ? 0 : e_hi_arr[sl];
  const int out_base = sl * B;
  __shared__ int sacc;
  __shared__ int sc[256];
  if (threadIdx.x == 0) sacc = 0;
  __syncthreads();
  for (int b0 = 0; b0 < B; b0 += 256) {
    const int b = b0 + threadIdx.x;
    int nz = 0;
    if (b < B) {
      if (FROM_NLOCAL) {
        nz = (n_local[in_base + b] > 0) ? 1 : 0;
      } else {
        for (int k = 0; k < K; ++k) {
          const int eg = topk_i[(size_t)(in_base + b) * K + k];
          if (eg >= e_lo && eg < e_hi) { nz = 1; break; }
        }
      }
    }
    sc[threadIdx.x] = nz;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
      const int v = (threadIdx.x >= off) ? sc[threadIdx.x - off] : 0;
      __syncthreads();
      sc[threadIdx.x] += v;
      __syncthreads();
    }
    const int excl = sc[threadIdx.x] - nz;
    const int tail = sc[255];
    if (b < B) {
      const int r = nz ? (sacc + excl) : -1;
      rank[out_base + b] = r;
      if (r >= 0) idx[out_base + r] = b;
    }
    __syncthreads();
    if (threadIdx.x == 0) sacc += tail;
    __syncthreads();
  }
  if (threadIdx.x == 0) cnt[sl] = sacc;
}
static inline void getp_rowskip_build_all_send(const int *n_local, int B, int ep, int skip_slice,
                                               int *rank, int *idx, int *cnt, hipStream_t st) {
  getp_rowskip_build_all_kernel<true><<<dim3((unsigned)ep), 256, 0, st>>>(
      n_local, nullptr, 0, B, 0, skip_slice, ep, nullptr, nullptr, rank, idx, cnt);
}
static inline void getp_rowskip_build_all_recv(const int *topk_i, int K, int B, int ep,
                                               int in_slice, int skip_slice,
                                               const int *e_lo, const int *e_hi,
                                               int *rank, int *idx, int *cnt, hipStream_t st) {
  getp_rowskip_build_all_kernel<false><<<dim3((unsigned)ep), 256, 0, st>>>(
      nullptr, topk_i, K, B, in_slice, skip_slice, ep, e_lo, e_hi, rank, idx, cnt);
}
#endif

#if GETP_AGSKIP
// Gather the rows of this rank's own slice that destination slice `sl` needs into one
// contiguous block. Grid (EP, B): block (sl, i) handles row i of the list for sl;
// i >= cnt[sl] returns immediately.
__global__ __launch_bounds__(256) void getp_agskip_gather_kernel(
    const __hip_bfloat16 *__restrict__ src, __hip_bfloat16 *__restrict__ dst,
    const int *__restrict__ idx, const int *__restrict__ cnt, int own, int B, int H) {
  const int sl = blockIdx.x;
  if (sl == own || (int)blockIdx.y >= cnt[sl]) return;
  const int b = idx[(size_t)sl * B + blockIdx.y];
  const __hip_bfloat16 *sp = src + (size_t)b * H;
  __hip_bfloat16 *dp = dst + ((size_t)sl * B + blockIdx.y) * H;
  const int H8 = H & ~7;
  for (int k = threadIdx.x * 8; k < H8; k += blockDim.x * 8)
    reinterpret_cast<uint4 *>(dp)[k >> 3] = reinterpret_cast<const uint4 *>(sp)[k >> 3];
  for (int k = H8 + threadIdx.x; k < H; k += blockDim.x) dp[k] = sp[k];
}
// Scatter the contiguous block just pulled back to its proper rows in the peer's bucket.
__global__ __launch_bounds__(256) void getp_agskip_scatter_kernel(
    __hip_bfloat16 *__restrict__ dst, const __hip_bfloat16 *__restrict__ src,
    const int *__restrict__ idx, int H) {
  const int b = idx[blockIdx.x];
  const __hip_bfloat16 *sp = src + (size_t)blockIdx.x * H;
  __hip_bfloat16 *dp = dst + (size_t)b * H;
  const int H8 = H & ~7;
  for (int k = threadIdx.x * 8; k < H8; k += blockDim.x * 8)
    reinterpret_cast<uint4 *>(dp)[k >> 3] = reinterpret_cast<const uint4 *>(sp)[k >> 3];
  for (int k = H8 + threadIdx.x; k < H; k += blockDim.x) dp[k] = sp[k];
}
#endif

static inline void getp_rowskip_build_send(const int *n_local, int B, int slice,
                                           int *rank, int *idx, int *cnt, hipStream_t st) {
  getp_rowskip_build_kernel<true><<<1, 256, 0, st>>>(
      n_local, nullptr, 0, B, slice, 0, 0, rank, idx, cnt, slice * B, slice);
}
static inline void getp_rowskip_build_recv(const int *topk_i, int K, int B, int in_slice,
                                           int e_lo, int e_hi, int *rank, int *idx, int *cnt,
                                           int out_slot, hipStream_t st) {
  getp_rowskip_build_kernel<false><<<1, 256, 0, st>>>(
      nullptr, topk_i, K, B, in_slice, e_lo, e_hi, rank, idx, cnt, out_slot * B, out_slot);
}

// dev_x[idx[r]*H + c] += src[r*H + c]. The same fp32 addition on the same values as the old
// vecadd; the skipped rows are all zero, so leaving them out changes nothing.
#if GETP_EAGG_BF16
// Identical to getp_rowskip_scatter_add_kernel but with a bf16 source. The addition into dev_x
// is still fp32.
__global__ __launch_bounds__(256) void getp_rowskip_scatter_add_bf16_kernel(
    float *__restrict__ x, const __hip_bfloat16 *__restrict__ src,
    const int *__restrict__ idx, const int *__restrict__ cnt, int H) {
  if ((int)blockIdx.y >= *cnt) return;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H) return;
  x[(size_t)idx[blockIdx.y] * H + c] += __bfloat162float(src[(size_t)blockIdx.y * H + c]);
}
#if GETP_SMALL_VEC
// Eight elements per thread (H % 8 == 0; the launcher checks): the same fp32 addition per element.
__global__ __launch_bounds__(128) void getp_rowskip_scatter_add_bf16_v8_kernel(
    float *__restrict__ x, const __hip_bfloat16 *__restrict__ src,
    const int *__restrict__ idx, const int *__restrict__ cnt, int H) {
  if ((int)blockIdx.y >= *cnt) return;
  const int c8 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c8 >= (H >> 3)) return;
  const uint4 v = reinterpret_cast<const uint4 *>(src + (size_t)blockIdx.y * H)[c8];
  float4 *xr = reinterpret_cast<float4 *>(x + (size_t)idx[blockIdx.y] * H) + 2 * c8;
  float4 a = xr[0], b = xr[1];
  a.x += bf16bits_to_f32((unsigned short)v.x); a.y += bf16bits_to_f32((unsigned short)(v.x >> 16));
  a.z += bf16bits_to_f32((unsigned short)v.y); a.w += bf16bits_to_f32((unsigned short)(v.y >> 16));
  b.x += bf16bits_to_f32((unsigned short)v.z); b.y += bf16bits_to_f32((unsigned short)(v.z >> 16));
  b.z += bf16bits_to_f32((unsigned short)v.w); b.w += bf16bits_to_f32((unsigned short)(v.w >> 16));
  xr[0] = a;
  xr[1] = b;
}
#endif
static inline void getp_rowskip_scatter_add_bf16(float *x, const __hip_bfloat16 *src,
                                                 const int *idx, const int *cnt, int H, int nrow,
                                                 hipStream_t st) {
  if (nrow <= 0) return;
#if GETP_SMALL_VEC
  if (H % 8 == 0) {
    dim3 blk8(128), grd8((unsigned)((H / 8 + 127) / 128), (unsigned)nrow);
    getp_rowskip_scatter_add_bf16_v8_kernel<<<grd8, blk8, 0, st>>>(x, src, idx, cnt, H);
    return;
  }
#endif
  dim3 blk(256), grd((unsigned)((H + 255) / 256), (unsigned)nrow);
  getp_rowskip_scatter_add_bf16_kernel<<<grd, blk, 0, st>>>(x, src, idx, cnt, H);
}
#endif

__global__ __launch_bounds__(256) void getp_rowskip_scatter_add_kernel(
    float *__restrict__ x, const float *__restrict__ src,
    const int *__restrict__ idx, const int *__restrict__ cnt, int H) {
  if ((int)blockIdx.y >= *cnt) return;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H) return;
  x[(size_t)idx[blockIdx.y] * H + c] += src[(size_t)blockIdx.y * H + c];
}
#if GETP_SMALL_VEC
__global__ __launch_bounds__(256) void getp_rowskip_scatter_add_v4_kernel(
    float *__restrict__ x, const float *__restrict__ src,
    const int *__restrict__ idx, const int *__restrict__ cnt, int H) {
  if ((int)blockIdx.y >= *cnt) return;
  const int c4 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c4 >= (H >> 2)) return;
  const float4 v = reinterpret_cast<const float4 *>(src + (size_t)blockIdx.y * H)[c4];
  float4 *xr = reinterpret_cast<float4 *>(x + (size_t)idx[blockIdx.y] * H) + c4;
  float4 a = *xr;
  a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
  *xr = a;
}
#endif
static inline void getp_rowskip_scatter_add(float *x, const float *src, const int *idx,
                                            const int *cnt, int H, int nrow, hipStream_t st) {
  if (nrow <= 0) return;
#if GETP_SMALL_VEC
  if (H % 4 == 0) {
    dim3 blk4(256), grd4((unsigned)((H / 4 + 255) / 256), (unsigned)nrow);
    getp_rowskip_scatter_add_v4_kernel<<<grd4, blk4, 0, st>>>(x, src, idx, cnt, H);
    return;
  }
#endif
  dim3 blk(256), grd((unsigned)((H + 255) / 256), (unsigned)nrow);
  getp_rowskip_scatter_add_kernel<<<grd, blk, 0, st>>>(x, src, idx, cnt, H);
}
#endif

__global__ void moe_gather_pairs_kernel(float *__restrict__ e_agg,
                                        float *__restrict__ x_resid,
                                        int own_lo, int own_hi,
                                        const float *__restrict__ z_partial,
                                        const int *__restrict__ pair_pos, int H,
                                        int K, int B,
                                        const int *__restrict__ rowskip_rank,
                                        int b_base, int store_bf16) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y + b_base;
  if (row >= H || b >= B) return;
  int base = b * K;
  float s = 0.f;
  for (int k = 0; k < K; ++k) {
    int pos = pair_pos[base + k];
    if (pos >= 0) s += z_partial[(size_t)pos * (size_t)H + row];
  }
  if (b >= own_lo && b < own_hi) {
    // Bit-exact: s is exactly the fp32 value e_agg would have held, and vecadd would have
    // added it exactly once in the same way.
    x_resid[(size_t)(b - own_lo) * (size_t)H + row] += s;
  } else {
    // OVERWRITE, do not accumulate: every cell is written exactly once (the B x H grid covers
    // it completely), so the 35,4 MB memset that used to precede this is no longer needed.
#if GETP_ROWSKIP
    if (rowskip_rank != nullptr) {
      // A zero row is neither written nor sent. Non-zero rows are packed to the FRONT of their
      // slice, so a peer only has to pull the first cnt rows.
      const int r = rowskip_rank[b];
      if (r < 0) return;
      const int Bslice = own_hi - own_lo;
      const int slice = b / Bslice;
      const size_t off = ((size_t)slice * Bslice + (size_t)r) * (size_t)H + (size_t)row;
#if GETP_EAGG_BF16
      // A branch predicted on a uniform value; negligible inside a 158 us kernel.
      if (store_bf16) {
        reinterpret_cast<__hip_bfloat16 *>(e_agg)[off] = __float2bfloat16(s);
        return;
      }
#endif
      e_agg[off] = s;
      return;
    }
#endif
    (void)rowskip_rank;
#if GETP_EAGG_BF16
    if (store_bf16) {
      reinterpret_cast<__hip_bfloat16 *>(e_agg)[(size_t)b * (size_t)H + row] =
          __float2bfloat16(s);
      return;
    }
#endif
    e_agg[(size_t)b * (size_t)H + row] = s;
  }
}


#if GETP_SMALL_VEC
// moe_gather_pairs_kernel with four consecutive columns per thread (H % 4 == 0; the launcher checks):
// the same K additions in the same order on each element, so bit-exact.
__global__ __launch_bounds__(256) void moe_gather_pairs_v4_kernel(float *__restrict__ e_agg,
                                        float *__restrict__ x_resid,
                                        int own_lo, int own_hi,
                                        const float *__restrict__ z_partial,
                                        const int *__restrict__ pair_pos, int H,
                                        int K, int B,
                                        const int *__restrict__ rowskip_rank,
                                        int b_base, int store_bf16) {
  const int c4 = blockIdx.x * blockDim.x + threadIdx.x;   // float4 index within the row
  const int b = blockIdx.y + b_base;
  const int H4 = H >> 2;
  if (c4 >= H4 || b >= B) return;
  const int base = b * K;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int k = 0; k < K; ++k) {
    const int pos = pair_pos[base + k];
    if (pos >= 0) {
      const float4 z = reinterpret_cast<const float4 *>(z_partial + (size_t)pos * (size_t)H)[c4];
      s.x += z.x; s.y += z.y; s.z += z.z; s.w += z.w;
    }
  }
  if (b >= own_lo && b < own_hi) {
    float4 *xr = reinterpret_cast<float4 *>(x_resid + (size_t)(b - own_lo) * (size_t)H) + c4;
    float4 x = *xr;
    x.x += s.x; x.y += s.y; x.z += s.z; x.w += s.w;
    *xr = x;
    return;
  }
  size_t row_off = (size_t)b * (size_t)H;
#if GETP_ROWSKIP
  if (rowskip_rank != nullptr) {
    const int r = rowskip_rank[b];
    if (r < 0) return;
    const int Bslice = own_hi - own_lo;
    const int slice = b / Bslice;
    row_off = ((size_t)slice * Bslice + (size_t)r) * (size_t)H;
  }
#endif
  (void)rowskip_rank;
#if GETP_EAGG_BF16
  if (store_bf16) {
    uint2 p;
    p.x = (unsigned)f32_to_bf16bits(s.x) | ((unsigned)f32_to_bf16bits(s.y) << 16);
    p.y = (unsigned)f32_to_bf16bits(s.z) | ((unsigned)f32_to_bf16bits(s.w) << 16);
    reinterpret_cast<uint2 *>(reinterpret_cast<unsigned short *>(e_agg) + row_off)[c4] = p;
    return;
  }
#endif
  (void)store_bf16;
  reinterpret_cast<float4 *>(e_agg + row_off)[c4] = s;
}
#endif
#if GETP_SMALL_VEC >= 2
// One wave per token row, lane l owning float4 columns l, l+64, ...: a row that is neither ours nor
// sent (rowskip rank < 0; 58 % of the 120B's rows) costs its wave one int load, and every lane keeps
// H4/64 loads in flight per expert row. Same additions, same order: bit-exact.
template <int H4>
__global__ __launch_bounds__(256) void moe_gather_pairs_w_kernel(float *__restrict__ e_agg,
                                        float *__restrict__ x_resid,
                                        int own_lo, int own_hi,
                                        const float *__restrict__ z_partial,
                                        const int *__restrict__ pair_pos,
                                        int K, int n,
                                        const int *__restrict__ rowskip_rank,
                                        int b_base, int store_bf16) {
  constexpr int H = 4 * H4, NI = (H4 + 63) / 64;
  const int lane = threadIdx.x & 63;
  const int bi = blockIdx.x * 4 + (threadIdx.x >> 6);
  if (bi >= n) return;
  const int b = bi + b_base;
  const bool own = b >= own_lo && b < own_hi;
  size_t row_off = (size_t)b * H;
#if GETP_ROWSKIP
  if (!own && rowskip_rank != nullptr) {
    const int r = rowskip_rank[b];
    if (r < 0) return;
    const int Bslice = own_hi - own_lo;
    row_off = ((size_t)(b / Bslice) * Bslice + (size_t)r) * H;
  }
#endif
  (void)rowskip_rank;
  float4 s[NI];
  #pragma unroll
  for (int i = 0; i < NI; ++i) s[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int k = 0; k < K; ++k) {
    const int pos = pair_pos[b * K + k];
    if (pos >= 0) {
      const float4 *zr = reinterpret_cast<const float4 *>(z_partial + (size_t)pos * H);
      #pragma unroll
      for (int i = 0; i < NI; ++i) {
        const int c4 = lane + 64 * i;
        if (c4 < H4) {
          const float4 z = zr[c4];
          s[i].x += z.x; s[i].y += z.y; s[i].z += z.z; s[i].w += z.w;
        }
      }
    }
  }
  if (own) {
    float4 *xr = reinterpret_cast<float4 *>(x_resid + (size_t)(b - own_lo) * H);
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
      const int c4 = lane + 64 * i;
      if (c4 < H4) {
        float4 x = xr[c4];
        x.x += s[i].x; x.y += s[i].y; x.z += s[i].z; x.w += s[i].w;
        xr[c4] = x;
      }
    }
    return;
  }
#if GETP_EAGG_BF16
  if (store_bf16) {
    uint2 *er = reinterpret_cast<uint2 *>(reinterpret_cast<unsigned short *>(e_agg) + row_off);
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
      const int c4 = lane + 64 * i;
      if (c4 < H4) {
        uint2 p;
        p.x = (unsigned)f32_to_bf16bits(s[i].x) | ((unsigned)f32_to_bf16bits(s[i].y) << 16);
        p.y = (unsigned)f32_to_bf16bits(s[i].z) | ((unsigned)f32_to_bf16bits(s[i].w) << 16);
        er[c4] = p;
      }
    }
    return;
  }
#endif
  (void)store_bf16;
  float4 *er = reinterpret_cast<float4 *>(e_agg + row_off);
  #pragma unroll
  for (int i = 0; i < NI; ++i) {
    const int c4 = lane + 64 * i;
    if (c4 < H4) er[c4] = s[i];
  }
}
#endif

static inline void moe_gather_pairs(float *e_agg, const float *z_partial,
                                    const int *pair_pos, int H, int K, int B,
                                    hipStream_t stream) {

  dim3 blk(256), grd((H + blk.x - 1) / blk.x, B);
  // own_lo == own_hi: no row goes into the residual, exactly as before.
  moe_gather_pairs_kernel<<<grd, blk, 0, stream>>>(e_agg, nullptr, 0, 0,
                                                   z_partial, pair_pos,
                                                   H, K, B, nullptr, 0, 0);

}

static inline void moe_gather_pairs_acc(float *e_agg, float *x_resid,
                                        int own_lo, int own_hi,
                                        const float *z_partial,
                                        const int *pair_pos, int H, int K,
                                        int B, hipStream_t stream,
                                        const int *rowskip_rank = nullptr,
                                        int b_base = 0, int b_count = -1,
                                        int store_bf16 = 0) {
  const int n = (b_count < 0) ? B : b_count;
  if (n <= 0) return;
#if GETP_SMALL_VEC >= 2
  if (H == 2880) {
    moe_gather_pairs_w_kernel<720><<<(unsigned)((n + 3) / 4), 256, 0, stream>>>(
        e_agg, x_resid, own_lo, own_hi, z_partial, pair_pos, K, n, rowskip_rank, b_base,
        store_bf16);
    return;
  }
#endif
#if GETP_SMALL_VEC
  if (H % 4 == 0) {
    dim3 blk4(256), grd4((H / 4 + blk4.x - 1) / blk4.x, (unsigned)n);
    moe_gather_pairs_v4_kernel<<<grd4, blk4, 0, stream>>>(e_agg, x_resid, own_lo, own_hi,
                                                          z_partial, pair_pos, H, K, B,
                                                          rowskip_rank, b_base, store_bf16);
    return;
  }
#endif
  dim3 blk(256), grd((H + blk.x - 1) / blk.x, (unsigned)n);
  moe_gather_pairs_kernel<<<grd, blk, 0, stream>>>(e_agg, x_resid,
                                                   own_lo, own_hi,
                                                   z_partial, pair_pos,
                                                   H, K, B, rowskip_rank,
                                                   b_base, store_bf16);
}

// ===========================================================================
// new_matmul_qkv_fused_kernel LDS layout: KPack = 4 -> tile[k/4][row][4]
//
// One SLOT = 4 shorts = 8 bytes = EXACTLY the 4 consecutive k elements a lane needs for
// v_mfma_f32_16x16x16bf16_1k (A[i][k]: lane L register r -> i = L%16,
// k = 4*(L/16)+r; B[k][j] likewise). So:
//   read : 1 x ds_read_b64   instead of 4 x ds_read_u16
//   write: 2 x ds_write_b64 instead of 8 x ds_write_b16 (one uint4 from HBM)
//
// Bank (32 bank x 4 byte = 128 byte/chu ky):
//   READ   the 16 lanes of one phase share the k0 plane (Ak = 4*(lane>>4) is constant) and
//        row = row0 + (lane&15) runs consecutively:
//          byte = C + 8*(lane&15) -> 128 contiguous bytes -> covering the 32 banks exactly
//          once. Conflict-free for EVERY pad value.
//   WRITE  loader: a = threadIdx.x % (BK/8) is the k chunk, b = threadIdx.x / (BK/8)
//          is the row. Within one 16-lane phase: a in [0,NKC), b in [0,16/NKC).
//          The first slot written has index = (2a)*S + b, where S is the number of slots per
//          -> byte = 16*S*a + 8*b  -> dword = 4*S*a + 2*b.
//        S = 128 + 2 = 130: 4*130 = 520 == 8 (mod 32) -> bank = 8a + 2b,
//          a in [0,4), b in [0,4) -> {0,2,...,30}: each bank used exactly once.
//          The second slot is a further 2*S = 260 == 4 (mod 32) away -> still a permutation.
//          => NO conflict on either the read or the write.
//        S = 128 + 0 = 128: 4*128 = 512 == 0 (mod 32) -> bank = 2b: 4-way.
//        S = 128 + 4 = 132: 4*132 = 528 == 16 (mod 32) -> {0,16,0,16}: 2-way.
//        (The 'flat' [row][k] layout also reads cleanly, but its write stays two-dimensional
//         for EVERY pad; this layout gives up nothing and is conflict-free both ways.)
//
// Alignment: 8-byte slots and a 16-byte aligned tile base -> every ds_read_b64 /
// ds_write_b64 needs 8-byte alignment. b128 is NOT usable here (the two slots of a uint4 are
// S slots apart, not adjacent) - a deliberate choice: b64 does not conflict.
// A conflict-free b64 beats a b128 that conflicts.
// ===========================================================================
constexpr bool getp_qkv_lds_write_conflict_free(int S, int NKC) {
  if (NKC <= 0 || NKC > 16 || 16 % NKC != 0) return false;
  int seen = 0;
  for (int l = 0; l < 16; ++l) {
    const int a = l % NKC;                 // k chunk (the loader's fast axis)
    const int b = l / NKC;                 // row within the 16-lane phase
    int bank = (4 * S * a + 2 * b) % 32;
    if (bank < 0) bank += 32;
    if (bank & 1) return false;            // a slot must be 8-byte aligned
    const int bit = 1 << (bank >> 1);
    if (seen & bit) return false;          // hai lane trung bank
    seen |= bit;
  }
  return seen == 0xFFFF;                   // covers all 16 even banks
}

// Derived padding: the smallest valid value, so it never wastes LDS.
// (row=128, NKC=4) -> 2 ; (row=128, NKC=2, i.e. BK=16) -> 4. That makes this layout
// general in BK instead of hard-wiring one constant.
constexpr int getp_qkv_pick_pad_slots(int rows, int NKC) {
  for (int p = 0; p < 16; ++p)
    if (getp_qkv_lds_write_conflict_free(rows + p, NKC)) return p;
  return -1;
}

// One LDS plan SHARED by the kernel body and the launcher.
template <int ROWS, int BK_, int PADS_>
struct GetpQkvLdsTile {
  static constexpr int KP   = 4;                  // short / o = 8 byte
  static constexpr int KSL  = BK_ / KP;           // so plane k0
  static constexpr int NKC  = BK_ / 8;            // number of loader k chunks
  static constexpr int PADS =
      (PADS_ >= 0) ? PADS_ : getp_qkv_pick_pad_slots(ROWS, NKC);
  static constexpr int S     = ROWS + PADS;       // slots per k0 plane
  static constexpr int slots = KSL * S;           // slots in the whole tile
  static constexpr size_t bytes =
      (size_t)slots * KP * sizeof(unsigned short);

  static_assert(BK_ % 8 == 0, "the loader loads a uint4 = 8 shorts along k");
  static_assert(BK_ % KP == 0, "BK must be a multiple of KPack = 4");
  static_assert(PADS >= 0,
                "no pad makes ds_write_b64 conflict-free: the loader has to change");
  static_assert(getp_qkv_lds_write_conflict_free(S, NKC),
                "GETP_QKV_LDS_PAD_SLOTS is wrong: ds_write_b64 bank conflict "
                "(ROWS=128: BK=32 -> takes 2, BK=16 -> takes 4)");
};

// QKV block geometry: ONE copy for both the kernel body and the launcher. The launcher used to
// keep its own copy and HAD drifted - 'const int BK = 16;' in the launcher while the kernel
// body used BK = 32. The variable was dead (only BM/BN feed the grid) so it was harmless, but
// it is exactly the class of bug where 'an old constant survives an edit'.
struct GetpQkvTile {
  static constexpr int BM = 128;
  static constexpr int BN = 128;
  static constexpr int BK = 32;
  static constexpr int WM = 64;
  static constexpr int WN = 64;
  static constexpr int THREADS = 256;
};

// Pack two dwords into one slot of 4 x bf16. A pure retype: both VGPRs are already in place,
// so it emits no VALU instruction (unlike the old version's 8 half-word extracts).
// __builtin_bit_cast is the standard spelling, not union punning.
__device__ __forceinline__ bf16x4 getp_qkv_pack2(unsigned int lo,
                                                 unsigned int hi) {
  struct Pair { unsigned int a, b; };
  static_assert(sizeof(Pair) == sizeof(bf16x4), "a slot must be 8 bytes");
  Pair p;
  p.a = lo;
  p.b = hi;
  return __builtin_bit_cast(bf16x4, p);
}

// The qkv GEMM's HBM loads can go two k tiles ahead, branch-free (the change GETP_MLP1_AHEAD2
// describes; template argument AHEAD2). Op test: M = 768 (the 120B, 240 blocks, one round) 275 ->
// 259 us, M = 1536 (the 20B, 480 blocks, 1.5 rounds) 455 -> 480 us, so the launcher takes it for
// M <= GETP_QKV_AHEAD2_MAXM only. Bit-exact either way.
#ifndef GETP_QKV_AHEAD2_MAXM
#define GETP_QKV_AHEAD2_MAXM 1024
#endif
// MFMAs between two consecutive HBM loads of the two-ahead loop.
#ifndef GETP_QKV_LOAD_GAP
#define GETP_QKV_LOAD_GAP 4
#endif
template <int V> struct QkvParity { static constexpr int value = V; };
template <bool AHEAD2>
__global__ void __launch_bounds__(256)
new_matmul_qkv_fused_kernel(
  float *__restrict__ q_out, float *__restrict__ k_out,
  float *__restrict__ v_out, const __hip_bfloat16 *__restrict__ x,
  const __hip_bfloat16 *__restrict__ w, const __hip_bfloat16 *__restrict__ b,
  int n, int q_len, int k_len, int v_len, int batch_size
#if GETP_ROPE_FUSED
  // The cos/sin table getp_compute_cos_sin builds ONCE per step (head_dim/2 elements, a few
  // hundred bytes - it sits in L1 after the first touch).
  // rope_cos == nullptr => fully off, the epilogue runs exactly the old path.
  , const float *rope_cos, const float *rope_sin, int head_dim
  // Only read under GETP_ROPE_FUSED_KV: this step's KV cache slot (the layer base plus
  // GETP_KV_OFF(slot, 0, 0, ...)), and the layer's ring capacity.
  , __hip_bfloat16 *__restrict__ k_slot_bf16
  , __hip_bfloat16 *__restrict__ v_slot_bf16
  , int kv_cap
  // GETP_KV8_K / _V: the scales of this slot (scale index GETP_KV_SIDX of the element offset).
  , float *__restrict__ k_slot_scale
  , float *__restrict__ v_slot_scale
  // 1/sqrt(head_dim); only read under GETP_Q_BF16. See qbf16.
  , float q_scale
#endif
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

  // The block geometry must match GetpQkvTile - the source of truth shared with the launcher.
  static_assert(BM == GetpQkvTile::BM && BN == GetpQkvTile::BN &&
                    BK == GetpQkvTile::BK && WM == GetpQkvTile::WM &&
                    WN == GetpQkvTile::WN &&
                    BLOCK_SIZE == GetpQkvTile::THREADS,
                "the kernel body and GetpQkvTile disagree");

  // LDS layout [k/4][row][4] - see GetpQkvLdsTile. The names xs/ws are REMOVED deliberately: any
  // site not yet converted becomes a compile error rather than a silently wrong address.
  using LdsX = GetpQkvLdsTile<BM, BK, GETP_QKV_LDS_PAD_SLOTS>;
  using LdsW = GetpQkvLdsTile<BN, BK, GETP_QKV_LDS_PAD_SLOTS>;
  constexpr int KP = LdsX::KP;    // 4 short / o
  constexpr int SX = LdsX::S;     // slots per k0 plane, tile A
  constexpr int SW = LdsW::S;     // slots per k0 plane, tile B
  static_assert(LdsX::KSL == LdsW::KSL, "both tiles must have the same number of k0 planes");
  static_assert(LdsX::bytes + LdsW::bytes <= 64 * 1024, "exceeds 64 KB of LDS per CU");

  // Afloat/Bfloat and bf16x4 are all 4 x unsigned short vectors (8 bytes), so they assign
  // directly; if the types ever drift apart this becomes a compile error.
  static_assert(sizeof(bf16x4) == sizeof(Afloat) &&
                    sizeof(bf16x4) == sizeof(Bfloat),
                "the MFMA A/B operands must be 8-byte vectors");

  // __align__(16) is insurance: bf16x4 only needs 8 bytes, but a 16-byte aligned tile base
  // makes EVERY slot address provably 8-byte aligned to LLVM, rather than merely
  // 'correct if you work it out'. That is the precondition for emitting ds_read_b64 / ds_write_b64.
  __shared__ __align__(16) bf16x4 xss[LdsX::slots];
  __shared__ __align__(16) bf16x4 wss[LdsW::slots];

#ifdef DEBUG_LDS_POISON
  // Fill LDS with bf16 qNaN: any slot the loader fails to write shows up as NaN. Useful for
  // bisecting when token agreement does not come out at exactly 100.00 %.
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
    // 8 consecutive shorts along k of ONE row = 2 adjacent slots along the k0 plane axis.
    // The half-word order is kept exactly (tmp.x low = k+0, tmp.x high = k+1, ...)
    // so this is a pure address change: bit-exact.
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

  if constexpr (AHEAD2) {
  // Branch-free HBM loads two k tiles ahead: step t loads tile t+2 into set t&1 and writes tile
  // t+1 from set (t+1)&1 to LDS. Unrolled by two so each set has a fixed register name.
  uint4 sX[2][nbLoadsX], sW[2][nbLoadsW];
  auto load_set = [&](uint4 (&rX)[nbLoadsX], uint4 (&rW)[nbLoadsW], int kt) {
    const int kc = min(kt, K - BK);
    #pragma unroll
    for (int i = 0; i < nbLoadsX; ++i) {
      const int index_y = min(BM * (int)blockIdx.y + loadXIdx_y + i * strideX, M - 1);
      rX[i] = *reinterpret_cast<const uint4 *>(&x[index_y * K + kc + 8 * loadXIdx_x]);
    }
    #pragma unroll
    for (int i = 0; i < nbLoadsW; ++i) {
      const int index_y = min(BN * (int)blockIdx.x + loadWIdx_y + i * strideW, N - 1);
      rW[i] = *reinterpret_cast<const uint4 *>(&w[index_y * K + kc + 8 * loadWIdx_x]);
    }
  };
  load_set(sX[1], sW[1], BK);
  auto kstep = [&](auto S_, int bkIdx) {
    constexpr int S = decltype(S_)::value;
    load_set(sX[S], sW[S], bkIdx + 2 * BK);
    // Computing
    #pragma unroll
    for (int k = 0; k < BK; k += MFMA_K) {
      #pragma unroll
      for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
        #pragma unroll
        for (int iterWaveN = 0; iterWaveN < nIterWaveN; ++iterWaveN) {
          // basek = k + 4*(lane>>4) is always a multiple of 4, so a row's 4 k elements fit
          // inside ONE slot -> 1 x ds_read_b64 instead of 4 x
          // ds_read_u16. Ak/Bk are kept so the MFMA lane mapping stays visible.
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
    // LDS reads first (the MFMAs wait on them), then one HBM load every GETP_QKV_LOAD_GAP MFMAs.
    constexpr int NMF = (BK / MFMA_K) * nIterWaveM * nIterWaveN;
    constexpr int NLD = nbLoadsX + nbLoadsW;
    static_assert(NLD * GETP_QKV_LOAD_GAP <= NMF, "GETP_QKV_LOAD_GAP too large for this tile");
    __builtin_amdgcn_sched_group_barrier(0x0100, (BK / MFMA_K) * (nIterWaveM + nIterWaveN), 0);
    #pragma unroll
    for (int i = 0; i < NLD; ++i) {
      __builtin_amdgcn_sched_group_barrier(0x0008, GETP_QKV_LOAD_GAP, 0);
      __builtin_amdgcn_sched_group_barrier(0x0020, 1, 0);
    }
    __builtin_amdgcn_sched_group_barrier(0x0008, NMF - NLD * GETP_QKV_LOAD_GAP, 0);
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();
    {
      const uint4 (&regX)[nbLoadsX] = sX[1 - S];
      const uint4 (&regW)[nbLoadsW] = sW[1 - S];
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
  };
  #pragma unroll 1
  for (int bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {   // K % (2 * BK) == 0: the launcher checks
    kstep(QkvParity<0>{}, bkIdx);
    kstep(QkvParity<1>{}, bkIdx + BK);
  }
  } else {
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
          // basek = k + 4*(lane>>4) is always a multiple of 4, so a row's 4 k elements fit
          // inside ONE slot -> 1 x ds_read_b64 instead of 4 x
          // ds_read_u16. Ak/Bk are kept so the MFMA lane mapping stays visible.
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
  }

#if GETP_ROPE_FUSED
  // ===================== RoPE folded into the epilogue ==========================
  // waveN0 is the N origin of the WHOLE wave: blockIdx.x*BN + WN*xInBlockTile -> a multiple
  // of 64. WN = 64 = head_dim, and the region boundaries (0, q_len, q_len+k_len) are also
  // multiples of 64 (the launcher re-checks before enabling this), so a wave lies ENTIRELY
  // inside ONE head of ONE region: the branches below are uniform across the wave, and the
  // partner at dim+half is always in the same lane. That is also why global_x < N no longer
  // needs checking: a wave is either wholly inside N or past the end - and then waveN0 >= N >=
  // q_len+k_len, so it falls through to the old path, which still bounds-checks fully.
  const int waveN0 = blockIdx.x * BN + WN * xInBlockTile;
  constexpr int nPairN = nIterWaveN / 2;
  static_assert(nIterWaveN % 2 == 0,
                "the low/high half of a head must be an even number of MFMA tiles");
  if (rope_cos != nullptr && waveN0 < q_len + k_len) {
    const int half = head_dim >> 1;            // 32
    const int CDj = (laneIdx % 16);
    // cos/sin depend only on (iterWaveN, CDj) - invariant across both loops
    // below - so load it once into a register instead of re-reading it 32 times.
    float cs[nPairN], sn[nPairN];
    #pragma unroll
    for (int p = 0; p < nPairN; ++p) {
      const int d = MFMA_N * p + CDj;          // 0..half-1
      cs[p] = rope_cos[d];
      sn[p] = rope_sin[d];
    }
    const bool in_q = (waveN0 < q_len);        // uniform across the wave
#if GETP_KV8_K
    if (!in_q) {
      // K: rotate as below, then int8 per (row, head). The row's 64 dims sit on the 16 lanes of
      // this lane group (CDj) x the lane's 2 pairs, so amax is the lane's 4 values then 4 shuffles.
      signed char *dst8 = reinterpret_cast<signed char *>(k_slot_bf16);
      const int head0 = waveN0 - q_len;          // the head's first K column
      #pragma unroll
      for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
        #pragma unroll
        for (int i = 0; i < GPRs_CD; ++i) {
          const int CDi = 4 * (laneIdx / 16) + (i % 4);
          const int global_y =
              blockIdx.y * BM + WM * yInBlockTile + iterWaveM * MFMA_M + CDi;
          float o[nPairN][2];
          float amax = 0.f;
          #pragma unroll
          for (int p = 0; p < nPairN; ++p) {
            const int gx1 = waveN0 + p * MFMA_N + CDj;
            const int gx2 = gx1 + half;
            float x1 = acc_local[iterWaveM * nIterWaveN + p][i];
            float x2 = acc_local[iterWaveM * nIterWaveN + p + nPairN][i];
            if (b) {
              x1 += __bfloat162float(b[gx1]);
              x2 += __bfloat162float(b[gx2]);
            }
            const float c = cs[p];
            const float sgn = sn[p];
            o[p][0] = x1 * c - x2 * sgn;
            o[p][1] = x2 * c + x1 * sgn;
            amax = fmaxf(amax, fmaxf(fabsf(o[p][0]), fabsf(o[p][1])));
          }
          amax = fmaxf(amax, __shfl_xor(amax, 1));
          amax = fmaxf(amax, __shfl_xor(amax, 2));
          amax = fmaxf(amax, __shfl_xor(amax, 4));
          amax = fmaxf(amax, __shfl_xor(amax, 8));
          if (global_y < M) {
            const float inv = amax > 0.f ? 127.f / amax : 0.f;
            #pragma unroll
            for (int p = 0; p < nPairN; ++p) {
              const int d1 = head0 + p * MFMA_N + CDj;
              dst8[GETP_KV_OFF(0, global_y, d1, M, k_len, kv_cap)] =
                  GETP_KV8_CODE(__float2int_rn(o[p][0] * inv));
              dst8[GETP_KV_OFF(0, global_y, d1 + half, M, k_len, kv_cap)] =
                  GETP_KV8_CODE(__float2int_rn(o[p][1] * inv));
            }
            if (CDj == 0)
              k_slot_scale[GETP_KV_SIDX(GETP_KV_OFF(0, global_y, head0, M, k_len, kv_cap))] =
                  amax * (1.f / 127.f);
          }
        }
      }
    } else
#endif
    #pragma unroll
    for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
      #pragma unroll
      for (int p = 0; p < nPairN; ++p) {
        // The partner at dim+half is N tile (p + nPairN): SAME lane, SAME i.
        const int accLo = iterWaveM * nIterWaveN + p;
        const int accHi = iterWaveM * nIterWaveN + p + nPairN;
        #pragma unroll
        for (int i = 0; i < GPRs_CD; ++i) {
          const int CDi = 4 * (laneIdx / 16) + (i % 4);
          const int global_y =
              blockIdx.y * BM + WM * yInBlockTile + iterWaveM * MFMA_M + CDi;
          const int gx1 = waveN0 + p * MFMA_N + CDj;   // low dim
          const int gx2 = gx1 + half;                  // its partner
          if (global_y < M) {
            float x1 = acc_local[accLo][i];
            float x2 = acc_local[accHi][i];
            if (b) {
              x1 += __bfloat162float(b[gx1]);
              x2 += __bfloat162float(b[gx2]);
            }
            const float c = cs[p];
            const float s = sn[p];
            // Copied VERBATIM from apply_rotary_emb_kernel: same operand order, same operation
            // order -> the compiler fuses exactly the same fma.
            // Changing one character on these two lines changes the last bit of the result.
            float o1 = x1 * c - x2 * s;
            float o2 = x2 * c + x1 * s;
            if (in_q) {
#if GETP_Q_BF16
              // q_scale is 2^-3, so multiplying before or after rounding gives the same bits.
              unsigned short *dst =
                  reinterpret_cast<unsigned short *>(q_out) + (size_t)global_y * q_len;
              dst[gx1] = f32_to_bf16bits(q_scale * o1);
              dst[gx2] = f32_to_bf16bits(q_scale * o2);
#else
              (void)q_scale;
              float *dst = q_out + (size_t)global_y * q_len;
              dst[gx1] = o1;
              dst[gx2] = o2;
#endif
            } else {
#if GETP_ROPE_FUSED_KV
              // kv_store_pair_fp32_to_bf16 has been removed: writing bf16 directly here uses
              // EXACTLY the old rounding helper, so the values match bit for bit.
              unsigned short *dst = reinterpret_cast<unsigned short *>(k_slot_bf16);
              dst[GETP_KV_OFF(0, global_y, gx1 - q_len, M, k_len, kv_cap)] = f32_to_bf16bits(o1);
              dst[GETP_KV_OFF(0, global_y, gx2 - q_len, M, k_len, kv_cap)] = f32_to_bf16bits(o2);
#else
              float *dst = k_out + (size_t)global_y * k_len;
              dst[gx1 - q_len] = o1;
              dst[gx2 - q_len] = o2;
#endif
            }
          }
        }
      }
    }
  }
#if GETP_ROPE_FUSED_KV
  else if (rope_cos != nullptr && waveN0 < N) {
    // The v region: no rotation, but since kv_store_pair is gone this has to convert to bf16
    // itself and write the cache slot directly. v_len == kv_dim, so the row stride is identical.
#if GETP_KV8_V
    // int8 per (row, head): the row's 64 dims are this lane group's 16 lanes x 4 N tiles.
    signed char *dst8 = reinterpret_cast<signed char *>(v_slot_bf16);
    const int head0 = waveN0 - q_len - k_len;
    #pragma unroll
    for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
      #pragma unroll
      for (int i = 0; i < GPRs_CD; ++i) {
        const int CDi = 4 * (laneIdx / 16) + (i % 4);
        const int CDj = (laneIdx % 16);
        const int global_y =
            blockIdx.y * BM + WM * yInBlockTile + iterWaveM * MFMA_M + CDi;
        float o[nIterWaveN];
        float amax = 0.f;
        #pragma unroll
        for (int iterWaveN = 0; iterWaveN < nIterWaveN; ++iterWaveN) {
          float r = acc_local[iterWaveM * nIterWaveN + iterWaveN][i];
          if (b) r += __bfloat162float(b[waveN0 + iterWaveN * MFMA_N + CDj]);
          o[iterWaveN] = r;
          amax = fmaxf(amax, fabsf(r));
        }
        amax = fmaxf(amax, __shfl_xor(amax, 1));
        amax = fmaxf(amax, __shfl_xor(amax, 2));
        amax = fmaxf(amax, __shfl_xor(amax, 4));
        amax = fmaxf(amax, __shfl_xor(amax, 8));
        if (global_y < M) {
          const float inv = amax > 0.f ? 127.f / amax : 0.f;
          #pragma unroll
          for (int iterWaveN = 0; iterWaveN < nIterWaveN; ++iterWaveN)
            dst8[GETP_KV_OFF(0, global_y, head0 + iterWaveN * MFMA_N + CDj, M, v_len, kv_cap)] =
                GETP_KV8_CODE(__float2int_rn(o[iterWaveN] * inv));
          if (CDj == 0)
            v_slot_scale[GETP_KV_SIDX(GETP_KV_OFF(0, global_y, head0, M, v_len, kv_cap))] =
                amax * (1.f / 127.f);
        }
      }
    }
#else
    #pragma unroll
    for (int iterWaveM = 0; iterWaveM < nIterWaveM; ++iterWaveM) {
      #pragma unroll
      for (int iterWaveN = 0; iterWaveN < nIterWaveN; ++iterWaveN) {
        const int acc_index = iterWaveM * nIterWaveN + iterWaveN;
        #pragma unroll
        for (int i = 0; i < GPRs_CD; ++i) {
          const int CDi = 4 * (laneIdx / 16) + (i % 4);
          const int CDj = (laneIdx % 16);
          const int global_x = waveN0 + iterWaveN * MFMA_N + CDj;
          const int global_y =
              blockIdx.y * BM + WM * yInBlockTile + iterWaveM * MFMA_M + CDi;
          float result = acc_local[acc_index][i];
          if (global_y < M) {
            if (b) result += __bfloat162float(b[global_x]);
            reinterpret_cast<unsigned short *>(v_slot_bf16)
                [GETP_KV_OFF(0, global_y, global_x - q_len - k_len, M, v_len, kv_cap)] =
                    f32_to_bf16bits(result);
          }
        }
      }
    }
#endif
  }
#endif
  else
#endif
  {
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
#if GETP_ROPE_FUSED
  // With RoPE alone (without _KV) these two pointers are unused - say so explicitly rather
  // than leaving a warning floating around.
  (void)k_slot_bf16;
  (void)v_slot_bf16;
  (void)kv_cap;
  (void)k_slot_scale;
  (void)v_slot_scale;
#endif
}

static inline void getp_matmul_qkv_fused_bf16(
    float *q, float *k, float *v,
    const __hip_bfloat16 *x_bf16,
    const __hip_bfloat16 *w_qkv_bf16,
    const __hip_bfloat16 *b_qkv_bf16,
    int n, int head_dim, int n_attn_heads, int n_kv_heads, int batch_size,
    hipStream_t stream
#if GETP_ROPE_FUSED
    // No default value: if any call site has not been updated, that must be a COMPILE ERROR
    // rather than a silent fall-through to the old path.
    , const float *rope_cos, const float *rope_sin,
    __hip_bfloat16 *k_slot, __hip_bfloat16 *v_slot, int kv_cap,
    // GETP_KV8_K / _V: the slot's scales (nullptr for a bf16 tensor)
    float *k_slot_scale, float *v_slot_scale
#endif
    ) {
#if GETP_KV_HEADMAJOR && !GETP_ROPE_FUSED_KV
#error "GETP_KV_HEADMAJOR needs GETP_ROPE_FUSED_KV: the unfused KV store writes the old layout"
#endif

  const int q_len = head_dim * n_attn_heads;
  const int k_len = head_dim * n_kv_heads;
  const int v_len = head_dim * n_kv_heads;

  const int M = batch_size;
  const int N = q_len + k_len + v_len;
  const int K = n;
  // Two-ahead loads for small batches only (see GETP_QKV_AHEAD2_MAXM); they need an even number of
  // k tiles.
  const bool ahead2 = M <= GETP_QKV_AHEAD2_MAXM && K % (2 * GetpQkvTile::BK) == 0;

  // ONE source of truth for the block geometry: taken from the constants the kernel body uses.
  // The old code here had 'const int BK = 16;' while the kernel body used BK = 32 -
  // a dead variable and so harmless, but a static_assert in the kernel body now pins
  // BM/BN/BK/WM/WN/THREADS to one another.
  const int BN = GetpQkvTile::BN;
  const int BM = GetpQkvTile::BM;

  dim3 block(GetpQkvTile::THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

#if GETP_ROPE_FUSED
  // The conditions that make 'one wave = one head' true. head_dim == WN puts a wave's N tile
  // exactly on one head; q_len/k_len divisible by head_dim puts each region's start on a
  // 64 boundary (so a wave never straddles a q|k|v boundary); N divisible by head_dim means a
  // wave is either wholly inside N or past the end.
  // If any of them fails, the check just below stops the run - see why there is no fallback.
  const bool rope_ok = (rope_cos != nullptr) && (rope_sin != nullptr) &&
                       (head_dim == GetpQkvTile::WN) && (head_dim % 2 == 0) &&
                       (q_len % head_dim == 0) && (k_len % head_dim == 0) &&
                       (N % head_dim == 0);
  // THERE IS NO FALLBACK. With GETP_ROPE_FUSED=1 the two getp_apply_rotary_emb calls have been
  // removed from BOTH loops (their only call sites), so 'rope_cos = nullptr -> take the old
  // path' no longer means 'rope still happens in a separate kernel' - it means NOBODY rotates
  // q/k any more. A shape that fails the conditions must therefore stop here outright, rather
  // than silently returning wrong numbers. (With the current configuration head_dim=64=WN,
  // q_len=4096, k_len=512, N=5120, rope_ok is always true; this check exists to stop a future
  // model with a different head_dim from slipping through unnoticed.)
  if (!rope_ok) {
    std::cerr << "GETP_ROPE_FUSED: the qkv shape does not meet the "
                 "one-wave-one-head condition (head_dim=" << head_dim
              << ", WN=" << GetpQkvTile::WN << ", q_len=" << q_len
              << ", k_len=" << k_len << ", N=" << N
              << ") - rope would not be applied anywhere." << std::endl;
    std::abort();
  }
#if GETP_ROPE_FUSED_KV
  // One more condition specific to the _KV step: kv_store_pair_fp32_to_bf16 is also gone, so
  // the KV slot may only be written if these two pointers really exist.
  if (k_slot == nullptr || v_slot == nullptr || (GETP_KV8_K && k_slot_scale == nullptr) ||
      (GETP_KV8_V && v_slot_scale == nullptr)) {
    std::cerr << "GETP_ROPE_FUSED_KV: missing KV slot pointer" << std::endl;
    std::abort();
  }
#endif
  if (ahead2)
    new_matmul_qkv_fused_kernel<true><<<grid, block, 0, stream>>>(
      q, k, v, x_bf16, w_qkv_bf16, b_qkv_bf16, n, q_len, k_len, v_len,
      batch_size, rope_ok ? rope_cos : nullptr, rope_sin, head_dim,
      k_slot, v_slot, kv_cap, k_slot_scale, v_slot_scale, 1.0f / sqrtf((float)head_dim));
  else
    new_matmul_qkv_fused_kernel<false><<<grid, block, 0, stream>>>(
      q, k, v, x_bf16, w_qkv_bf16, b_qkv_bf16, n, q_len, k_len, v_len,
      batch_size, rope_ok ? rope_cos : nullptr, rope_sin, head_dim,
      k_slot, v_slot, kv_cap, k_slot_scale, v_slot_scale, 1.0f / sqrtf((float)head_dim));
#else
  if (ahead2)
    new_matmul_qkv_fused_kernel<true><<<grid, block, 0, stream>>>(
      q, k, v, x_bf16, w_qkv_bf16, b_qkv_bf16, n, q_len, k_len, v_len, batch_size);
  else
    new_matmul_qkv_fused_kernel<false><<<grid, block, 0, stream>>>(
      q, k, v, x_bf16, w_qkv_bf16, b_qkv_bf16, n, q_len, k_len, v_len, batch_size);
#endif

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
    int *__restrict__ topk_i_out, const int *__restrict__ relabel,
    const int *__restrict__ mask_on) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int K = experts_per_token;
  const float eps = 1e-6f;

  router_score += (size_t)b * n_experts;
  topk_v_out += (size_t)b * K;
  topk_i_out += (size_t)b * K;
  // A finished sequence goes to no expert: id -1, weight 0. Every reader of the top-k table skips
  // ids outside its expert range, so the row costs the MoE nothing. b is uniform over the block.
  if (mask_on != nullptr && !mask_on[b]) {
    for (int i = tid; i < K; i += blockDim.x) { topk_v_out[i] = 0.f; topk_i_out[i] = -1; }
    return;
  }

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
      // The selection above is on expert ids; only the id WRITTEN is relabelled (GETP_EPLB).
      topk_i_out[i] = relabel ? relabel[topi[i]] : topi[i];
    }
  }
}

static inline void getp_router_topk_softmax_batch(
    const float *router_score, int n_experts, int experts_per_token,
    float *topk_v_out, int *topk_i_out, int batch_size, hipStream_t stream,
    const int *relabel = nullptr, const int *mask_on = nullptr) {

#if GETP_SMALL_TOPK
  // n_experts = 32 (20b) / 128 (120b), yet the block was still 1024 threads = 16 waves.
  // Two consequences, both of them dead time:
  //   * occupancy: 16 waves/block, capped at 8 waves/SIMD -> 2 blocks/CU -> 208 blocks
  //     resident out of 1536 -> 7,4 serial passes;
  //   * the reduction loop runs log2(1024) = 10 stages, EACH with a __syncthreads
  //     of 16 waves, repeated K = 4 times -> roughly 44 heavy barriers per block.
  // Measured at 2,5 ms/step for 250 KB of traffic (a floor of 0,2 us): almost all of it is
  // barriers and serialised execution, not memory.
  //
  // WHY IT IS BIT-EXACT to shrink the block to BLK = 2^ceil(log2(n_experts)) >= 64:
  //   (a) The per-thread phase: the loop `for (i = tid; i < n_experts; i += blockDim.x)`
  //       runs EXACTLY one pass for every BLK >= n_experts -> thread tid holds
  //       (scores[tid], tid) if tid < n_experts, otherwise it keeps (-INF, n_experts).
  //       Independent of BLK.
  //   (b) The reduction tree: the stage with stride s merges left (vl, il) with right (vr, ir).
  //       For EVERY stage with stride >= n_experts, the right-hand side is certainly the
  //       fall (vr = -INF, ir = n_experts), and the selection rule is
  //           take_r = (vr > vl + thr) || (|vr - vl| <= thr && ir < il),
  //           thr = eps * max(|vr|, |vl|) = eps * INF = INF.
  //       First branch: -INF > vl + INF is false for finite vl; for vl = -INF,
  //       vl + INF = NaN so the comparison is false too. Second branch: |vr - vl| <= INF
  //       holds, BUT ir = n_experts while il <= n_experts, so `ir < il` is always
  //       FALSE. => the left side is always kept. Every stage with stride >= n_experts is the
  //       identity, for ANY data (the argument assumes nothing about vl, so
  //       still holds when scores contain NaN or +INF).
  //   The 1024 version runs stages 512,256,...,1; the BLK version runs BLK/2,...,1. The part that
  //   differ only in stages >= n_experts, which (b) just proved to be the identity;
  //   the stages < n_experts match pairwise, same operands, same order, same rounding ->
  //   same bits.
  //   Preconditions: BLK is a power of 2 (the `s = blockDim.x >> 1` loop only covers it then)
  //   and BLK >= n_experts. For n_experts > 1024 it stays at 1024 = the old path.
  int BLK = 64;
  while (BLK < n_experts && BLK < 1024) BLK <<= 1;
  const dim3 block(BLK);
#else
  const int BLK = 1024;
  const dim3 block(BLK);
#endif
  const dim3 grid(batch_size);
  const size_t shmem = (size_t)n_experts * sizeof(float) +
                       (size_t)BLK * (sizeof(float) + sizeof(int)) +
                       GETP_ROUTER_TOPK_MAXK * (sizeof(float) + sizeof(int));
  router_topk_softmax_batch_kernel<<<grid, block, shmem, stream>>>(
      router_score, n_experts, experts_per_token, topk_v_out, topk_i_out, relabel, mask_on);

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
#if GETP_SMALL_VEC
// x[i] += y[i] over batch_size contiguous rows, four elements per thread.
__global__ __launch_bounds__(256) void vecadd_v4_kernel(float4 *__restrict__ x,
                                                        const float4 *__restrict__ y, size_t n4) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  float4 a = x[i];
  const float4 b = y[i];
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
  x[i] = a;
}
#endif
void getp_vecadd(float *x, float *y, int size, int batch_size,
                 hipStream_t stream) {
#if GETP_SMALL_VEC
  if (size % 4 == 0) {
    const size_t n4 = (size_t)size / 4 * (size_t)batch_size;
    vecadd_v4_kernel<<<(unsigned)((n4 + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<float4 *>(x), reinterpret_cast<const float4 *>(y), n4);
    return;
  }
#endif

  dim3 blockDim(1024);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x, batch_size);
  vecadd_kernel<<<gridDim, blockDim, 0, stream>>>(x, y, size);

}

// ============================================================================
// Decode attention using mfma_f32_16x16x16bf16_1k.
//
// The old version put the K axis of the product across the 16 BLOCKS of an mfma 4x4x4, so after
// each MFMA instruction it still had to merge 16 blocks with 4 __shfl_down rounds and then
// broadcast the score with 16 more __shfl: 128 cross-lane instructions and 80 __expf per 16 keys,
// in exchange for exactly 4 MFMA instructions. A flow test (replacing the whole compute section
// with a single add, keeping the HBM + LDS + barrier path intact) measured 56 % of the kernel's
// time in that compute section - so it was the expensive part, not the memory.
//
// This version puts KEY on the M axis and HEAD on the N axis:
//
//     S^T = K   . Q^T    A = K  [16 key x 16 dim]   B = Q^T [16 dim x 16 head]
//     O^T = V^T . P^T    A = V^T[16 dim x 16 key]   B = P^T [16 key x 16 head]
//
// The key point: the MFMA result layout D (lane holds D[4*(lane>>4)+m][lane&15]) MATCHES the
// B-operand layout exactly. So P^T flows straight from the first MFMA into the second - no
// transpose, not one cross-lane instruction. A whole 16-key tile is down to 2 __shfl_xor
// (merging the max across group g) and 5 exp instructions.
//
// The price: each wave holds only HPW real heads in the 16 columns of the N axis, so the column
// the rest is zero-filled. HPW = 4 keeps the old head split (2 waves/block, 25% of the columns
// are useful); HPW = 8 lets one wave cover all 8 heads of a kv_head - half the MFMA
// instructions, but also half the threads loading data. Selectable at compile time.
// ============================================================================
#ifndef FLASH_HEADS_PER_WAVE
#define FLASH_HEADS_PER_WAVE 8
#endif
#ifndef FLASH_LDS_PAD
#define FLASH_LDS_PAD 4
#endif
// Split P into two bf16 values (high part + remainder) and add them with TWO MFMA instructions, so
// so the PV product keeps close to the old fp32 accuracy. Meant for testing the hypothesis
// that 'rounding P is the cause of the drift'; keep it only if it really is the cause, since it
// costs 4 more MFMA instructions per tile.
#ifndef FLASH_P_HILO
#define FLASH_P_HILO 0
#endif

// Two KV tiles in flight per wave instead of one; see the loop. Bit-exact. On the time-major KV layout
// this gained nothing; on the padded head-major one (GETP_KV_HEADMAJOR) it is 2-4 % (a full 20B layer at
// 1024 keys, 2651 -> 2605 us), at 140 registers - three waves per SIMD instead of four.
#ifndef GETP_ATTN_PREFETCH2
#define GETP_ATTN_PREFETCH2 1
#endif
template <int HPW, bool APPLY_MASK>
__global__ void __launch_bounds__(64 * (8 / HPW))
flash_attn_decode_mfma16_kernel(
  __hip_bfloat16 *__restrict__ tb_bf16, float *__restrict__ tb_f32,
  const __hip_bfloat16 *__restrict__ key_cache,
  const __hip_bfloat16 *__restrict__ value_cache, const float *__restrict__ q,
  const int *__restrict__ mask_on,
  // GETP_KV8_K / _V: the layer's scales, one per 64-element vector (GETP_KV_SIDX)
  const float *__restrict__ k_scale, const float *__restrict__ v_scale,
  const float *__restrict__ attn_sinks, int head_dim, int n_attn_heads,
  int n_kv_heads, int pos, int batch_size, int kv_dim, int kv_mul,
  int sliding_window, int cache_tcap, float inv_sqrt_d
) {
  constexpr int HEAD_DIM   = 64;
  constexpr int TILE       = 16;                 // = M = N = K of the MFMA
  constexpr int NCHUNK     = HEAD_DIM / TILE;    // 4 chunks of 16 dims
  constexpr int KV_MUL     = 8;
  constexpr int WAVES      = KV_MUL / HPW;
  constexpr int BLOCK_SIZE = 64 * WAVES;
  constexpr int PAD        = FLASH_LDS_PAD;

  // PAD = 4 -> a row is 68 ushorts = 136 bytes = 34 banks; 34 mod 32 = 2, so the 16 lanes
  // a b64 read covers the 32 banks exactly once, with no conflict. In exchange a row needs only
  // 8 bytes, so the load path writes two b64 instructions instead of one b128.
  static_assert(HPW == 4 || HPW == 8, "HPW only supports 4 or 8");
  static_assert(PAD % 4 == 0, "PAD must be a multiple of 4 so a row still needs only 8-byte alignment");
  static_assert(TILE * (HEAD_DIM / 8) % BLOCK_SIZE == 0, "the KV load does not divide evenly");

  using bf16x4 = __attribute__((__vector_size__(8)))  unsigned short;
  using f32x4  = __attribute__((__vector_size__(16))) float;

  assert(head_dim == HEAD_DIM);
  assert(kv_mul == KV_MUL);
  assert(blockDim.x == BLOCK_SIZE);

  const int waveIdx = threadIdx.x / warpSize;
  const int lane    = threadIdx.x % warpSize;
  const int ln      = lane & 15;   // M axis of A, N axis of B and D
  const int g       = lane >> 4;   // K axis of A and B, row group of D

  const int kv_h = blockIdx.x;
  const int b    = blockIdx.y;
  if (!mask_on[b]) return;

  const int t_start = APPLY_MASK ? MAX(0, pos - sliding_window + 1)
                                 : MAX(0, pos + 1 - cache_tcap);
  const int n_steps = pos + 1 - t_start;
  const int t_end   = t_start + n_steps;

  // A single LDS region for all three. `ot` only lives in the epilogue, by which time ks/vs are
  // dead, so giving it its own allocation would hold 2 KB across the whole loop unused -
  // and LDS is what decides how many workgroups fit per CU.
  constexpr int KV_SHORTS = TILE * (HEAD_DIM + PAD);
  constexpr int OT_SHORTS = WAVES * HPW * HEAD_DIM * 2;   // float = 2 short
  static_assert(OT_SHORTS <= 2 * KV_SHORTS,
                "ot does not fit inside the ks+vs region - it needs its own allocation again");
  __shared__ __align__(16) unsigned short kvo[2 * KV_SHORTS];
  unsigned short (*ks)[HEAD_DIM + PAD] =
      reinterpret_cast<unsigned short (*)[HEAD_DIM + PAD]>(kvo);
  unsigned short (*vs)[HEAD_DIM + PAD] =
      reinterpret_cast<unsigned short (*)[HEAD_DIM + PAD]>(kvo + KV_SHORTS);
  // The same address as ks: may only be touched AFTER the loop.
  float (*ot)[HEAD_DIM] = reinterpret_cast<float (*)[HEAD_DIM]>(kvo);

  // ---- Q^T as the B operand: lane holds B[4g+kk][ln] = q[head ln][chunk*16+4g+kk].
  // Columns ln >= HPW correspond to no head, so they are filled with 0; their scores are 0
  // (finite, no NaN) and are never written out. The head index is clamped so the pointer never
  // leaves the q array.
  const int hg_self = kv_h * KV_MUL + waveIdx * HPW + (ln < HPW ? ln : 0);
  bf16x4 qB[NCHUNK];
  {
#if GETP_Q_BF16
    // The qkv epilogue already wrote bf16 ALREADY MULTIPLIED by inv_sqrt_d. The ternary must
    // STAY: without it the padding columns (ln >= HPW) get head 0's real q instead of zero.
    (void)inv_sqrt_d;
    const unsigned short *qrow =
        reinterpret_cast<const unsigned short *>(q) +
        (size_t)b * n_attn_heads * head_dim + (size_t)hg_self * head_dim;
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        qB[c][kk] = (ln < HPW) ? qrow[c * TILE + 4 * g + kk] : (unsigned short)0;
    }
#else
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
#endif
  }

  constexpr int nbLoadsKV = TILE * (HEAD_DIM / 8) / BLOCK_SIZE;
  constexpr int strideKV  = BLOCK_SIZE / (HEAD_DIM / 8);
  constexpr int nbLoadsKVc = nbLoadsKV, strideKVc = strideKV;
#if GETP_KV8_K
  // Q^T in fp16 for the fp16 S MFMA
  using f16x4 = _Float16 __attribute__((ext_vector_type(4)));
  f16x4 qH[NCHUNK];
  #pragma unroll
  for (int c = 0; c < NCHUNK; ++c) {
    #pragma unroll
    for (int kk = 0; kk < 4; ++kk)
      qH[c][kk] = (_Float16)__bfloat162float(bf16bits_to_bf16(qB[c][kk]));
  }
#endif
#if GETP_KV8_V
  using f16x4v = _Float16 __attribute__((ext_vector_type(4)));
  // P goes into the fp16 PV MFMA as e * 2^15: fp16-normal down to e ~ 2e-9; undone at the end.
  constexpr float PSCALE = 32768.f;
#endif
  const int ly = threadIdx.x / (HEAD_DIM / 8);
  const int lx = threadIdx.x % (HEAD_DIM / 8);

  // An LDS row only needs 8 bytes (136 = 8 x 17), so a b128 instruction would be misaligned;
  // two b64 instructions move the same 16 bytes and stay valid.
  auto put16 = [](unsigned short *dst, uint4 v) {
    *reinterpret_cast<uint2 *>(dst)     = make_uint2(v.x, v.y);
    *reinterpret_cast<uint2 *>(dst + 4) = make_uint2(v.z, v.w);
  };
  // ---- one tile (TILE keys) of K and V into registers, and from there into LDS. A bf16 tensor
  // takes nbLoadsKV 16-byte loads per lane (key ly + i * strideKV, dims 8 * lx); an int8 tensor
  // (GETP_KV8_*) takes one (key lane >> 2, dims 16 * (lane & 3)) plus its key's scale, and is
  // staged as fp16 q * scale.
#if GETP_KV8_ANY
  static_assert(BLOCK_SIZE == 64, "the int8 tile loader assumes one wave per block (HPW = 8)");
  static_assert(!FLASH_P_HILO, "FLASH_P_HILO is a bf16-P path");
  const int ly8 = lane >> 2, lx8 = lane & 3;
  using h2 = _Float16 __attribute__((ext_vector_type(2)));
  // codes u of bytes sh, sh+1 of w -> the fp16 pair (u - 128) * scale: v_perm_b32 takes byte 0/1
  // (or 2/3) of w for the low halves and 0x64 for the high ones (1024 + u, exact), then a packed
  // subtract of 1152 (exact) and a packed multiply by the scale.
  auto cvt2 = [](unsigned int w, int sh, h2 sc) {
    h2 x = __builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, w,
                                                        sh == 0 ? 0x04010400u : 0x04030402u));
    x = (x - (h2){(_Float16)1152.f, (_Float16)1152.f}) * sc;
    return __builtin_bit_cast(unsigned int, x);
  };
  auto put8 = [&](unsigned short *dst, uint4 w, float scale) {
    const h2 sc = {(_Float16)scale, (_Float16)scale};
    put16(dst, make_uint4(cvt2(w.x, 0, sc), cvt2(w.x, 2, sc), cvt2(w.y, 0, sc), cvt2(w.y, 2, sc)));
    put16(dst + 8,
          make_uint4(cvt2(w.z, 0, sc), cvt2(w.z, 2, sc), cvt2(w.w, 0, sc), cvt2(w.w, 2, sc)));
  };
#else
  (void)k_scale;
  (void)v_scale;
#endif
  struct KvTile {
    uint4 k[GETP_KV8_K ? 1 : nbLoadsKVc], v[GETP_KV8_V ? 1 : nbLoadsKVc];
    float sk, sv;
  };
  auto load_tile = [&](int tile0) {
    KvTile r;
    r.sk = r.sv = 0.f;
#if GETP_KV8_ANY
    {
      const int index_y = tile0 + ly8;
      const bool ok = index_y < t_end;
      const size_t off = ok ? GETP_KV_OFF(index_y % cache_tcap, b, kv_h * head_dim + 16 * lx8,
                                          batch_size, kv_dim, cache_tcap) : 0;
#if GETP_KV8_K
      r.k[0] = ok ? *reinterpret_cast<const uint4 *>(
                        reinterpret_cast<const signed char *>(key_cache) + off)
                  : make_uint4(0, 0, 0, 0);
      r.sk = ok ? k_scale[GETP_KV_SIDX(off - 16 * lx8)] : 0.f;
#endif
#if GETP_KV8_V
      r.v[0] = ok ? *reinterpret_cast<const uint4 *>(
                        reinterpret_cast<const signed char *>(value_cache) + off)
                  : make_uint4(0, 0, 0, 0);
      r.sv = ok ? v_scale[GETP_KV_SIDX(off - 16 * lx8)] : 0.f;
#endif
    }
#endif
#if !GETP_KV8_K || !GETP_KV8_V
    #pragma unroll
    for (int i = 0; i < nbLoadsKVc; ++i) {
      const int index_y = tile0 + ly + i * strideKVc;
      const bool ok = index_y < t_end;
      const size_t off = ok ? GETP_KV_OFF(index_y % cache_tcap, b, kv_h * head_dim + 8 * lx,
                                          batch_size, kv_dim, cache_tcap) : 0;
#if !GETP_KV8_K
      r.k[i] = ok ? *reinterpret_cast<const uint4 *>(key_cache + off) : make_uint4(0, 0, 0, 0);
#endif
#if !GETP_KV8_V
      r.v[i] = ok ? *reinterpret_cast<const uint4 *>(value_cache + off) : make_uint4(0, 0, 0, 0);
#endif
    }
#endif
    return r;
  };
  auto stage_tile = [&](const KvTile &r) {
#if GETP_KV8_K
    put8(&ks[ly8][16 * lx8], r.k[0], r.sk);
#else
    #pragma unroll
    for (int i = 0; i < nbLoadsKVc; ++i) put16(&ks[ly + i * strideKVc][lx * 8], r.k[i]);
#endif
#if GETP_KV8_V
    put8(&vs[ly8][16 * lx8], r.v[0], r.sv);
#else
    #pragma unroll
    for (int i = 0; i < nbLoadsKVc; ++i) put16(&vs[ly + i * strideKVc][lx * 8], r.v[i]);
#endif
  };
#if GETP_KV8_ANY && !GETP_ATTN_PREFETCH2
#error "GETP_KV8_* is written for the GETP_ATTN_PREFETCH2 loop"
#endif

  stage_tile(load_tile(t_start));
  __syncthreads();

  f32x4 acc_o[NCHUNK];
  #pragma unroll
  for (int c = 0; c < NCHUNK; ++c) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc_o[c][i] = 0.f;
  }
  float m_run = -INFINITY;
  float l_run = 0.f;

#if GETP_ATTN_PREFETCH2
  // Two tiles in flight: while tile bt is computed from LDS, tile bt+TILE waits in one register set
  // (loaded a trip earlier) and tile bt+2*TILE is loading into the other. The loop is unrolled by
  // hand, twice the old body with the sets swapped, so every array index stays a compile-time
  // constant and the sets stay in registers. The compute section is the old one, verbatim.
  KvTile rt0 = load_tile(t_start + TILE), rt1 = rt0;
  for (int bt0 = t_start; bt0 < t_end; bt0 += 2 * TILE) {
    {
      const int bt = bt0;   // tile bt in LDS, tile bt + TILE (if any) in set 0
      if (bt + 2 * TILE < t_end) rt1 = load_tile(bt + 2 * TILE);

    // ---- S^T = K . Q^T : A[key][dim] read with one b64 instruction per chunk
    f32x4 st;
    #pragma unroll
    for (int i = 0; i < 4; ++i) st[i] = 0.f;
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
#if GETP_KV8_K
      f16x4 kA;
      *reinterpret_cast<uint2 *>(&kA) =
          *reinterpret_cast<const uint2 *>(&ks[ln][c * TILE + 4 * g]);
      st = __builtin_amdgcn_mfma_f32_16x16x16f16(kA, qH[c], st, 0, 0, 0);
#else
      bf16x4 kA;
      *reinterpret_cast<uint2 *>(&kA) =
          *reinterpret_cast<const uint2 *>(&ks[ln][c * TILE + 4 * g]);
      st = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(kA, qB[c], st, 0, 0, 0);
#endif
    }

    // ---- the lower mask. Out-of-range ks rows are zero-filled and so score 0, and
    // exp(0 - m) > 0 would corrupt the sum -> they have to be forced to -INFINITY.
    const int kbase = bt + 4 * g;
    #pragma unroll
    for (int m = 0; m < 4; ++m)
      if (kbase + m >= t_end) st[m] = -INFINITY;

    // ---- the tile max, shared by all 4 g groups of the same head. Merging BEFORE computing
    // m_new is mandatory: one g group may hold no valid key at all (m_tile = -INF) and would
    // produce NaN if it computed its own. After the merge, every tile
    // is guaranteed to hold at least one valid key, so m_tile is finite.
    float m_tile = fmaxf(fmaxf(st[0], st[1]), fmaxf(st[2], st[3]));
    m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 16));
    m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 32));

    const float m_new = fmaxf(m_run, m_tile);
    const float alpha = __expf(m_run - m_new);

    // ---- P^T = exp(S^T - m_new), rounded to bf16 straight away to serve as the B operand.
#if GETP_KV8_V
    f16x4v pB;
#else
    bf16x4 pB;
#endif
#if FLASH_P_HILO
    bf16x4 pLo;
#endif
    float lsum = 0.f;
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float e = __expf(st[m] - m_new);   // = 0 when st[m] = -INF
      lsum += e;
#if GETP_KV8_V
      pB[m] = (_Float16)(e * PSCALE);
#else
      pB[m] = f32_to_bf16bits(e);
#endif
#if FLASH_P_HILO
      pLo[m] = f32_to_bf16bits(e - __bfloat162float(bf16bits_to_bf16(pB[m])));
#endif
    }

    // l is kept per g group; since all 4 groups share m_new, adding them up once in the
    // epilogue is enough - merging per tile would be redundant.
    l_run = l_run * alpha + lsum;
    m_run = m_new;

    // ---- O^T += V^T . P^T
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
#if GETP_KV8_V
      f16x4v vA;
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        vA[kk] = __builtin_bit_cast(_Float16, vs[4 * g + kk][c * TILE + ln]);
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc_o[c][i] *= alpha;
      acc_o[c] = __builtin_amdgcn_mfma_f32_16x16x16f16(vA, pB, acc_o[c], 0, 0, 0);
#else
      bf16x4 vA;
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk) vA[kk] = vs[4 * g + kk][c * TILE + ln];
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc_o[c][i] *= alpha;
      acc_o[c] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(vA, pB, acc_o[c], 0, 0, 0);
#endif
#if FLASH_P_HILO
      acc_o[c] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(vA, pLo, acc_o[c], 0, 0, 0);
#endif
    }

    __syncthreads();
    if (bt + TILE < t_end) stage_tile(rt0);
    __syncthreads();
    }
    if (bt0 + TILE >= t_end) break;
    {
      const int bt = bt0 + TILE;   // tile bt in LDS, tile bt + TILE (if any) in set 1
      if (bt + 2 * TILE < t_end) rt0 = load_tile(bt + 2 * TILE);

    // ---- S^T = K . Q^T : A[key][dim] read with one b64 instruction per chunk
    f32x4 st;
    #pragma unroll
    for (int i = 0; i < 4; ++i) st[i] = 0.f;
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
#if GETP_KV8_K
      f16x4 kA;
      *reinterpret_cast<uint2 *>(&kA) =
          *reinterpret_cast<const uint2 *>(&ks[ln][c * TILE + 4 * g]);
      st = __builtin_amdgcn_mfma_f32_16x16x16f16(kA, qH[c], st, 0, 0, 0);
#else
      bf16x4 kA;
      *reinterpret_cast<uint2 *>(&kA) =
          *reinterpret_cast<const uint2 *>(&ks[ln][c * TILE + 4 * g]);
      st = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(kA, qB[c], st, 0, 0, 0);
#endif
    }

    // ---- the lower mask. Out-of-range ks rows are zero-filled and so score 0, and
    // exp(0 - m) > 0 would corrupt the sum -> they have to be forced to -INFINITY.
    const int kbase = bt + 4 * g;
    #pragma unroll
    for (int m = 0; m < 4; ++m)
      if (kbase + m >= t_end) st[m] = -INFINITY;

    // ---- the tile max, shared by all 4 g groups of the same head. Merging BEFORE computing
    // m_new is mandatory: one g group may hold no valid key at all (m_tile = -INF) and would
    // produce NaN if it computed its own. After the merge, every tile
    // is guaranteed to hold at least one valid key, so m_tile is finite.
    float m_tile = fmaxf(fmaxf(st[0], st[1]), fmaxf(st[2], st[3]));
    m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 16));
    m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 32));

    const float m_new = fmaxf(m_run, m_tile);
    const float alpha = __expf(m_run - m_new);

    // ---- P^T = exp(S^T - m_new), rounded to bf16 straight away to serve as the B operand.
#if GETP_KV8_V
    f16x4v pB;
#else
    bf16x4 pB;
#endif
#if FLASH_P_HILO
    bf16x4 pLo;
#endif
    float lsum = 0.f;
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float e = __expf(st[m] - m_new);   // = 0 when st[m] = -INF
      lsum += e;
#if GETP_KV8_V
      pB[m] = (_Float16)(e * PSCALE);
#else
      pB[m] = f32_to_bf16bits(e);
#endif
#if FLASH_P_HILO
      pLo[m] = f32_to_bf16bits(e - __bfloat162float(bf16bits_to_bf16(pB[m])));
#endif
    }

    // l is kept per g group; since all 4 groups share m_new, adding them up once in the
    // epilogue is enough - merging per tile would be redundant.
    l_run = l_run * alpha + lsum;
    m_run = m_new;

    // ---- O^T += V^T . P^T
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
#if GETP_KV8_V
      f16x4v vA;
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        vA[kk] = __builtin_bit_cast(_Float16, vs[4 * g + kk][c * TILE + ln]);
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc_o[c][i] *= alpha;
      acc_o[c] = __builtin_amdgcn_mfma_f32_16x16x16f16(vA, pB, acc_o[c], 0, 0, 0);
#else
      bf16x4 vA;
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk) vA[kk] = vs[4 * g + kk][c * TILE + ln];
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc_o[c][i] *= alpha;
      acc_o[c] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(vA, pB, acc_o[c], 0, 0, 0);
#endif
#if FLASH_P_HILO
      acc_o[c] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(vA, pLo, acc_o[c], 0, 0, 0);
#endif
    }

    __syncthreads();
    if (bt + TILE < t_end) stage_tile(rt1);
    __syncthreads();
    }
  }
#else
  for (int bt = t_start; bt < t_end; bt += TILE) {
    uint4 regK[nbLoadsKV];
    uint4 regV[nbLoadsKV];
    const bool has_next = (bt + TILE < t_end);
    if (has_next) {
      #pragma unroll
      for (int i = 0; i < nbLoadsKV; ++i) {
        const int index_y = bt + TILE + ly + i * strideKV;
        if (index_y < t_end) {
          const size_t off = GETP_KV_OFF(index_y % cache_tcap, b, kv_h * head_dim + 8 * lx,
                                         batch_size, kv_dim, cache_tcap);
          regK[i] = *reinterpret_cast<const uint4 *>(key_cache + off);
          regV[i] = *reinterpret_cast<const uint4 *>(value_cache + off);
        } else {
          regK[i] = regV[i] = make_uint4(0, 0, 0, 0);
        }
      }
    }

    // ---- S^T = K . Q^T : A[key][dim] read with one b64 instruction per chunk
    f32x4 st;
    #pragma unroll
    for (int i = 0; i < 4; ++i) st[i] = 0.f;
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
#if GETP_KV8_K
      f16x4 kA;
      *reinterpret_cast<uint2 *>(&kA) =
          *reinterpret_cast<const uint2 *>(&ks[ln][c * TILE + 4 * g]);
      st = __builtin_amdgcn_mfma_f32_16x16x16f16(kA, qH[c], st, 0, 0, 0);
#else
      bf16x4 kA;
      *reinterpret_cast<uint2 *>(&kA) =
          *reinterpret_cast<const uint2 *>(&ks[ln][c * TILE + 4 * g]);
      st = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(kA, qB[c], st, 0, 0, 0);
#endif
    }

    // ---- the lower mask. Out-of-range ks rows are zero-filled and so score 0, and
    // exp(0 - m) > 0 would corrupt the sum -> they have to be forced to -INFINITY.
    const int kbase = bt + 4 * g;
    #pragma unroll
    for (int m = 0; m < 4; ++m)
      if (kbase + m >= t_end) st[m] = -INFINITY;

    // ---- the tile max, shared by all 4 g groups of the same head. Merging BEFORE computing
    // m_new is mandatory: one g group may hold no valid key at all (m_tile = -INF) and would
    // produce NaN if it computed its own. After the merge, every tile
    // is guaranteed to hold at least one valid key, so m_tile is finite.
    float m_tile = fmaxf(fmaxf(st[0], st[1]), fmaxf(st[2], st[3]));
    m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 16));
    m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 32));

    const float m_new = fmaxf(m_run, m_tile);
    const float alpha = __expf(m_run - m_new);

    // ---- P^T = exp(S^T - m_new), rounded to bf16 straight away to serve as the B operand.
#if GETP_KV8_V
    f16x4v pB;
#else
    bf16x4 pB;
#endif
#if FLASH_P_HILO
    bf16x4 pLo;
#endif
    float lsum = 0.f;
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float e = __expf(st[m] - m_new);   // = 0 when st[m] = -INF
      lsum += e;
#if GETP_KV8_V
      pB[m] = (_Float16)(e * PSCALE);
#else
      pB[m] = f32_to_bf16bits(e);
#endif
#if FLASH_P_HILO
      pLo[m] = f32_to_bf16bits(e - __bfloat162float(bf16bits_to_bf16(pB[m])));
#endif
    }

    // l is kept per g group; since all 4 groups share m_new, adding them up once in the
    // epilogue is enough - merging per tile would be redundant.
    l_run = l_run * alpha + lsum;
    m_run = m_new;

    // ---- O^T += V^T . P^T
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
#if GETP_KV8_V
      f16x4v vA;
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        vA[kk] = __builtin_bit_cast(_Float16, vs[4 * g + kk][c * TILE + ln]);
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc_o[c][i] *= alpha;
      acc_o[c] = __builtin_amdgcn_mfma_f32_16x16x16f16(vA, pB, acc_o[c], 0, 0, 0);
#else
      bf16x4 vA;
      #pragma unroll
      for (int kk = 0; kk < 4; ++kk) vA[kk] = vs[4 * g + kk][c * TILE + ln];
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc_o[c][i] *= alpha;
      acc_o[c] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(vA, pB, acc_o[c], 0, 0, 0);
#endif
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
#endif

  // ---- merge l across the 4 g groups (once only)
  // Barrier before reusing the ks/vs LDS region as `ot`: every wave must have finished reading
  // the last tile before any wave overwrites it.
  __syncthreads();
  l_run += __shfl_xor(l_run, 16);
  l_run += __shfl_xor(l_run, 32);

  if (ln < HPW) {
    const float sink  = attn_sinks[hg_self];
    const float m_fin = fmaxf(m_run, sink);
    const float a     = __expf(m_run - m_fin);
    const float l     = l_run * a + __expf(sink - m_fin);
#if GETP_KV8_V
    const float scale = a / l * (1.f / 32768.f);   // P went into the PV MFMA as e * 2^15
#else
    const float scale = a / l;
#endif
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c)
      *reinterpret_cast<float4 *>(&ot[waveIdx * HPW + ln][c * TILE + 4 * g]) =
          make_float4(acc_o[c][0] * scale, acc_o[c][1] * scale,
                      acc_o[c][2] * scale, acc_o[c][3] * scale);
  }
  __syncthreads();

  // ---- transpose through LDS, then write contiguously: each head is 256 contiguous bytes,
  // instead of 16 lanes scattering writes across different head rows.
  // Write bf16 DIRECTLY: tb's only consumer is the attn_o matmul, and it only reads bf16. The
  // rounding happens INSIDE the lane - each lane already holds exactly 4 consecutive floats of
  // one head - so no extra LDS exchange is needed. The arithmetic is identical to
  // fp32_to_bf16_kernel: that kernel re-read these same 4 floats from HBM and called
  // f32_to_bf16bits. One full HBM read-and-write pass over the tensor disappears.
  unsigned short *obase = reinterpret_cast<unsigned short *>(tb_bf16) +
                          (size_t)b * n_attn_heads * head_dim;
  #pragma unroll
  for (int idx = lane; idx < HPW * HEAD_DIM / 4; idx += 64) {
    const int hh = idx / (HEAD_DIM / 4);
    const int d4 = idx % (HEAD_DIM / 4);
    const float4 v = *reinterpret_cast<const float4 *>(&ot[waveIdx * HPW + hh][d4 * 4]);
    uint2 packed;
    packed.x = (unsigned)f32_to_bf16bits(v.x) |
               ((unsigned)f32_to_bf16bits(v.y) << 16);
    packed.y = (unsigned)f32_to_bf16bits(v.z) |
               ((unsigned)f32_to_bf16bits(v.w) << 16);
    const size_t off =
        (size_t)(kv_h * KV_MUL + waveIdx * HPW + hh) * head_dim + d4 * 4;
    *reinterpret_cast<uint2 *>(obase + off) = packed;
#if GETP_DUMP_ATTN
    // Debug path: the fp32 copy is kept so getp_dump_attn can compare.
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
      // inv_sqrt_d is folded in here rather than multiplied into each score after the __shfl,
      // where the value is wave-uniform so 63 of 64 lanes are wasted. Bit-exact because
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

      // One 128-bit LDS store instead of 8 16-bit stores: these 8 shorts are adjacent on the
      // second axis and the starting column is always a multiple of 16 bytes.
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

      // Online softmax per TILE instead of per key. For each i: one accumulator rescale and
      // 1 + MFMA_N exp calls, instead of 4 rescales and 8 exp calls.
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
            // Out-of-range ks rows are zero-padded and so score 0, and exp(0) = 1
            // would corrupt the sum -> they have to be forced to -INFINITY.
            s[j] = (j < nvalid) ? (__shfl(acc[i], j)) : -INFINITY;
            m_tile = fmaxf(m_tile, s[j]);
          }
          const float m_new = fmaxf(m_values[i], m_tile);
          const float alpha = __expf(m_values[i] - m_new);
          float lsum = 0.f;
          float osum = 0.f;
          #pragma unroll
          for (int j = 0; j < MFMA_N; ++j) {
            const float e = __expf(s[j] - m_new);   // = 0 when s[j] = -INF
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

        // One 128-bit LDS store instead of 8 16-bit stores: these 8 shorts are adjacent on the
        // second axis and the starting column is always a multiple of 16 bytes.
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
      // inv_sqrt_d is folded in here rather than multiplied into each score after the __shfl,
      // where the value is wave-uniform so 63 of 64 lanes are wasted. Bit-exact because
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

      // One 128-bit LDS store instead of 8 16-bit stores: these 8 shorts are adjacent on the
      // second axis and the starting column is always a multiple of 16 bytes.
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

      // Online softmax per TILE instead of per key. For each i: one accumulator rescale and
      // 1 + MFMA_N exp calls, instead of 4 rescales and 8 exp calls.
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
            // Out-of-range ks rows are zero-padded and so score 0, and exp(0) = 1
            // would corrupt the sum -> they have to be forced to -INFINITY.
            s[j] = (j < nvalid) ? (__shfl(acc[i], j)) : -INFINITY;
            m_tile = fmaxf(m_tile, s[j]);
          }
          const float m_new = fmaxf(m_values[i], m_tile);
          const float alpha = __expf(m_values[i] - m_new);
          float lsum = 0.f;
          float osum = 0.f;
          #pragma unroll
          for (int j = 0; j < MFMA_N; ++j) {
            const float e = __expf(s[j] - m_new);   // = 0 when s[j] = -INF
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

        // One 128-bit LDS store instead of 8 16-bit stores: these 8 shorts are adjacent on the
        // second axis and the starting column is always a multiple of 16 bytes.
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
// 1 = the mfma 16x16x16 attention kernel (see the note at its definition),
// 0 = the old mfma 4x4x4 path. Both are kept so they can still be A/B'd.
#ifndef FLASH_MFMA16
#define FLASH_MFMA16 1
#endif
// Enable one layer at a time to bisect: if only one side is wrong, the fault is in
// in the sliding-mask path, not in the multiply.
#ifndef FLASH_MFMA16_EVEN
#define FLASH_MFMA16_EVEN FLASH_MFMA16
#endif
#ifndef FLASH_MFMA16_ODD
#define FLASH_MFMA16_ODD  FLASH_MFMA16
#endif
#if GETP_KV_HEADMAJOR && !(FLASH_MFMA16_EVEN && FLASH_MFMA16_ODD)
#error "GETP_KV_HEADMAJOR is only wired into the mfma16 attention kernel; the older kernels read the old layout"
#endif
#if FLASH_MFMA16_EVEN || FLASH_MFMA16_ODD
static_assert(FLASH_DECODE_TILE_T == 16,
              "the mfma 16x16x16 kernel needs a tile step of 16");
#endif
// Dump tb to a file to compare the arithmetic between the two kernel versions. Only under GETP_DUMP_ATTN=1
// at compile time AND the GETP_DUMP_FILE environment variable at run time.
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

// The scale array of an int8 KV allocation: its upper half. The int8 data takes kv_elems bytes of
// the kv_elems bf16 slots transformer.cpp allocates (the same formula), so the scales - kv_elems/64
// floats - start at the next 256-byte boundary and end well inside the allocation.
static inline float *getp_kv8_scales(void *cache, const Config *p, int B) {
  const int kv_dim = p->head_dim * p->n_kv_heads;
  const int even_layers = (p->n_layers + 1) / 2;
  const int even_tcap = (p->sliding_window > 0 ? p->sliding_window : 1);
  int odd_tcap = p->seq_len / 2;
  if (odd_tcap < 1) odd_tcap = 1;
  const size_t total_t = (size_t)even_layers * (size_t)even_tcap +
                         (size_t)(p->n_layers - even_layers) * (size_t)odd_tcap;
  const size_t kv_elems = (size_t)B * total_t * (size_t)kv_dim +
                          (size_t)p->n_layers * GETP_KV_LAYER_PAD(B, kv_dim);
  return reinterpret_cast<float *>(reinterpret_cast<char *>(cache) +
                                   ((kv_elems + 255) & ~(size_t)255));
}

static inline void getp_flash_attn_decode_bf16(
    __hip_bfloat16 *tb_bf16, float *tb, const __hip_bfloat16 *key_cache_layer,
    const __hip_bfloat16 *value_cache_layer, const float *q, const int *mask_on,
    const float *attn_sinks_layer, int head_dim, int n_attn_heads,
    int n_kv_heads, int pos, int seq_len, int sliding_window, int layer_id,
    int batch_size, int cache_tcap, hipStream_t stream,
    // GETP_KV8_K / _V: the layer's scales (nullptr for a bf16 tensor)
    const float *k_scale_layer = nullptr, const float *v_scale_layer = nullptr) {
#if GETP_KV8_ANY && !(FLASH_MFMA16_EVEN && FLASH_MFMA16_ODD)
#error "GETP_KV8_* needs the mfma16 attention kernel on both layer kinds"
#endif

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
       mask_on, k_scale_layer, v_scale_layer,
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
      mask_on, k_scale_layer, v_scale_layer,
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


// Split count of getp_matmul_router_bf16, factored out so the fused kernel can reproduce it.
static inline int getp_router_splits(int M, int N, int K) {
  constexpr int BM = 32, BK = 32;
  const int BNt = 32 * GETP_BN_AGG;
  const int gx = (N + BNt - 1) / BNt, gy = (M + BM - 1) / BM;
  const int grid_xy = max(1, gx) * max(1, gy);
  int dev = 0, cu = 104;
  HIP_CHECK(hipGetDevice(&dev));
  HIP_CHECK(hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev));
  if (cu <= 0) cu = 104;
  const int target_cta = cu * 4;
  int splits = (grid_xy >= target_cta) ? 1 : (target_cta + grid_xy - 1) / grid_xy;
  splits = min(8, max(1, splits));
  return min(splits, max(1, (K + BK - 1) / BK));
}

#if GETP_ROUTER_FUSED
// The router of getp_matmul_router_bf16, same bits, in one kernel. A workgroup owns a 16*RB x 16*CBW
// output tile; its wave s computes split s: kBeg = s*kChunk, and for each 16 x 16 block the SAME
// mfma chain as matmul_kernel_splitk_store_tuned, which walks 32-k tiles from kBeg while bk < kEnd,
// two 16-k MFMAs per tile, zero-filling every 8-element group with kk + 7 >= kEnd. The partials then
// meet in LDS and are summed in split order from 0.f, plus the bias, as reduce_splitk_with_bias does.
// Each A load feeds CBW MFMAs and each B load RB; loads run one G-step chunk ahead of the MFMAs.
constexpr int GETP_ROUTER_MAXT = 12;   // 32-k tiles per split: kChunk <= 384 (the launcher checks)
template <int SPLITS, int RB, int CBW, int G>
__global__ __launch_bounds__(64 * SPLITS) void router_fused_bf16_kernel(
    float *__restrict__ out, const __hip_bfloat16 *__restrict__ xb,
    const __hip_bfloat16 *__restrict__ w, const __hip_bfloat16 *__restrict__ bias,
    int M, int N, int K, int kChunk) {
  constexpr int NS = 2 * GETP_ROUTER_MAXT, NCH = NS / G;
  static_assert(NS % G == 0, "G must divide the step count");
  __shared__ float red[SPLITS][16 * RB][16 * CBW];
  const int split = threadIdx.x >> 6, lane = threadIdx.x & 63;
  const int xMF = lane & 15, yMF = lane >> 4;
  const int r0 = blockIdx.x * 16 * RB, c0 = blockIdx.y * 16 * CBW;
  const int kBeg = split * kChunk, kEnd = min(K, kBeg + kChunk);
  const int nsteps = kBeg < kEnd ? 2 * ((kEnd - kBeg + 31) / 32) : 0;
  const unsigned short *xs = reinterpret_cast<const unsigned short *>(xb);
  const unsigned short *ws = reinterpret_cast<const unsigned short *>(w);
  // A: row r, k .. k+3; B: column c, k .. k+3 - the operand layout of the block kernel's LDS reads
  uint2 a[2][RB][G], b[2][CBW][G];
  auto load = [&](int ch, int buf) {
    #pragma unroll
    for (int g = 0; g < G; ++g) {
      const int u = ch * G + g;
      const int k = kBeg + 16 * u + 4 * yMF;
      const int kk = kBeg + 16 * u + ((4 * yMF) & ~7);   // the 8-element group the block kernel loads
      const bool kok = u < nsteps && kk + 7 < kEnd;
      #pragma unroll
      for (int rb = 0; rb < RB; ++rb) {
        const int row = r0 + 16 * rb + xMF;
        a[buf][rb][g] = (kok && row < M)
                            ? *reinterpret_cast<const uint2 *>(xs + (size_t)row * K + k)
                            : make_uint2(0u, 0u);
      }
      #pragma unroll
      for (int cb = 0; cb < CBW; ++cb) {
        const int col = c0 + 16 * cb + xMF;
        b[buf][cb][g] = (kok && col < N)
                            ? *reinterpret_cast<const uint2 *>(ws + (size_t)col * K + k)
                            : make_uint2(0u, 0u);
      }
    }
  };
  f32x4 d[RB][CBW];
  #pragma unroll
  for (int rb = 0; rb < RB; ++rb)
    #pragma unroll
    for (int cb = 0; cb < CBW; ++cb) d[rb][cb] = f32x4{0, 0, 0, 0};
  load(0, 0);
  #pragma unroll
  for (int ch = 0; ch < NCH; ++ch) {
    if (ch + 1 < NCH) load(ch + 1, (ch + 1) & 1);
    #pragma unroll
    for (int g = 0; g < G; ++g) {
      if (ch * G + g < nsteps) {
        #pragma unroll
        for (int rb = 0; rb < RB; ++rb) {
          const uint2 ua = a[ch & 1][rb][g];
          const bf16x4 av = {(unsigned short)(ua.x & 0xFFFF), (unsigned short)(ua.x >> 16),
                             (unsigned short)(ua.y & 0xFFFF), (unsigned short)(ua.y >> 16)};
          #pragma unroll
          for (int cb = 0; cb < CBW; ++cb) {
            const uint2 ub = b[ch & 1][cb][g];
            const bf16x4 bv = {(unsigned short)(ub.x & 0xFFFF), (unsigned short)(ub.x >> 16),
                               (unsigned short)(ub.y & 0xFFFF), (unsigned short)(ub.y >> 16)};
            d[rb][cb] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, d[rb][cb], 0, 0, 0);
          }
        }
      }
    }
  }
  #pragma unroll
  for (int rb = 0; rb < RB; ++rb)
    #pragma unroll
    for (int cb = 0; cb < CBW; ++cb)
      #pragma unroll
      for (int i = 0; i < 4; ++i) red[split][16 * rb + 4 * yMF + i][16 * cb + xMF] = d[rb][cb][i];
  __syncthreads();
  for (int t = threadIdx.x; t < 16 * RB * 16 * CBW; t += 64 * SPLITS) {
    const int r = t / (16 * CBW), c = t % (16 * CBW);
    const int row = r0 + r, col = c0 + c;
    if (row >= M || col >= N) continue;
    float sum = 0.f;
    #pragma unroll
    for (int sp = 0; sp < SPLITS; ++sp) sum += red[sp][r][c];
    if (bias) sum += __bfloat162float(bias[col]);
    out[(size_t)row * N + col] = sum;
  }
}

// True when the fused router reproduces getp_matmul_router_bf16 for this shape.
static inline bool getp_router_fused_ok(int M, int N, int K) {
  const int splits = getp_router_splits(M, N, K);
  const int kChunk = (K + splits - 1) / splits;
  return splits == 8 && kChunk <= 32 * GETP_ROUTER_MAXT && kChunk % 8 == 0 && K % 4 == 0 &&
         N % 16 == 0;
}

// xout[M x N] = xb[M x K] . w[N x K]^T + b, with xb the bf16 copy of what getp_matmul_router_bf16
// reads in fp32. Only call when getp_router_fused_ok.
// Tile per wave, measured on norm + router (smallbench, us per layer; the split-K pair: 113 / 70):
//              RB,CBW = 1,1   1,2   2,1   2,2   4,1   1,4
//   20B  N =  32:        64    59    58    66    73     -
//   120B N = 128:        81    68    67    57    68    71
static inline void getp_matmul_router_fused(float *xout, const __hip_bfloat16 *xb,
                                            const __hip_bfloat16 *w, const __hip_bfloat16 *b,
                                            int n, int d, int batch_size, hipStream_t stream) {
  const int M = batch_size, N = d, K = n;
  const int kChunk = (K + 7) / 8;
  constexpr int RB = 2;
  const dim3 grid_x((unsigned)((M + 16 * RB - 1) / (16 * RB)));
  if (N % 32 == 0 && N >= 64)
    router_fused_bf16_kernel<8, RB, 2, GETP_ROUTER_G><<<dim3(grid_x.x, N / 32), 64 * 8, 0, stream>>>(
        xout, xb, w, b, M, N, K, kChunk);
  else
    router_fused_bf16_kernel<8, RB, 1, GETP_ROUTER_G><<<dim3(grid_x.x, N / 16), 64 * 8, 0, stream>>>(
        xout, xb, w, b, M, N, K, kChunk);
}
#endif

static inline void getp_matmul_router_bf16(float *xout, float *x, __hip_bfloat16 *w, __hip_bfloat16 *b, int n, int d, int batch_size, hipStream_t stream) {

  const int M = batch_size, N = d, K = n;
  constexpr int BM = 32, BN = 32, BK = 32;
  const int BNt = BN * GETP_BN_AGG;

  dim3 block(256);
  const int gx = (N + BNt - 1) / BNt;
  const int gy = (M + BM - 1) / BM;
  const int splits = getp_router_splits(M, N, K);

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
// The macro tile below is used only with GETP_ATTN_O_AUTO_TILE=0. By default the launcher picks
// between BN 128 / WN 64 and BN 64 / WN 32 at run time, because no single tile suits both models:
// the grid is (N/BN, BATCH_SIZE/BM) and the two models run different batch sizes. Measured, paired:
//   120B (BATCH_SIZE 768):  BN 128 -> 64  +0,6 %    (138 workgroups -> 270)
//   20B  (BATCH_SIZE 1536): BN 128 -> 64  -2,7 %    (276 workgroups -> 540)
// At BN 128 the kernel runs at 3 waves/SIMD, i.e. 3 workgroups per CU, 312 slots on 104 CUs, so the
// 120B fills 44 % of the machine and the 20B 88 %. The rule is to halve the tile when the wide one
// would fill less than half the resident slots; see getp_matmul_attn_o_bf16_acc.
#ifndef GETP_ATTN_O_AUTO_TILE
#define GETP_ATTN_O_AUTO_TILE 1
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
// Derived from the tile: the kernel body uses TILE_N = 64 * (BM/WM)*(BN/WN) threads.
// Overriding with -D is only allowed if it still equals this expression (there is a static_assert).
#define MATMUL_ATTN_O_WAVES_PER_BLOCK (((MATMUL_ATTN_O_BLOCK_ROWS) / (MATMUL_ATTN_O_WARP_TILE_M)) * ((MATMUL_ATTN_O_BLOCK_COLS) / (MATMUL_ATTN_O_WARP_TILE_N)))
#endif

// ===========================================================================
// attn_o LDS: from k-major [BK][BM+GETP_LDS_PAD] to the 8-byte SLOT layout.
//
//   Slot (k0, row) holds the 4 shorts k = 4*k0 .. 4*k0+3 of the SAME row.
//   * One MFMA fragment (A[i][k] or B[k][j]) = exactly 4 consecutive k shorts of one
//     row  ->  ONE ds_read_b64 instead of FOUR ds_read_u16.
//   * One uint4 loaded from HBM = 8 consecutive k shorts = two slots  ->  TWO ds_write_b64
//     instead of EIGHT ds_write_b16.
//
// Bank (32 banks x 4 bytes = 128 bytes/cycle; b64 is served in 16-lane phases, each
// lane taking ONE even/odd bank pair, conflict-free <=> the 16 lanes cover all 16 pairs):
//
//   READ   the 16 lanes of one phase share k0 (k0 depends only on lane>>4) and take 16 rows
//         consecutive (Ai = Bj = lane&15):
//             byte = C + 8*(lane&15)  ->  128 contiguous bytes  ->  32 banks, each
//          each bank used once. TRUE FOR EVERY PADS - the read puts no constraint on the pad.
//
//   WRITE  the 16 lanes of one phase have a = lane % NKC (k chunk, the loader's fast axis) and
//          b = lane / NKC (row), NKC = BK/8. The destination slot is k0 = 2a (and 2a+1), so
//             dword = 2*(k0*S + row),  S = BM + PADS (or BN + PADS)
//             bank  = (4*S*a + 2*b) mod 32.
//         The pad has to make the 16 lanes cover the 16 even banks:
//             BK=16 (NKC=2): 4S == 16 (mod 32) -> S == 4 (mod 8) -> PADS 4, 12
//             BK=32 (NKC=4): 4S == 8 hay 24    -> S == 2 hay 6   -> PADS 2, 6, 10, 14
//             BK=64 (NKC=8): 4S == 4,12,20,28  -> S le           -> PADS 1, 3, 5, 7
//          there is NO constant that works for every BK, so PADS has to be DERIVED from BK
//          (this kernel defaults to BK=16, the winning config is BK=32, sweeps use BK=64).
//
// The LDS size does NOT grow: BK*(BM+BN+2*PADS)*2 bytes, equal to or smaller than the old
// BK*(BM+BN+2*GETP_LDS_PAD)*2 vi PADS <= 4.
//     BK=16 PADS=4:  8448 B  (cu  8448 B)
//     BK=32 PADS=2: 16640 B  (cu 16896 B)
//     BK=64 PADS=1: 33024 B  (cu 33792 B)
// ===========================================================================
constexpr bool attn_o_lds_write_conflict_free(int S, int NKC) {
  if (NKC <= 0 || NKC > 16 || 16 % NKC != 0) return false;
  int seen = 0;
  for (int l = 0; l < 16; ++l) {
    const int a = l % NKC;                 // loader k chunk
    const int b = l / NKC;                 // row within the phase
    int bank = (4 * S * a + 2 * b) % 32;
    if (bank < 0) bank += 32;
    if (bank & 1) return false;            // a slot must be 8-byte aligned
    const int bit = 1 << (bank >> 1);
    if (seen & bit) return false;          // two lanes on the same bank pair
    seen |= bit;
  }
  return seen == 0xFFFF;                   // covers all 16 bank pairs
}

// The smallest valid PADS for BOTH tiles (X uses S = BM+PADS, W uses S = BN+PADS).
constexpr int attn_o_lds_pad_slots(int BM_, int BN_, int NKC) {
  for (int p = 0; p < 16; ++p)
    if (attn_o_lds_write_conflict_free(BM_ + p, NKC) &&
        attn_o_lds_write_conflict_free(BN_ + p, NKC))
      return p;
  return -1;
}

// attn_o's OWN pad - NOT shared with GETP_LDS_PAD (that macro still belongs to
// qkv:1392-1393 and logits:3656-3657). The two macros count two different things:
//   GETP_LDS_PAD                : a count of SHORTS along the ROW axis of one k plane;
//                                 any value is valid, it only changes performance;
//                                 (autotune.sh:38-40 and run2.sh:35-37 swept
//                                 2 / 8 / 16).
//   MATMUL_ATTN_O_LDS_PAD_SLOTS : a count of 8-BYTE SLOTS along the row axis of one slot plane;
//                                 constrained, and the constraint depends on BK (table above).
//                                 Putting 2 / 8 / 16 here either fails to compile or turns into
//                                 a two-dimensional write conflict that reads as measurement noise.
// -1 = derive from (BM, BN, BK). Only set it by hand to sweep 6/10/14 at BK=32.
#ifndef MATMUL_ATTN_O_LDS_PAD_SLOTS
#define MATMUL_ATTN_O_LDS_PAD_SLOTS (-1)
#endif

// Pack two consecutive dwords of a uint4 into one slot of 4 x bf16. A pure retype:
// both VGPRs are already in place (the result of global_load_dwordx4), so it emits NO
// VALU instruction - unlike the old version's 4 half-word extracts (& 0xFFFF / >> 16).
// __builtin_bit_cast rather than union punning: same codegen, standard spelling.
// The attn_o GEMM's HBM loads can go two k tiles ahead, branch-free (the change GETP_MLP1_AHEAD2
// describes; template argument AHEAD2), chosen per tile shape. Op test: the wide tile (BN 128, the
// 20B, M = 1536) 503 -> 446 us, the narrow one (BN 64, the 120B, M = 768) 284 -> 302 us, so the
// narrow tile keeps the one-ahead loop. Bit-exact either way. GETP_ATTN_O_MINW_WIDE promises the
// register allocator that many waves per SIMD for the wide tile (1 = no promise; 3, which would keep
// three blocks per CU, measured 489 us).
#ifndef GETP_ATTN_O_AHEAD2_WIDE
#define GETP_ATTN_O_AHEAD2_WIDE 1
#endif
#ifndef GETP_ATTN_O_AHEAD2_NARROW
#define GETP_ATTN_O_AHEAD2_NARROW 0
#endif
#ifndef GETP_ATTN_O_MINW_WIDE
#define GETP_ATTN_O_MINW_WIDE 1
#endif
// MFMAs between two consecutive HBM loads of the two-ahead loop.
#ifndef GETP_ATTN_O_LOAD_GAP
#define GETP_ATTN_O_LOAD_GAP 4
#endif
template <int V> struct AttnOParity { static constexpr int value = V; };
__device__ __forceinline__ bf16x4 attn_o_pack2(unsigned int lo, unsigned int hi) {
  struct Pair { unsigned int a, b; };
  static_assert(sizeof(Pair) == sizeof(bf16x4), "pack2: size mismatch");
  Pair p; p.a = lo; p.b = hi;
  return __builtin_bit_cast(bf16x4, p);
}

template<
  int BM, int BN, int BK,
  int WM, int WN,
  int WARPS_PER_BLOCK,
  bool ACC = false,
  int THREADS = 64 * WARPS_PER_BLOCK,
  // AHEAD2: HBM loads two k tiles ahead (see the k loop). MINW: the minimum waves per SIMD promised
  // to the register allocator (1 = no promise). The launcher picks both per tile shape.
  bool AHEAD2 = false, int MINW = 1
>
__global__ void __launch_bounds__(THREADS, MINW)
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
  // LATENT BUG FIXED (the same class as forward.hip:1371-1372 in qkv): xInBlockTile runs along
  // the N axis so it must divide by BN/WN; yInBlockTile runs along the M axis so it is the
  // quotient by the same denominator. The old code used (BM/WM) for both - harmless when
  // BM/WM == BN/WN (every configuration in use, including -DMATMUL_ATTN_O_BLOCK_DEPTH=32)
  // but WRONG for attno_cfg.txt line 5 (bm256) and line 8 (bk32_bm256): there BM/WM=4,
  // BN/WN=2 -> xInBlockTile reaches 3 -> col = 3*64+63 = 255 >= BN=128, a read outside
  // tile W and writes the wrong column, and only half the rows get computed. These two lines
  // generate IDENTICAL CODE to the old version for every configuration with BM/WM == BN/WN.
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
  const int loadW_k0   = 2 * loadWIdx_x;      // slot at the start of the uint4 within tile W
  constexpr int strideW = THREADS / (BK / 8);
  // Removed nbLoadsX / loadXIdx_x / loadXIdx_y / strideX: they were NEVER
  // used (the X load path goes through the *_x8 variables below). Keeping them is exactly the
  // 'an old stride survives an edit' class of bug.

  constexpr int KP   = 4;             // short / o = 8 byte = 1 ds_read_b64
  constexpr int KSL  = BK / KP;       // so mat o (k0)
  constexpr int NKC  = BK / 8;        // number of loader k chunks (uint4 = 8 shorts)
  constexpr int PADS = (MATMUL_ATTN_O_LDS_PAD_SLOTS >= 0)
                         ? MATMUL_ATTN_O_LDS_PAD_SLOTS
                         : attn_o_lds_pad_slots(BM, BN, NKC);
  static_assert(BK % 8 == 0, "the loader loads a uint4 = 8 shorts along k");
  static_assert(BK % KP == 0, "BK must be a multiple of KP = 4");
  static_assert(PADS >= 0,
                "no PADS value makes ds_write_b64 free of bank conflicts; "
                "see attn_o_lds_pad_slots()");
  static_assert(attn_o_lds_write_conflict_free(BM + PADS, NKC),
                "MATMUL_ATTN_O_LDS_PAD_SLOTS is wrong: ds_write_b64 conflict on tile X "
                "bank (BK=16 -> 4, BK=32 -> 2, BK=64 -> 1)");
  static_assert(attn_o_lds_write_conflict_free(BN + PADS, NKC),
                "MATMUL_ATTN_O_LDS_PAD_SLOTS is wrong: ds_write_b64 conflict on tile W "
                "bank (BK=16 -> 4, BK=32 -> 2, BK=64 -> 1)");
  static_assert(sizeof(bf16x4) == 8 && sizeof(Afloat) == 8 &&
                sizeof(Bfloat) == 8, "an MFMA fragment must be 8 bytes");

  // The new names (xsT/wsT) are deliberate: any remaining xs[..][..] site becomes a COMPILE
  // ERROR rather than a silently wrong address.
  // The element type is bf16x4 (alignment 8), so EVERY access is one ds_read_b64 /
  // ds_write_b64 - no reinterpret_cast needed anywhere. __align__(16)
  // guarantees the array base: a __shared__ unsigned short array is only 2-byte aligned.
  __shared__ __align__(16) bf16x4 xsT[KSL][BM + PADS];
  __shared__ __align__(16) bf16x4 wsT[KSL][BN + PADS];
  static_assert(sizeof(xsT) + sizeof(wsT) <= 64 * 1024, "LDS > 64 KB per CU");

  CDfloat acc_local[nIterWaveM * nIterWaveN] = {0};

  // AGGRESSIVE X load: uint4 for 8 bf16 elements (16 bytes)
  constexpr int X_LOAD_VEC = 8;
  const int loadX_x8 = threadIdx.x % (BK / X_LOAD_VEC);
  const int loadX_y8 = threadIdx.x / (BK / X_LOAD_VEC);
  const int loadX_k0 = 2 * loadX_x8;          // slot at the start of the uint4 within tile X
  constexpr int strideX8 = THREADS / (BK / X_LOAD_VEC);
  constexpr int nbLoadsX8 = BM * (BK / X_LOAD_VEC) / THREADS;
  // The loader has to cover the tile. Without these two asserts, wm32/wn32 at BK=16 leaves part
  // nbLoadsX8 = 128*2/512 = 0, and bm256 at BK=16 gives nbLoadsW = 128*2/512 = 0:
  // of the tile unloaded and the result is SILENTLY wrong (and faster). Three config lines
  // in attno_cfg.txt must be paired with -DMATMUL_ATTN_O_BLOCK_DEPTH=32 or higher.
  static_assert(nbLoadsX8 >= 1 && (BM * (BK / X_LOAD_VEC)) % THREADS == 0,
                "loader X does not cover the tile: BM*(BK/8) must be a multiple of THREADS");
  static_assert(nbLoadsW >= 1 && (BN * (BK / 8)) % THREADS == 0,
                "loader W does not cover the tile: BN*(BK/8) must be a multiple of THREADS");

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

      // 8 consecutive k shorts = two slots -> two ds_write_b64 (was: 8 ds_write_b16).
      // The half-word order is kept EXACTLY: v.x low = k+0, v.x high = k+1, v.y low =
      // k+2, ... -> the same byte image -> bit-exact.
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

  if constexpr (AHEAD2) {
  // Branch-free HBM loads two k tiles ahead: step t loads tile t+2 into set t&1 and writes tile
  // t+1 from set (t+1)&1 to LDS. Unrolled by two so each set has a fixed register name.
  uint4 sX[2][nbLoadsX8], sW[2][nbLoadsW];
  auto load_set = [&](uint4 (&rX)[nbLoadsX8], uint4 (&rW)[nbLoadsW], int kt) {
    const int kc = min(kt, K - BK);
    #pragma unroll
    for (int i = 0; i < nbLoadsX8; ++i) {
      const int rg = min(BM * (int)blockIdx.y + loadX_y8 + i * strideX8, M - 1);
      rX[i] = *reinterpret_cast<const uint4*>(x + (size_t)rg * K + kc + X_LOAD_VEC * loadX_x8);
    }
    #pragma unroll
    for (int i = 0; i < nbLoadsW; ++i) {
      const int cy = min(BN * (int)blockIdx.x + loadWIdx_y + i * strideW, N - 1);
      rW[i] = *reinterpret_cast<const uint4*>(&w[(size_t)cy * K + kc + 8 * loadWIdx_x]);
    }
  };
  load_set(sX[1], sW[1], BK);
  auto kstep = [&](auto S_, int bkIdx) {
    constexpr int S = decltype(S_)::value;
    load_set(sX[S], sW[S], bkIdx + 2 * BK);
    // MFMA compute loop - fully unrolled for maximum ILP
    #pragma unroll
    for (int k = 0; k < BK; k += MFMA_K) {
      // base_k = k + Ak is a multiple of 4 (k a multiple of 16, Ak = 4*(lane>>4)), so all 4
      // k shorts fit INSIDE one slot -> slot index = base_k >> 2. The index stays 32-bit:
      // LDS addresses are 32-bit, so no (size_t) goes into the hot loop.
      const int ak0 = (k + Ak) >> 2;
      const int bk0 = (k + Bk) >> 2;

      #pragma unroll
      for (int im = 0; im < nIterWaveM; ++im) {
        const int row = yInBlockTile * WM + MFMA_M * im + Ai;
        // ONE ds_read_b64 (was: 4 ds_read_u16 with stride BM+GETP_LDS_PAD).
        const Afloat av = xsT[ak0][row];

        #pragma unroll
        for (int in = 0; in < nIterWaveN; ++in) {
          const int col = xInBlockTile * WN + MFMA_N * in + Bj;

          // ONE ds_read_b64 (was: 4 ds_read_u16 with stride BN+GETP_LDS_PAD).
          const Bfloat bv = wsT[bk0][col];

          const int idx = im * nIterWaveN + in;
          acc_local[idx] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc_local[idx], 0, 0, 0);
        }
      }
    }
    // LDS reads first (the MFMAs wait on them), then one HBM load every GETP_ATTN_O_LOAD_GAP MFMAs.
    constexpr int NMF = (BK / MFMA_K) * nIterWaveM * nIterWaveN;
    constexpr int NLD = nbLoadsX8 + nbLoadsW;
    static_assert(NLD * GETP_ATTN_O_LOAD_GAP <= NMF, "GETP_ATTN_O_LOAD_GAP too large for this tile");
    __builtin_amdgcn_sched_group_barrier(0x0100, (BK / MFMA_K) * (nIterWaveM + nIterWaveN), 0);
    #pragma unroll
    for (int i = 0; i < NLD; ++i) {
      __builtin_amdgcn_sched_group_barrier(0x0008, GETP_ATTN_O_LOAD_GAP, 0);
      __builtin_amdgcn_sched_group_barrier(0x0020, 1, 0);
    }
    __builtin_amdgcn_sched_group_barrier(0x0008, NMF - NLD * GETP_ATTN_O_LOAD_GAP, 0);
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();
    {
      const uint4 (&regX8)[nbLoadsX8] = sX[1 - S];
      const uint4 (&regW)[nbLoadsW] = sW[1 - S];
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
  };
  #pragma unroll 1
  for (int bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {   // K % (2 * BK) == 0: the launcher checks
    kstep(AttnOParity<0>{}, bkIdx);
    kstep(AttnOParity<1>{}, bkIdx + BK);
  }
  } else {
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
      // base_k = k + Ak is a multiple of 4 (k a multiple of 16, Ak = 4*(lane>>4)), so all 4
      // k shorts fit INSIDE one slot -> slot index = base_k >> 2. The index stays 32-bit:
      // LDS addresses are 32-bit, so no (size_t) goes into the hot loop.
      const int ak0 = (k + Ak) >> 2;
      const int bk0 = (k + Bk) >> 2;

      #pragma unroll
      for (int im = 0; im < nIterWaveM; ++im) {
        const int row = yInBlockTile * WM + MFMA_M * im + Ai;
        // ONE ds_read_b64 (was: 4 ds_read_u16 with stride BM+GETP_LDS_PAD).
        const Afloat av = xsT[ak0][row];

        #pragma unroll
        for (int in = 0; in < nIterWaveN; ++in) {
          const int col = xInBlockTile * WN + MFMA_N * in + Bj;

          // ONE ds_read_b64 (was: 4 ds_read_u16 with stride BN+GETP_LDS_PAD).
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
          // ACC: add straight into the residual (dev_x) instead of writing tb2 for vecadd to
          // read back. Every cell (gy,gx) is produced by EXACTLY ONE thread - there is no
          // split-k in this kernel - so read-modify-write is safe without an atomic.
          // Bit-exact: v is exactly the fp32 value tb2 would have held.
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

  // Note on __launch_bounds__: HIP's SECOND argument is the MINIMUM NUMBER OF WAVES PER SIMD,
  // not blocks per CU. This kernel passes only one argument (THREADS) so it is unaffected; if
  // anyone adds a second argument, the promised workgroup count is 4*CTA/WARPS, not CTA.
  dim3 block(64 * WARPS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  matmul_attn_o_bf16_kernel_tuned<BM,BN,BK,WM,WN,WARPS,false>
      <<<grid, block, 0, stream>>>(xout, x_bf16, w, b, M, N, K);

}

// As above but ADDS into xout (the fp32 residual) instead of overwriting. Removes the tb2
// tensor entirely: one 17,7 MB write and one 17,7 MB read per layer.
static inline void getp_matmul_attn_o_bf16_acc(
    float *x_resid, const __hip_bfloat16 *x_bf16,
    const __hip_bfloat16 *w, const __hip_bfloat16 *b,
    int n, int d, int batch_size, hipStream_t stream) {

  const int M = batch_size, N = d, K = n;

  constexpr int BM = MATMUL_ATTN_O_BLOCK_ROWS;
  constexpr int BK = MATMUL_ATTN_O_BLOCK_DEPTH;
  constexpr int WM = MATMUL_ATTN_O_WARP_TILE_M;
#if GETP_ATTN_O_AUTO_TILE
  // Wide = BN 128 / WN 64, narrow = BN 64 / WN 32; both are 4 waves = 256 threads.
  constexpr int BNW = 128, WNW = 64, WARPSW = (BM / WM) * (BNW / WNW);
  constexpr int BNN = 64, WNN = 32, WARPSN = (BM / WM) * (BNN / WNN);
  static_assert(WARPSW == WARPSN, "both tiles must keep the same block size");
  // Resident workgroup slots for the WIDE kernel, asked of the runtime once per thread (one
  // thread per device) rather than hard-wired, so the rule survives a tile or GPU change.
  // The two-ahead loop (GETP_ATTN_O_AHEAD2_*) needs an even number of k tiles.
  if ((GETP_ATTN_O_AHEAD2_WIDE || GETP_ATTN_O_AHEAD2_NARROW) && K % (2 * BK) != 0) {
    fprintf(stderr, "attn_o: GETP_ATTN_O_AHEAD2_* needs K %% (2 * BK) == 0 (K = %d)\n", K);
    abort();
  }
  // The tile rule stays tied to the one-ahead kernel's occupancy (3 blocks per CU): the two-ahead
  // wide kernel fits only 2, which would move the 120B (M = 768) to the wide tile - slower there.
  static thread_local int slots = 0;
  if (slots == 0) {
    int dev = 0, per_cu = 0;
    HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_cu, matmul_attn_o_bf16_kernel_tuned<BM, BNW, BK, WM, WNW, WARPSW, true>,
        64 * WARPSW, 0));
    slots = (per_cu > 0 ? per_cu : 1) * prop.multiProcessorCount;
  }
  const int rows = (M + BM - 1) / BM;
  const int wide = ((N + BNW - 1) / BNW) * rows;
  if (2 * wide < slots) {
    // Less than half the machine: twice as many, half-width workgroups keep more CUs busy.
    matmul_attn_o_bf16_kernel_tuned<BM, BNN, BK, WM, WNN, WARPSN, true, 64 * WARPSN,
                                    GETP_ATTN_O_AHEAD2_NARROW != 0>
        <<<dim3((N + BNN - 1) / BNN, rows), dim3(64 * WARPSN), 0, stream>>>(
            x_resid, x_bf16, w, b, M, N, K);
  } else {
    matmul_attn_o_bf16_kernel_tuned<BM, BNW, BK, WM, WNW, WARPSW, true, 64 * WARPSW,
                                    GETP_ATTN_O_AHEAD2_WIDE != 0, GETP_ATTN_O_MINW_WIDE>
        <<<dim3((N + BNW - 1) / BNW, rows), dim3(64 * WARPSW), 0, stream>>>(
            x_resid, x_bf16, w, b, M, N, K);
  }
#else
  constexpr int BN = MATMUL_ATTN_O_BLOCK_COLS;
  constexpr int WN = MATMUL_ATTN_O_WARP_TILE_N;
  constexpr int WARPS = MATMUL_ATTN_O_WAVES_PER_BLOCK;

  dim3 block(64 * WARPS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  matmul_attn_o_bf16_kernel_tuned<BM,BN,BK,WM,WN,WARPS,true>
      <<<grid, block, 0, stream>>>(x_resid, x_bf16, w, b, M, N, K);
#endif
}


// Map float -> uint32 preserving order, so a floating-point comparison becomes an
// unsigned integer comparison: non-negative numbers move to the upper half, negatives are flipped.
__device__ inline unsigned int order_preserving_bits(float v) {
  unsigned int b = __float_as_uint(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ inline float order_preserving_bits_inv(unsigned int k) {
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
// A monotone key on (value, -idx): a larger value wins, and on a tie the SMALLER idx wins.
// That reduces the argmax to a single atomicMax - no CAS loop and no
// dependence on block arrival order.
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
  // This used to be a CAS loop whose comparison used a RELATIVE tolerance
  // (1e-6 * max(|v|,|ov|)), which is NOT a transitive relation: A ~ B and B ~ C do not
  // imply A ~ C. With thousands of column blocks racing for one cell, the winner depended on
  // arrival order, and the result changed between two runs of the same binary. The monotone key
  // above turns this into a single atomicMax with a strict total order.
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

// Group swizzle for the logits: G ROW blocks run fastest within each group, so that G blocks
// share one weight tile in L2. G has to be small enough that G X tiles still fit L2 (see lgswzg).
#ifndef GETP_LOGITS_SWZ
#define GETP_LOGITS_SWZ 1   // on by default; set 0 to A/B
#endif
#ifndef GETP_LOGITS_SWZ_G
// A group's X working set is G x BM x K x (4 or 2) bytes, and the best G is the one that keeps
// it inside the 8 MB L2. With fp32 X (1,47 MB/tile) the peak is at G=2; with bf16 X (737 KB/tile) it moves to G=6.
//   X fp32:  G=2 20.783 | G=3 21.071 | G=4 21.455 | G=6 22.467   (control 21.521)
//   X bf16:  G=0 18.125 | G=2 15.903 | G=3 15.487 | G=4 15.312 | G=6 15.178 | G=12 15.182
#define GETP_LOGITS_SWZ_G 6
#endif

// Convert the logits' X to bf16 once per step instead of reading fp32 and throwing half the
// bits away inside the kernel. See lgxbf16: it is what lets the group swizzle go further.
#ifndef GETP_LOGITS_XBF16
#define GETP_LOGITS_XBF16 1   // on by default; set 0 to A/B
#endif

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
// Derived from the tile: the kernel body uses TILE_N = 64 * (BM/WM)*(BN/WN) threads.
// Overriding with -D is only allowed if it still equals this expression (there is a static_assert).
#define MATMUL_LOGITS_WAVES_PER_BLOCK (((MATMUL_LOGITS_BLOCK_ROWS) / (MATMUL_LOGITS_WARP_TILE_M)) * ((MATMUL_LOGITS_BLOCK_COLS) / (MATMUL_LOGITS_WARP_TILE_N)))
#endif

// ---------------------------------------------------------------------------
// LDS geometry of the logits kernel (LM head).
//
// The tile is laid out as [row][k] - k CONSECUTIVE within a row - so 'padding' here pads ONE
// k ROW, NOT a k plane the way GETP_LDS_PAD does.
// The two quantities share a name but mean different things, have different validity conditions
// and different units, so this kernel needs its OWN pad macro. GETP_LDS_PAD keeps its old
// meaning and still serves the two remaining kernels on the [k][row] layout: qkv (:1392-1393)
// and attn_o (:3324-3325).
//
//   Row stride = (BK+PAD) shorts = 2*(BK+PAD) bytes = (BK+PAD)/2 banks.
//   One MFMA fragment = 4 shorts = 8 bytes = 2 banks = 1 "double-bank".
//   The 16 lanes of one ds_read_b64 phase move 16*8 = 128 bytes = exactly 32 banks.
//   Within a phase, L/16 is CONSTANT so basek is fixed and row = row0 + (L&15)
//   runs over 16 consecutive rows, i.e. db(i) = (S*i + c) mod 16 with S = (BK+PAD)/4.
//   Spread over all 32 banks <=> gcd(S,16) = 1 <=> S ODD.
//   BK a multiple of 16 (an existing static_assert) => BK/4 even => S = BK/4 + PAD/4 odd <=>
//   PAD/4 odd. So PAD = 4 works for EVERY valid BK (BK=16 -> S=5; 32 -> S=9;
//   64 -> S=17) and is the CHEAPEST value; 12/20/28 are also valid but only cost more
//   LDS without removing a single conflict.
//
// Example BK=32 (the flag in the winning build): KPAD = 36 shorts = 72 bytes
// = 18 banks = 9 double-banks. The FIRST bank of the 16 lanes in a phase (row0=0,
// basek=0) is 18*i mod 32 = 18,4,22,8,26,12,30,16,2,20,6,24,10,28,14,0 - pairwise
// distinct, and each lane occupies the bank pair {b, b+1} -> covering all 32 banks exactly once,
// with NO conflict. The WRITE path still has a 2-way conflict that no pad can fix:
// writing X needs {0,S} + {0..7} to cover Z16 => S = 8 (mod 16) => S EVEN; writing W needs
// {0,S,2S,3S} = {a,a+8,b,b+8} => 2S = 8 (mod 16) => S EVEN; both contradict
// the READ condition (S odd). Every PAD in 0..64 was scanned exhaustively.
#ifndef MATMUL_LOGITS_LDS_KPAD
#define MATMUL_LOGITS_LDS_KPAD 4
#endif

// The argmax epilogue reduces in registers before touching LDS (see GETP_LOGITS_EPI).
// 0 = the old path, bit-identical. Enable with -DGETP_LOGITS_EPI=1.
#ifndef GETP_LOGITS_EPI
#define GETP_LOGITS_EPI 1   // on by default (measured end to end: -0,35 ms/step); set 0 to A/B
#endif

// A single source of truth for the logits tile geometry.
constexpr int logits_lds_kpad(int BK, int PAD) { return BK + PAD; }
constexpr size_t logits_lds_shorts(int BM, int BN, int BK, int PAD) {
  // xsT[BM][KPAD] + wsT[BN][KPAD]
  return (size_t)(BM + BN) * (size_t)logits_lds_kpad(BK, PAD);
}

// 8 consecutive bf16 along k (one uint4 from HBM) -> 2 x ds_write_b64.
// Copy the dword VERBATIM: within a dword the low half is k+0 and the high half is k+1,
// exactly as before, so the result is bit-exact AND it costs no VALU instruction at all
// (no more & 0xFFFF / >> 16). The same pattern as flash attn's put16 (:2418).
__device__ __forceinline__ void logits_lds_store8(unsigned short* dst, const uint4 v) {
  *reinterpret_cast<uint2*>(dst + 0) = make_uint2(v.x, v.y);
  *reinterpret_cast<uint2*>(dst + 4) = make_uint2(v.z, v.w);
}

// 4 floats -> 4 bf16 consecutive along k -> 1 x ds_write_b64. f32_to_bf16bits still
// sees the same values in the same order; only the LDS address changes.
__device__ __forceinline__ void logits_lds_store4(unsigned short* dst, const float4 v) {
  const bf16x4 p = { f32_to_bf16bits(v.x), f32_to_bf16bits(v.y),
                     f32_to_bf16bits(v.z), f32_to_bf16bits(v.w) };
  *reinterpret_cast<bf16x4*>(dst) = p;
}

#if GETP_LOGITS_XBF16
// 4 fp32 -> 4 consecutive bf16. The same conversion helper the logits kernel still calls, so the bits match.
__global__ void logits_x_to_bf16_kernel(bf16x4* __restrict__ out,
                                        const float4* __restrict__ in, size_t n4) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  const float4 v = in[i];
  const bf16x4 p = { f32_to_bf16bits(v.x), f32_to_bf16bits(v.y),
                     f32_to_bf16bits(v.z), f32_to_bf16bits(v.w) };
  out[i] = p;
}
// 4 bf16 = 8 bytes = exactly one ds_write_b64, the same as the fp32 path but with no VALU instruction.
__device__ __forceinline__ void logits_lds_storeX(unsigned short* dst, const uint2 v) {
  *reinterpret_cast<uint2*>(dst) = v;
}
typedef uint2 getp_xload_t;
#else
__device__ __forceinline__ void logits_lds_storeX(unsigned short* dst, const float4 v) {
  logits_lds_store4(dst, v);
}
typedef float4 getp_xload_t;
#endif

template<
  int BM, int BN, int BK,
  int WM, int WN,
  int WARPS_PER_BLOCK,
  int THREADS = 64 * WARPS_PER_BLOCK
>
// __launch_bounds__ takes only ONE argument here. If a second is ever added, note that in HIP
// it is the MINIMUM number of waves per SIMD, not blocks per CU; the corresponding
// workgroups/CU figure is 4*CTA/WARPS_PER_BLOCK.
__global__ void __launch_bounds__(THREADS)
matmul_logits_argmax_bf16_kernel_tuned(
    const float* __restrict__ x,
    const unsigned short* __restrict__ xb,   // bf16 copy of x; only used under GETP_LOGITS_XBF16
    const __hip_bfloat16* __restrict__ w,
    int M, int N, int K,
    unsigned long long* __restrict__ pairs) {

  // Map block -> (row block, column block). The default keeps the old order.
  const int nrow_blk = (M + BM - 1) / BM;
#if GETP_LOGITS_SWZ
  // A one-dimensional grid, split again: within each group of GZ row blocks, the ROW block runs
  // fastest. GZ adjacent blocks read the SAME weight tile, so it comes from HBM once and then
  // stays in L2 for the whole group.
  constexpr int GZ = GETP_LOGITS_SWZ_G;
  static_assert(GZ >= 1, "G must be positive");
  const int ncol_blk = (N + BN - 1) / BN;
  const int perg = GZ * ncol_blk;
  const int gz = (int)blockIdx.x / perg;
  const int tz = (int)blockIdx.x - gz * perg;
  const int blk_row = gz * GZ + (tz % GZ);
  const int blk_col = tz / GZ;
  // Leftover blocks when nrow_blk is not a multiple of GZ. blk_row is block-uniform so the whole
  // block exits together, and this sits before any __syncthreads.
  if (blk_row >= nrow_blk) return;
#else
  const int blk_row = (int)blockIdx.y;
  const int blk_col = (int)blockIdx.x;
  (void)nrow_blk;
#endif

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
  // Wave -> (COLUMN tile, ROW tile). xInBlockTile indexes columns so the denominator must be
  // BN/WN. The old code divided by BM/WM: equal in the shipping configuration (2 == 2), so this
  // change is BIT-EXACT, but it was silently wrong - and read outside the LDS tile - for other
  // sweep lines bm256 / bn256 / wm32 / wn32 / bk32_wn32 in logits_cfg.txt.
  // The same CLASS OF BUG as qkv's :1371-1372.
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

  // LDS tile as [row][k]: the 4 k elements ONE lane needs for an mfma are consecutive
  // -> one ds_read_b64 instead of 4 ds_read_u16, and the 8 shorts loaded from HBM become
  // 2 ds_write_b64 instead of 8 ds_write_b16. Renaming xs/ws -> xsT/wsT is deliberate:
  // any site still reading the old [k][row] layout becomes a COMPILE ERROR, not a silently
  // wrong address. Loop indices stay 32-bit (LDS addresses are 32-bit).
  constexpr int PAD  = MATMUL_LOGITS_LDS_KPAD;
  constexpr int KPAD = logits_lds_kpad(BK, PAD);
  static_assert(KPAD % 4 == 0,
                "KPAD must be a multiple of 4 shorts: ds_read_b64/ds_write_b64 need 8-byte alignment.");
  static_assert((KPAD / 4) % 2 == 1,
                "(BK+PAD)/4 must be ODD for the 16 lanes of a b64 phase to spread over all 32 banks.");
  static_assert(logits_lds_shorts(BM, BN, BK, PAD) * sizeof(unsigned short)
                    + sizeof(float) * BM + sizeof(int) * BM <= 64 * 1024,
                "The LDS tile + smax/sidx exceeds 64 KB of LDS per CU.");

  __shared__ __align__(16) unsigned short xsT[BM][KPAD];
  __shared__ __align__(16) unsigned short wsT[BN][KPAD];
  __shared__ float smax[BM];
  __shared__ int sidx[BM];

#ifdef DEBUG_LDS_POISON
  // Launch check: the kernel body assumes exactly THREADS threads.
  if (blockDim.x != THREADS) {
    if (threadIdx.x == 0)
      printf("BAD LAUNCH: blockDim=%d but THREADS=%d\n", (int)blockDim.x, (int)THREADS);
    return;
  }
  // Fill the tile with bf16 qNaN: any slot the loader fails to write shows up as NaN.
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
    const int rg = BM * blk_row + ry;
    const int kk = 4 * loadXIdx_x;
#if GETP_LOGITS_XBF16
    getp_xload_t tmp = (kk < K && rg < M)
        ? *reinterpret_cast<const uint2 *>(&xb[(size_t)rg * K + kk]) : make_uint2(0, 0);
#else
    float4 tmp = (kk < K && rg < M) ? *reinterpret_cast<const float4 *>(&x[(size_t)rg * K + kk]) : make_float4(0,0,0,0);
#endif
    // address = 2*(ry*KPAD + 4*loadXIdx_x) bytes: KPAD a multiple of 4 and the chunk a multiple of 4
    // => a multiple of 8 => valid for ds_write_b64.
    logits_lds_storeX(&xsT[ry][4 * loadXIdx_x], tmp);
  }
  for (int i = 0; i < nbLoadsW; ++i) {
    int off = i * strideW;
    int kk = 0 + 8 * loadWIdx_x;
    int cy = BN * blk_col + loadWIdx_y + off;
    uint4 t = (kk < K && cy < N) ? *reinterpret_cast<const uint4 *>(&w[(size_t)cy * K + kk]) : make_uint4(0,0,0,0);
    logits_lds_store8(&wsT[loadWIdx_y + off][8 * loadWIdx_x], t);
  }
  __syncthreads();

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    getp_xload_t regX[nbLoadsX];
    uint4  regW[nbLoadsW];
    if (bkIdx < K - BK) {
      for (int i = 0; i < nbLoadsX; ++i) {
        int off = i * strideX;
        int ry = loadXIdx_y + off;
        int rg = BM * blk_row + ry;
        int kk = bkIdx + BK + 4 * loadXIdx_x;
#if GETP_LOGITS_XBF16
        regX[i] = (kk < K && rg < M)
            ? *reinterpret_cast<const uint2 *>(&xb[(size_t)rg * K + kk]) : make_uint2(0, 0);
#else
        regX[i] = (kk < K && rg < M) ? *reinterpret_cast<const float4 *>(&x[(size_t)rg * K + kk]) : make_float4(0,0,0,0);
#endif
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;
        int kk = bkIdx + BK + 8 * loadWIdx_x;
        int cy = BN * blk_col + loadWIdx_y + off;
        regW[i] = (kk < K && cy < N) ? *reinterpret_cast<const uint4 *>(&w[(size_t)cy * K + kk]) : make_uint4(0,0,0,0);
      }
    }
    for (int k = 0; k < BK; k += MFMA_K) {
      for (int im = 0; im < nIterWaveM; ++im) {
        for (int in = 0; in < nIterWaveN; ++in) {
          const int row = yInBlockTile * WM + MFMA_M * im + Ai;
          const int col = xInBlockTile * WN + MFMA_N * in + Bj;
          // Lane L needs A[i = L%16][k = 4*(L/16)+r], r = 0..3: FOUR consecutive k elements
          // of the SAME row -> one ds_read_b64 (the address is a multiple of 8 because KPAD is a
          // multiple of 4, k of 16 and Ak/Bk of 4). Afloat/Bfloat are 8-byte vectors, so this
          // cast is precisely what tells LLVM it may emit a single instruction.
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
        getp_xload_t v = regX[i];
        logits_lds_storeX(&xsT[ry][4 * loadXIdx_x], v);
      }
      for (int i = 0; i < nbLoadsW; ++i) {
        int off = i * strideW;
        logits_lds_store8(&wsT[loadWIdx_y + off][8 * loadWIdx_x], regW[i]);
      }
    }
    __syncthreads();
  }

#if GETP_LOGITS_EPI
  // ---- argmax epilogue: reduce in REGISTERS before touching LDS -----------------
  //
  // The row of an acc element,
  //     ly = WM*yInBlockTile + MFMA_M*im + 4*(laneIdx>>4) + i,
  // does NOT depend on `in` and does NOT depend on (laneIdx & 15). So the old version pushed the
  // 4 `in` values of one (im,i) into EXACTLY ONE smax[ly] slot, and all 16 lanes of the same
  // group (laneIdx>>4) pushed into that same slot: every ds_cmpst_rtn_b32 of the CAS loop inside
  // atomicMaxFloat (:4162) was a 16-way conflict on ONE address - LDS serialises it 16 times,
  // and the while loop spins again because the losers must retry. 64 dependent
  // read->CAS->branch chains per thread, sitting after the k loop where no MFMA can hide them.
  //
  // Reduce along exactly those two axes before touching LDS:
  //   (1) across `in`: register-only, 3 v_max_f32;
  //   (2) across the 16 lanes of a group: 4 x __shfl_xor with masks 1,2,4,8 - masks < 16, so the
  //       xor never leaves the 16-lane group and no width argument is needed (the same way
  //       flash attn uses at :2923).
  // Only lane (laneIdx&15)==0 writes LDS: 16 atomics per thread instead of 64, and each atomic is
  // now 4 lanes on 4 DIFFERENT addresses -> no conflict, and the CAS loop finishes on its first pass.
  //
  // Bit-exact: max/min are associative and commutative so the merge order does not change the
  // result; masked slots are replaced by the identity element (-INFINITY / INT_MAX) so they can
  // never win, exactly as the old version skipped them. No extra live registers: acc_local
  // already had to live across both phases in both versions, so 3 blocks/CU is unchanged.
  //
  // The only gap is the (+0,-0) pair: the CAS loop's `val > oldf` keeps whichever zero ARRIVES
  // FIRST, while v_max_f32 keeps the zero in a fixed position - and that difference is NOT
  // swallowed, because order_preserving_bits(+0)=0x80000000 differs from order_preserving_bits(-0)=0x7FFFFFFF.
  // This gap is unreachable: acc_local starts at +0.0 and MFMA accumulates in RN, and under RN
  // `a + b = -0.0` only when a = b = -0.0; starting from +0.0 that never
  // yields -0.0. So acc_local never holds -0.0.
  {
    const int lane16 = laneIdx & 15;
    const int gbase  = blk_col * BN + WN * xInBlockTile;
    #pragma unroll
    for (int im = 0; im < nIterWaveM; ++im) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int CDi = 4 * (laneIdx >> 4) + (i & 3);
        const int ly  = WM * yInBlockTile + MFMA_M * im + CDi;
        const int gy  = blk_row * BM + ly;
        float v = -INFINITY;
        #pragma unroll
        for (int in = 0; in < nIterWaveN; ++in) {
          const int gx = gbase + in * MFMA_N + lane16;
          const bool ok = (gx < N) && (gy < M);
          // fmaxf -> v_max_f32 (IEEE maxNum): a NaN operand loses to a non-NaN operand, exactly
          // as the old CAS loop's `val > oldf` never let a NaN win.
          v = fmaxf(v, ok ? acc_local[im * nIterWaveN + in][i] : -INFINITY);
        }
        // A butterfly within the 16-lane group: after 4 steps every lane holds the group max.
        v = fmaxf(v, __shfl_xor(v, 1));
        v = fmaxf(v, __shfl_xor(v, 2));
        v = fmaxf(v, __shfl_xor(v, 4));
        v = fmaxf(v, __shfl_xor(v, 8));
        // v == -INFINITY <=> no valid slot (or all of them are -INFINITY). In both
        // case the old atomicMaxFloat was also a no-op because `val > oldf` is false, so skipping
        // it is EQUIVALENT, not an approximation.
        if (lane16 == 0 && v > -INFINITY) atomicMaxFloat(&smax[ly], v);
      }
    }
  }
  __syncthreads();
  {
    const int lane16 = laneIdx & 15;
    const int gbase  = blk_col * BN + WN * xInBlockTile;
    #pragma unroll
    for (int im = 0; im < nIterWaveM; ++im) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int CDi = 4 * (laneIdx >> 4) + (i & 3);
        const int ly  = WM * yInBlockTile + MFMA_M * im + CDi;
        const int gy  = blk_row * BM + ly;
        // Hoist: smax[ly] does not depend on `in`, so 64 ds_read_b32 -> 16. The unconditional read
        // is safe (ly < BM always holds, smax is initialised to -INFINITY); when gy >= M
        // no lane contributes, so the value read back is never used.
        const float mv = smax[ly];
        int cand = INT_MAX;
        #pragma unroll
        for (int in = 0; in < nIterWaveN; ++in) {
          const int gx = gbase + in * MFMA_N + lane16;
          const float v = acc_local[im * nIterWaveN + in][i];
          const float thr = 1e-6f * fmaxf(fabsf(v), fabsf(mv));
          const bool hit = (gx < N) && (gy < M) && (fabsf(v - mv) <= thr);
          if (hit && gx < cand) cand = gx;
        }
        // The same butterfly with min. Split into a temporary so each mask costs one ds_bpermute.
        int o;
        o = __shfl_xor(cand, 1); cand = (o < cand) ? o : cand;
        o = __shfl_xor(cand, 2); cand = (o < cand) ? o : cand;
        o = __shfl_xor(cand, 4); cand = (o < cand) ? o : cand;
        o = __shfl_xor(cand, 8); cand = (o < cand) ? o : cand;
        // sidx starts at INT_MAX so atomicMin(INT_MAX) is a no-op: skipping it is equivalent.
        if (lane16 == 0 && cand != INT_MAX) atomicMin(&sidx[ly], cand);
      }
    }
  }
  __syncthreads();
#else
  for (int im = 0; im < nIterWaveM; ++im) {
    for (int in = 0; in < nIterWaveN; ++in) {
      int acc_index = im * nIterWaveN + in;
      for (int i = 0; i < 4; ++i) {
        int CDi = 4 * (laneIdx >> 4) + (i & 3);
        int CDj = laneIdx & 15;
        int gx = blk_col * BN + WN * xInBlockTile + in * MFMA_N + CDj;
        int ly = WM * yInBlockTile + MFMA_M * im + CDi;
        int gy = blk_row * BM + ly;
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
        int gx = blk_col * BN + WN * xInBlockTile + in * MFMA_N + CDj;
        int ly = WM * yInBlockTile + MFMA_M * im + CDi;
        int gy = blk_row * BM + ly;
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
#endif
  for (int r = threadIdx.x; r < BM; r += blockDim.x) {
    int gy = blk_row * BM + r;
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
#if GETP_LOGITS_SWZ
  // A one-dimensional grid; the kernel splits it back into (row, column). Rounding the row-block
  // count up to a multiple of G is what makes the split a bijection - the surplus blocks are empty and exit at once.
  const int nrow_up = (((M + BM - 1) / BM + GETP_LOGITS_SWZ_G - 1) / GETP_LOGITS_SWZ_G)
                      * GETP_LOGITS_SWZ_G;
  dim3 grid(((N + BN - 1) / BN) * nrow_up);
#else
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
#endif

  // hipFree synchronises the whole device, so allocating and freeing every step is one full
  // stall per step for a fixed-size 12 KB buffer.
  static thread_local unsigned long long *pairs = nullptr;
  static thread_local int pairs_cap = 0;
  if (pairs_cap < M) {
    if (pairs) HIP_CHECK(hipFree(pairs));
    HIP_CHECK(hipMalloc(&pairs, sizeof(unsigned long long) * M));
    pairs_cap = M;
  }
  dim3 gi((M + 255) / 256), bi(256);
  init_argmax_pairs<<<gi, bi, 0, stream>>>(pairs, M, N);

#if GETP_LOGITS_XBF16
  // Same reason as `pairs` above: hipFree synchronises the whole device, so this buffer is kept.
  static thread_local unsigned short *xb = nullptr;
  static thread_local size_t xb_cap = 0;
  const size_t xn = (size_t)M * (size_t)K;
  // xn = M*K is always a multiple of 4 because K (hidden_dim) is; no tail handling needed.
  if (xb_cap < xn) {
    if (xb) HIP_CHECK(hipFree(xb));
    HIP_CHECK(hipMalloc(&xb, sizeof(unsigned short) * xn));
    xb_cap = xn;
  }
  {
    const size_t n4 = xn / 4;
    dim3 bx(256), gx((unsigned)((n4 + 255) / 256));
    logits_x_to_bf16_kernel<<<gx, bx, 0, stream>>>(
        reinterpret_cast<bf16x4 *>(xb), reinterpret_cast<const float4 *>(x), n4);
  }
#else
  const unsigned short *xb = nullptr;
#endif

  matmul_logits_argmax_bf16_kernel_tuned<BM,BN,BK,WM,WN,WARPS>
      <<<grid, block, 0, stream>>>(x, xb, w, M, N, K, pairs);

  extract_argmax_pairs<<<gi, bi, 0, stream>>>(pairs, out, M);
  // (pairs is kept for the next step)

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

// On by default, but it only has an effect at EXPERT_PARALLELISM <= 2 - see the note in
// sync_workers_dev. Pass -DGETP_DEVSYNC=0 to go back to the old barrier in every configuration.
#ifndef GETP_DEVSYNC
#define GETP_DEVSYNC 1
#endif

#if GETP_DEVSYNC
// Each device's 'reached this point' event, one slot per (layer, phase). Other devices wait on
// it with hipStreamWaitEvent instead of having the host act as the queue's throttle.
#if GETP_AGSKIP
// All eight ranks are threads of one process, so this table goes through ordinary host memory.
// Written before the sp1 barrier and read after it => the barrier itself makes it visible, with no extra sync.
static __hip_bfloat16 *g_ag_send[MAXIMUM_GPU];   // [device] -> [EP][B][H] packed by destination
static int *g_ag_idx[MAXIMUM_GPU];               // [device] -> rr_idx [EP][B]
static int g_ag_cnt[MAXIMUM_GPU][MAXIMUM_GPU];   // [sending slice][receiving slice]
#endif

#ifndef GETP_XPULL
#define GETP_XPULL 1
#endif
#if GETP_XPULL
// Published once per device for the direct pulls: its row lists (rr_*) and 'my lists are built'
// events, one slot per layer.
static int *g_rr_idx[MAXIMUM_GPU], *g_rr_cnt[MAXIMUM_GPU];
static hipEvent_t g_ev_ag[MAXIMUM_GPU][128];
struct XPeers {
  const __hip_bfloat16 *t[MAXIMUM_GPU];   // peer's ext_t_bf16 [EP * B][H]
  const int *idx[MAXIMUM_GPU];            // peer's rr_idx [EP][B]: its rows destination sl needs
  const int *cnt[MAXIMUM_GPU];            // peer's rr_cnt [EP]
  const int *ti[MAXIMUM_GPU];             // peer's ext_topk_i [EP * B][K]
  const float *tv[MAXIMUM_GPU];           // peer's ext_topk_v
  const __hip_bfloat16 *e[MAXIMUM_GPU];   // peer's e_agg of this layer's parity, bf16 [EP][B][H]
};
// All-gather: block (slice j, i) copies row i of peer j's list for us from peer j's own slice into
// ours; blocks with i >= that count also copy a chunk of peer j's top-k (ids and weights).
__global__ __launch_bounds__(256) void xpull_ag_kernel(__hip_bfloat16 *__restrict__ t,
                                                       int *__restrict__ ti, float *__restrict__ tv,
                                                       XPeers pp, int own, int B, int H, int K) {
  const int j = blockIdx.x;
  if (j == own) return;
  const int i = blockIdx.y;
  // top-k of peer j's slice: B*K entries, 256 per block
  const int nk = B * K;
  for (int q = i * 256 + threadIdx.x; q < nk && i * 256 < nk; q += (int)gridDim.y * 256) {
    ti[(size_t)j * nk + q] = pp.ti[j][(size_t)j * nk + q];
    tv[(size_t)j * nk + q] = pp.tv[j][(size_t)j * nk + q];
  }
  if (i >= pp.cnt[j][own]) return;
  const int b = pp.idx[j][(size_t)own * B + i];
  const uint4 *src = reinterpret_cast<const uint4 *>(pp.t[j] + ((size_t)j * B + b) * H);
  uint4 *dst = reinterpret_cast<uint4 *>(t + ((size_t)j * B + b) * H);
  for (int k = threadIdx.x; k < (H >> 3); k += blockDim.x) dst[k] = src[k];
}
// Combine: thread (row b, 8 columns) adds each peer's partial sum of row b, in `order` (the old
// scatter-adds' delta order): x += p1; x += p2; ... - the same fp32 additions, bit-exact.
__global__ __launch_bounds__(128) void xpull_rs_kernel(float *__restrict__ x, XPeers pp,
                                                       const int *__restrict__ rank,
                                                       const int *__restrict__ order, int npeer,
                                                       int own, int B, int H) {
  const int b = blockIdx.y;
  const int c8 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c8 >= (H >> 3)) return;
  float4 *xr = reinterpret_cast<float4 *>(x + (size_t)b * H) + 2 * c8;
  float4 a = xr[0], c = xr[1];
  bool any = false;
  for (int d = 0; d < npeer; ++d) {
    const int ps = order[d];
    const int r = rank[(size_t)ps * B + b];
    if (r < 0) continue;
    any = true;
    const uint4 v = reinterpret_cast<const uint4 *>(pp.e[ps] + ((size_t)own * B + r) * H)[c8];
    a.x += bf16bits_to_f32((unsigned short)v.x); a.y += bf16bits_to_f32((unsigned short)(v.x >> 16));
    a.z += bf16bits_to_f32((unsigned short)v.y); a.w += bf16bits_to_f32((unsigned short)(v.y >> 16));
    c.x += bf16bits_to_f32((unsigned short)v.z); c.y += bf16bits_to_f32((unsigned short)(v.z >> 16));
    c.z += bf16bits_to_f32((unsigned short)v.w); c.w += bf16bits_to_f32((unsigned short)(v.w >> 16));
  }
  if (any) { xr[0] = a; xr[1] = c; }
}
#endif

#if GETP_EVAGG
// The 'my e_agg is written' event, one slot per layer. sp1 is still a barrier, so two ranks are
// never more than one layer apart and 128 slots is far more than enough.
static hipEvent_t g_ev_eagg[MAXIMUM_GPU][128];
#if GETP_GPSLICE
// Fires as soon as JUST the slice the first puller wants is done, without waiting for all eight.
static hipEvent_t g_ev_eagg_first[MAXIMUM_GPU][128];
#endif
#endif

#define GETP_SYNC_SLOTS 128
static hipEvent_t g_ev_sync[MAXIMUM_GPU][GETP_SYNC_SLOTS];
static bool g_ev_sync_ready[MAXIMUM_GPU] = {false};

// rec_stream: the stream that WROTE the data a peer is about to read. wait_stream: the stream
// that will READ the peer's data (here memory_stream, where the hipMemcpyPeerAsync calls run) -
// waiting on that exact stream is what makes it correct.
static inline void sync_workers_dev(hipStream_t rec_stream, hipStream_t wait_stream,
                                    Barrier &sync_point,
                                    int dev, int n_dev, int base_dev, int slot) {
  if (!g_ev_sync_ready[dev]) {
    for (int i = 0; i < GETP_SYNC_SLOTS; ++i)
      HIP_CHECK(hipEventCreateWithFlags(&g_ev_sync[dev][i], hipEventDisableTiming));
    g_ev_sync_ready[dev] = true;
    HIP_CHECK(hipStreamSynchronize(rec_stream));
    sync_point.wait();          // every device has created its events
  }
  // Only take the event path when there are FEW peers. Measured on 8 GPUs over 1024 steps: at
  // EXPERT_PARALLELISM = 2 (20b) it is 0,7 % faster (63.788 -> 64.361); at EXPERT_PARALLELISM =
  // 8 (120b) it is 15 % SLOWER (20.003 -> 16.931), because each sync point queues 7
  // hipStreamWaitEvent on memory_stream and, with the host no longer throttling, that queue
  // grows deeper and deeper. With many peers the throttle is the cheaper option.
  if (n_dev > 2) {
    HIP_CHECK(hipStreamSynchronize(rec_stream));
    sync_point.wait();
    return;
  }
  const int k = slot % GETP_SYNC_SLOTS;
  HIP_CHECK(hipEventRecord(g_ev_sync[dev][k], rec_stream));
  sync_point.wait();            // a HOST-only sync: every thread has issued its event
  for (int i = 0; i < n_dev; ++i) {
    const int pd = base_dev + i;
    if (pd != dev) HIP_CHECK(hipStreamWaitEvent(wait_stream, g_ev_sync[pd][k], 0));
  }
}
#endif

// GETP_EPLB (default 1): move a few experts between GPUs once, early in the run, so that the
// per-layer MoE load evens out. See the note at its call site in getp_forward_120b.
#ifndef GETP_EPLB
#define GETP_EPLB 1
#endif
#ifndef GETP_EPLB_AT
#define GETP_EPLB_AT 40      // the first move happens at the start of this step
#endif
#ifndef GETP_EPLB_PERIOD
#define GETP_EPLB_PERIOD 100 // and again every PERIOD steps (0 = only once): popularity drifts
#endif
#ifndef GETP_EPLB_WINDOW
#define GETP_EPLB_WINDOW 32  // each move is planned from the tiles of the WINDOW steps before it
#endif
// The smallest EXPERT_PARALLELISM that balances (2: the 20B's pairs too; 3: only the 120B).
#ifndef GETP_EPLB_MIN_EP
#define GETP_EPLB_MIN_EP 2
#endif
// Finished sequences (mask_on[b] == 0) are routed to no expert. The batch runs until its longest
// sequence ends and attention already skipped the finished rows, but the MoE did not: 5.9 % of the
// 20B's row-steps on the 8-GPU benchmark input, 3 % of the 120B's. Each MoE output row depends only
// on its own pairs and the grids are sized from host constants, so with a fixed expert placement the
// live rows are bit-exact. GETP_EPLB plans its moves from pair counts, which no longer include the
// finished rows, so its moves - and the grouping of a token's partial sums - change with this.
#ifndef GETP_DEAD_ROWS
#define GETP_DEAD_ROWS 1
#endif
#ifndef GETP_EPLB_MOVES
#define GETP_EPLB_MOVES 8    // experts allowed to change GPU, per layer and per move
#endif
#if GETP_EPLB
#define GETP_EPLB_MAXL 64
#define GETP_EPLB_MAXE 128
// Each rank's counted tiles, [layer][its local label], published for the others at the plan.
static int g_eplb_tiles[MAXIMUM_GPU][GETP_EPLB_MAXL * GETP_EPLB_MAXE];
static int g_eplb_ok[MAXIMUM_GPU];
static int g_eplb_step[MAXIMUM_GPU];
static int *g_eplb_acc[MAXIMUM_GPU];
static int *g_eplb_relabel[MAXIMUM_GPU];
static std::vector<int> g_eplb_label[MAXIMUM_GPU];
static __hip_bfloat16 *g_eplb_stage[MAXIMUM_GPU];
static int g_eplb_stage_cap[MAXIMUM_GPU];

__global__ void eplb_count_kernel(int *acc, const int *e_counts, int n, int bm) {
  const int i = threadIdx.x;
  if (i < n) acc[i] += (e_counts[i] + bm - 1) / bm;
}

// owner[l*E + label] = rank that holds that (current) label after the plan. Pure integer greedy
// with fixed tie-breaks, so every rank that runs it on the same table gets the same answer.
static void eplb_plan(const std::vector<int> &tiles, int L, int E, int EP, int moves,
                      std::vector<int> &owner) {
  const int epd = E / EP;
  owner.assign((size_t)L * E, 0);
  for (int l = 0; l < L; ++l) {
    const int *t = tiles.data() + (size_t)l * E;
    int *own = owner.data() + (size_t)l * E;
    long long load[MAXIMUM_GPU] = {};
    for (int e = 0; e < E; ++e) { own[e] = e / epd; load[own[e]] += t[e]; }
    for (int moved = 0; moved + 2 <= moves;) {
      int hi = 0, lo = 0;
      for (int r = 1; r < EP; ++r) {
        if (load[r] > load[hi]) hi = r;
        if (load[r] < load[lo]) lo = r;
      }
      long long best = load[hi];
      int ba = -1, bb = -1;
      for (int a = 0; a < E; ++a) {
        if (own[a] != hi) continue;
        for (int b = 0; b < E; ++b) {
          if (own[b] != lo) continue;
          const long long d = (long long)t[a] - t[b];
          if (d <= 0) continue;
          const long long m = std::max(load[hi] - d, load[lo] + d);
          if (m < best) { best = m; ba = a; bb = b; }
        }
      }
      if (ba < 0) break;
      const long long d = (long long)t[ba] - t[bb];
      own[ba] = lo; own[bb] = hi; load[hi] -= d; load[lo] += d;
      moved += 2;
    }
  }
}
#endif

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

#ifndef GETP_COPY_STREAMS
// Streams the 120B's peer pulls are spread over (1 = memory_stream only). Measured on top of
// GETP_EPLB, all bit-identical: 1 stream 28.4-28.7k tok/s; 3, 4, 5, 7 streams 29.8-30.5k. 4, not 7:
// with GPU_MAX_HW_QUEUES raised to 16, 7 streams collapsed to 14.2k while 4 still gave 29.8k.
#define GETP_COPY_STREAMS 4
#endif

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

  // These 5 events used to be created EVERY STEP and never destroyed. Now they are created once
  // per thread (one thread per GPUWorker, one device per thread).
  static thread_local hipEvent_t event_rmsnorm = nullptr;
  static thread_local hipEvent_t event_router_topk = nullptr;
  static thread_local hipEvent_t events_e_agg[MAXIMUM_GPU] = {};
  static thread_local hipEvent_t events_pull[MAXIMUM_GPU] = {};
  if (!event_rmsnorm) {
    HIP_CHECK(hipEventCreateWithFlags(&event_rmsnorm, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&event_router_topk, hipEventDisableTiming));
    for (int i = 0; i < EXPERT_PARALLELISM; ++i) {
      HIP_CHECK(hipEventCreateWithFlags(&events_e_agg[i], hipEventDisableTiming));
      HIP_CHECK(hipEventCreateWithFlags(&events_pull[i], hipEventDisableTiming));
    }
  }

#if GETP_ROWSKIP
  // Only on at EP > 2 (see point 2 at the head of the file). Small buffers: 3 arrays of EP*B int, ~150 KB.
  const bool rs_on = (EXPERT_PARALLELISM > 2);
  static thread_local int *rs_rank = nullptr, *rs_idx = nullptr, *rs_cnt = nullptr;
  static thread_local int *rr_idx = nullptr, *rr_rank = nullptr, *rr_cnt = nullptr;
  static thread_local int *rs_hcnt = nullptr;
#if GETP_AGSKIP
  const bool ag_on = rs_on;
  static thread_local __hip_bfloat16 *ag_send = nullptr, *ag_recv = nullptr;
  static thread_local int *ag_ridx = nullptr;
#endif
#if GETP_RS_MERGE
  // Each slice's expert range; fixed for the whole run, so upload it once.
  static thread_local int *rs_elo = nullptr, *rs_ehi = nullptr;
  if (rs_on && rs_elo == nullptr) {
    int hlo[MAXIMUM_GPU], hhi[MAXIMUM_GPU];
    for (int q = 0; q < EXPERT_PARALLELISM; ++q) {
      hlo[q] = workers[q].expert_start; hhi[q] = workers[q].expert_end;
    }
    HIP_CHECK(hipMalloc(&rs_elo, sizeof(int) * EXPERT_PARALLELISM));
    HIP_CHECK(hipMalloc(&rs_ehi, sizeof(int) * EXPERT_PARALLELISM));
    HIP_CHECK(hipMemcpy(rs_elo, hlo, sizeof(int) * EXPERT_PARALLELISM, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(rs_ehi, hhi, sizeof(int) * EXPERT_PARALLELISM, hipMemcpyHostToDevice));
  }
#endif
  if (rs_on && rs_rank == nullptr) {
    const size_t n = (size_t)EXPERT_PARALLELISM * BATCH_SIZE;
    HIP_CHECK(hipMalloc(&rs_rank, sizeof(int) * n));
    HIP_CHECK(hipMalloc(&rs_idx,  sizeof(int) * n));
    HIP_CHECK(hipMalloc(&rr_idx,  sizeof(int) * n));
    HIP_CHECK(hipMalloc(&rr_rank, sizeof(int) * n));
    HIP_CHECK(hipMalloc(&rs_cnt,  sizeof(int) * EXPERT_PARALLELISM));
    HIP_CHECK(hipMalloc(&rr_cnt,  sizeof(int) * EXPERT_PARALLELISM));
    HIP_CHECK(hipHostMalloc(&rs_hcnt, sizeof(int) * EXPERT_PARALLELISM, hipHostMallocDefault));
#if GETP_AGSKIP
    // Publish for the other threads. Reads only happen after the sp1 barrier, so this is safe.
    HIP_CHECK(hipMalloc(&ag_send, sizeof(__hip_bfloat16) * (size_t)EXPERT_PARALLELISM *
                                      BATCH_SIZE * hidden_dim));
    HIP_CHECK(hipMalloc(&ag_recv, sizeof(__hip_bfloat16) * (size_t)EXPERT_PARALLELISM *
                                      BATCH_SIZE * hidden_dim));
    HIP_CHECK(hipMalloc(&ag_ridx, sizeof(int) * (size_t)EXPERT_PARALLELISM * BATCH_SIZE));
    g_ag_send[device_index] = ag_send;
    g_ag_idx[device_index] = rr_idx;
#if GETP_XPULL
    g_rr_idx[device_index] = rr_idx;
    g_rr_cnt[device_index] = rr_cnt;
    for (int q = 0; q < 128; ++q)
      HIP_CHECK(hipEventCreateWithFlags(&g_ev_ag[device_index][q], hipEventDisableTiming));
    // peer access to every other GPU of the group: the pulls read their memory directly
    for (int q = 0; q < EXPERT_PARALLELISM; ++q) {
      const int pd = workers[q].device_index;
      if (pd == device_index) continue;
      const hipError_t e = hipDeviceEnablePeerAccess(pd, 0);
      if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
        fprintf(stderr, "xpull: no peer access %d -> %d\n", device_index, pd);
        abort();
      }
      (void)hipGetLastError();
    }
#endif
#if GETP_EVAGG
    for (int q = 0; q < 128; ++q)
      HIP_CHECK(hipEventCreateWithFlags(&g_ev_eagg[device_index][q], hipEventDisableTiming));
#if GETP_GPSLICE
    for (int q = 0; q < 128; ++q)
      HIP_CHECK(hipEventCreateWithFlags(&g_ev_eagg_first[device_index][q], hipEventDisableTiming));
#endif
#endif
#endif
  }
#endif
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
  // Copy streams for the peer pulls: [0] is memory_stream, the rest are created once per thread,
  // and only at EP > 2 - the 20B has one peer, so extra streams would only be idle queue users.
  const int n_copy_streams = (EXPERT_PARALLELISM > 2) ? GETP_COPY_STREAMS : 1;
  static thread_local hipStream_t copy_streams[GETP_COPY_STREAMS] = {};
  if (copy_streams[0] == nullptr) {
    copy_streams[0] = memory_stream;
    for (int c = 1; c < n_copy_streams; ++c)
      HIP_CHECK(hipStreamCreateWithFlags(&copy_streams[c], hipStreamNonBlocking));
  }
#define GETP_CS(j_) (copy_streams[(j_) % n_copy_streams])

#if GETP_EPLB
  // Expert load balancing. The router writes LABELS (relabel[l][expert]); rank r owns labels
  // [epd*r, epd*r + epd) and label epd*r + i is its weight slot i, so everything downstream keeps
  // working on contiguous ranges. The WINDOW steps before each move count each slot's padded tiles; at
  // the start of a move step every rank agrees on the same swaps (eplb_plan), pulls the experts it receives into a
  // staging buffer, waits until every rank has done the same, copies them into the vacated slots,
  // and installs the new labels. Only at EP > 2 (rs_on); the table stays the identity otherwise.
  const int eplb_epd = worker->expert_end - worker->expert_start;
  // Per worker SLOT, not per thread: the 120B starts fresh threads for every wave of requests,
  // and a new thread must keep using the labels that match the weights already moved.
  int &eplb_step = g_eplb_step[device_index];
  int *&eplb_acc = g_eplb_acc[device_index];            // device [L][epd] tiles counted
  int *&eplb_relabel = g_eplb_relabel[device_index];    // device [L][E] expert -> label
  std::vector<int> &eplb_label = g_eplb_label[device_index];  // host copy of eplb_relabel
  const bool eplb_on = (EXPERT_PARALLELISM >= GETP_EPLB_MIN_EP) && p->n_layers <= GETP_EPLB_MAXL &&
                       n_experts <= GETP_EPLB_MAXE;
  if (eplb_on && eplb_relabel == nullptr) {
    eplb_label.resize((size_t)p->n_layers * n_experts);
    for (int l = 0; l < p->n_layers; ++l)
      for (int e = 0; e < n_experts; ++e) eplb_label[(size_t)l * n_experts + e] = e;
    // Every transfer to or from these tables goes on compute_stream, the stream whose kernels
    // read and write them, and is waited for: the null stream is NOT ordered against our
    // non-blocking streams, so a plain hipMemcpy/hipMemset here could land after the router.
    HIP_CHECK(hipMalloc(&eplb_relabel, sizeof(int) * eplb_label.size()));
    HIP_CHECK(hipMemcpyAsync(eplb_relabel, eplb_label.data(), sizeof(int) * eplb_label.size(),
                             hipMemcpyHostToDevice, compute_stream));
    HIP_CHECK(hipMalloc(&eplb_acc, sizeof(int) * (size_t)p->n_layers * eplb_epd));
    HIP_CHECK(hipMemsetAsync(eplb_acc, 0, sizeof(int) * (size_t)p->n_layers * eplb_epd,
                             compute_stream));
    HIP_CHECK(hipStreamSynchronize(compute_stream));
  }
  const bool eplb_move_now =
      eplb_on && eplb_step >= GETP_EPLB_AT &&
      (GETP_EPLB_PERIOD > 0 ? (eplb_step - GETP_EPLB_AT) % GETP_EPLB_PERIOD == 0
                            : eplb_step == GETP_EPLB_AT);
  if (eplb_move_now) {
    const int L = p->n_layers, E = n_experts, EP = EXPERT_PARALLELISM, epd = eplb_epd;
    const auto t0 = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpyAsync(g_eplb_tiles[device_index], eplb_acc, sizeof(int) * (size_t)L * epd,
                             hipMemcpyDeviceToHost, compute_stream));
    HIP_CHECK(hipStreamSynchronize(compute_stream));
    sync_point.wait();
    std::vector<int> tl((size_t)L * E), owner;
    for (int q = 0; q < EP; ++q)
      for (int l = 0; l < L; ++l)
        for (int i = 0; i < epd; ++i)
          tl[(size_t)l * E + q * epd + i] = g_eplb_tiles[base_device_index + q][(size_t)l * epd + i];
    eplb_plan(tl, L, E, EP, GETP_EPLB_MOVES, owner);
    // New slot of every current label: labels that stay keep their slot; each rank's incoming
    // labels, in increasing order, fill its vacated slots in increasing order.
    std::vector<int> newlab((size_t)L * E);
    std::vector<std::vector<std::pair<int, int>>> incoming(L);  // mine: (my slot, source label)
    int max_in = 0, moved = 0;
    for (int l = 0; l < L; ++l) {
      const int *own = owner.data() + (size_t)l * E;
      for (int r = 0; r < EP; ++r) {
        std::vector<int> vac, in;
        for (int i = 0; i < epd; ++i) {
          const int lab = r * epd + i;
          if (own[lab] == r) newlab[(size_t)l * E + lab] = lab; else vac.push_back(i);
        }
        for (int lab = 0; lab < E; ++lab)
          if (own[lab] == r && lab / epd != r) in.push_back(lab);
        for (size_t k = 0; k < in.size(); ++k) {
          newlab[(size_t)l * E + in[k]] = r * epd + vac[k];
          if (r == own_slice) incoming[l].push_back({vac[k], in[k]});
        }
        moved += (int)in.size();
      }
      max_in = std::max(max_in, (int)incoming[l].size());
    }
    const int H = hidden_dim, I = p->intermediate_dim;
    const size_t n_w1 = (size_t)2 * I * H, n_b1 = (size_t)2 * I, n_w2 = (size_t)H * I, n_b2 = H;
    const size_t n_ex = n_w1 + n_b1 + n_w2 + n_b2;
    // Staging is kept for the whole run and only ever grown, so no move frees device memory
    // while the engine is running.
    __hip_bfloat16 *&stage = g_eplb_stage[device_index];
    int &stage_cap = g_eplb_stage_cap[device_index];
    bool ok = true;
    if (max_in > stage_cap) {
      if (stage) HIP_CHECK(hipFree(stage));
      stage = nullptr; stage_cap = 0;
      ok = hipMalloc(&stage, sizeof(__hip_bfloat16) * n_ex * max_in) == hipSuccess;
      if (ok) stage_cap = max_in; else stage = nullptr;
    }
    g_eplb_ok[device_index] = ok;
    sync_point.wait();
    bool all_ok = true;
    for (int q = 0; q < EP; ++q) all_ok = all_ok && g_eplb_ok[base_device_index + q];
    if (all_ok && moved > 0) {
      for (int l = 0; l < L; ++l) {
        for (size_t k = 0; k < incoming[l].size(); ++k) {
          const int src_lab = incoming[l][k].second, q = src_lab / epd, j = src_lab % epd;
          const DeviceTransformerWeights *sw = &dev_transformers[base_device_index + q]->weights;
          const int sdev = workers[q].device_index;
          __hip_bfloat16 *st = stage + k * n_ex;
          const size_t e = (size_t)l * epd + j;
          HIP_CHECK(hipMemcpyPeerAsync(st, device_index, sw->w_mlp1 + e * n_w1, sdev,
                                       sizeof(__hip_bfloat16) * n_w1, memory_stream));
          HIP_CHECK(hipMemcpyPeerAsync(st + n_w1, device_index, sw->b_mlp1 + e * n_b1, sdev,
                                       sizeof(__hip_bfloat16) * n_b1, memory_stream));
          HIP_CHECK(hipMemcpyPeerAsync(st + n_w1 + n_b1, device_index, sw->w_mlp2 + e * n_w2, sdev,
                                       sizeof(__hip_bfloat16) * n_w2, memory_stream));
          HIP_CHECK(hipMemcpyPeerAsync(st + n_w1 + n_b1 + n_w2, device_index, sw->b_mlp2 + e * n_b2,
                                       sdev, sizeof(__hip_bfloat16) * n_b2, memory_stream));
        }
        HIP_CHECK(hipStreamSynchronize(memory_stream));
        sync_point.wait();   // every rank has READ its sources for this layer: slots may change now
        for (size_t k = 0; k < incoming[l].size(); ++k) {
          const size_t e = (size_t)l * epd + incoming[l][k].first;
          const __hip_bfloat16 *st = stage + k * n_ex;
          HIP_CHECK(hipMemcpyAsync(dev_w->w_mlp1 + e * n_w1, st, sizeof(__hip_bfloat16) * n_w1,
                                   hipMemcpyDeviceToDevice, memory_stream));
          HIP_CHECK(hipMemcpyAsync(dev_w->b_mlp1 + e * n_b1, st + n_w1, sizeof(__hip_bfloat16) * n_b1,
                                   hipMemcpyDeviceToDevice, memory_stream));
          HIP_CHECK(hipMemcpyAsync(dev_w->w_mlp2 + e * n_w2, st + n_w1 + n_b1,
                                   sizeof(__hip_bfloat16) * n_w2, hipMemcpyDeviceToDevice,
                                   memory_stream));
          HIP_CHECK(hipMemcpyAsync(dev_w->b_mlp2 + e * n_b2, st + n_w1 + n_b1 + n_w2,
                                   sizeof(__hip_bfloat16) * n_b2, hipMemcpyDeviceToDevice,
                                   memory_stream));
        }
        HIP_CHECK(hipStreamSynchronize(memory_stream));
      }
      for (size_t x = 0; x < eplb_label.size(); ++x) {
        const int l = (int)(x / E);
        eplb_label[x] = newlab[(size_t)l * E + eplb_label[x]];
      }
      HIP_CHECK(hipMemcpyAsync(eplb_relabel, eplb_label.data(), sizeof(int) * eplb_label.size(),
                               hipMemcpyHostToDevice, compute_stream));
    }
    HIP_CHECK(hipMemsetAsync(eplb_acc, 0, sizeof(int) * (size_t)L * epd, compute_stream));
    HIP_CHECK(hipStreamSynchronize(compute_stream));   // new labels and an empty window, in place
    sync_point.wait();
    // One line per move with GETP_EPLB_LOG set; a move that could not be applied is always reported.
    static const bool eplb_log = getenv("GETP_EPLB_LOG") != nullptr;
    if (own_slice == 0 && (eplb_log || !all_ok))
      fprintf(stderr, "eplb: step %d, %d of %d experts moved%s, %.0f ms\n", eplb_step, moved,
              L * E, all_ok ? "" : " (NOT applied: staging allocation failed)",
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  // Count during the WINDOW steps before each move: [AT + k*PERIOD - WINDOW, AT + k*PERIOD).
  const int eplb_rel = eplb_step - (GETP_EPLB_AT - GETP_EPLB_WINDOW);
  const bool eplb_count =
      eplb_on && eplb_rel >= 0 &&
      (GETP_EPLB_PERIOD > 0 ? eplb_rel % GETP_EPLB_PERIOD < GETP_EPLB_WINDOW
                            : eplb_rel < GETP_EPLB_WINDOW);
  if (eplb_on) ++eplb_step;
#endif

  for (unsigned long long l = 0; l < p->n_layers; l++) {
    // rmsnorm writes bf16 DIRECTLY into pre_qkv_bf16: the fp32 form (dev_s->t) no longer
    // has any consumer on the 120b path.
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
    // Layers are laid out even ones first, then odd ones (layer_toff); each also carries its padding.
    const size_t base_off = layer_toff * (size_t)BATCH_SIZE * (size_t)kv_dim +
                            (size_t)(is_even ? idx_half : even_layers + idx_half) *
                                GETP_KV_LAYER_PAD(BATCH_SIZE, kv_dim);

    const size_t slot_off = base_off + GETP_KV_OFF(tslot, 0, 0, BATCH_SIZE, kv_dim, cache_tcap);
    __hip_bfloat16 *k_slot = GETP_KV_PTR_K(dev_s->key_cache, slot_off);
    __hip_bfloat16 *v_slot = GETP_KV_PTR_V(dev_s->value_cache, slot_off);
    float *k_slot_scale = nullptr, *v_slot_scale = nullptr;
    const float *k_layer_scale = nullptr, *v_layer_scale = nullptr;
#if GETP_KV8_K
    k_slot_scale = getp_kv8_scales(dev_s->key_cache, p, BATCH_SIZE) + GETP_KV_SIDX(slot_off);
    k_layer_scale = getp_kv8_scales(dev_s->key_cache, p, BATCH_SIZE) + GETP_KV_SIDX(base_off);
#endif
#if GETP_KV8_V
    v_slot_scale = getp_kv8_scales(dev_s->value_cache, p, BATCH_SIZE) + GETP_KV_SIDX(slot_off);
    v_layer_scale = getp_kv8_scales(dev_s->value_cache, p, BATCH_SIZE) + GETP_KV_SIDX(base_off);
#endif

    float *k_step = dev_s->tb2;
    float *v_step = dev_s->tb2 + (size_t)BATCH_SIZE * (size_t)kv_dim;

    __hip_bfloat16 *dev_w_qkv =
        dev_w->w_qkv_bf16 +
        1ll * l * hidden_dim *
            (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    __hip_bfloat16 *dev_b_qkv =
        dev_w->b_qkv_bf16 +
        1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
#if GETP_ROPE_FUSED
    // RoPE now lives in the qkv epilogue (and with GETP_ROPE_FUSED_KV so does the
    // bf16 conversion into the cache slot). dev_cos_vals/dev_sin_vals are generated
    // before the layer loop on the SAME stream, so the ordering still holds; the live range
    // of the two tables is unchanged - apply_rotary_emb used to read them at EXACTLY
    // this point, right after qkv and before the layer's MoE.
    getp_matmul_qkv_fused_bf16(dev_s->q, k_step, v_step, ext->pre_qkv_bf16,
                               dev_w_qkv, dev_b_qkv, hidden_dim, head_dim,
                               p->n_attn_heads, p->n_kv_heads, BATCH_SIZE,
                               compute_stream, dev_cos_vals, dev_sin_vals,
                               k_slot, v_slot, cache_tcap, k_slot_scale, v_slot_scale);
#if !GETP_ROPE_FUSED_KV
    kv_store_pair_fp32_to_bf16(k_slot, v_slot, k_step, v_step,
                               BATCH_SIZE * kv_dim, compute_stream);
#endif
#else
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
#endif

    {
      const __hip_bfloat16 *k_layer = GETP_KV_PTR_K(dev_s->key_cache, base_off);
      const __hip_bfloat16 *v_layer = GETP_KV_PTR_V(dev_s->value_cache, base_off);
      const float *q_ptr = dev_s->q;
      const float *attn_sinks_layer =
          dev_w->attn_sinks + (size_t)l * p->n_attn_heads;
      getp_flash_attn_decode_bf16(
          ext->attn_o_bf16, dev_s->tb,
          k_layer, v_layer, q_ptr, ext->mask_on, attn_sinks_layer, head_dim,
          p->n_attn_heads, p->n_kv_heads, pos, p->seq_len, p->sliding_window,
          (int)l, BATCH_SIZE, cache_tcap, compute_stream, k_layer_scale, v_layer_scale);
    }

    __hip_bfloat16 *dev_w_o =
        dev_w->w_o_bf16 + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    __hip_bfloat16 *dev_b_o = dev_w->b_o_bf16 + 1ll * l * hidden_dim;
    // attention has already written bf16 into attn_o_bf16; the matmul adds straight into dev_x.
    getp_matmul_attn_o_bf16_acc(dev_x, ext->attn_o_bf16, dev_w_o, dev_b_o,
                                head_dim * p->n_attn_heads, hidden_dim,
                                BATCH_SIZE, compute_stream);

#if GETP_ROUTER_FUSED
    // The fused router reads ext_t_bf16, so nothing reads the fp32 ext_t any more.
    const bool router_fused = getp_router_fused_ok(BATCH_SIZE, n_experts, hidden_dim);
#else
    const bool router_fused = false;
#endif
    // The router reads ext_t fp32 (unless fused) and the peer copy reads ext_t_bf16: produce both
    // in one pass, instead of a separate conversion kernel on memory_stream.
    if (router_fused)
      getp_rmsnorm_bf16(ext->ext_t_bf16 + (size_t)own_slice * BATCH_SIZE * hidden_dim,
                        dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim,
                        BATCH_SIZE, hidden_dim, compute_stream);
    else
    getp_rmsnorm_f32_bf16(
        ext->ext_t + (size_t)own_slice * BATCH_SIZE * hidden_dim,
        ext->ext_t_bf16 + (size_t)own_slice * BATCH_SIZE * hidden_dim,
        dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim,
        BATCH_SIZE, hidden_dim, compute_stream);
    HIP_CHECK(hipEventRecord(event_rmsnorm, compute_stream));

    __hip_bfloat16 *dev_w_router =
        dev_w->w_router_bf16 + 1ll * l * hidden_dim * n_experts;
    __hip_bfloat16 *dev_b_router = dev_w->b_router_bf16 + 1ll * l * n_experts;
#if GETP_ROUTER_FUSED
    if (router_fused)
      getp_matmul_router_fused(dev_s->router_score,
                               ext->ext_t_bf16 + (size_t)own_slice * BATCH_SIZE * hidden_dim,
                               dev_w_router, dev_b_router, hidden_dim, n_experts, BATCH_SIZE,
                               compute_stream);
    else
#endif
    getp_matmul_router_bf16(
      dev_s->router_score,
      ext->ext_t + (size_t)(device_index - base_device_index) * BATCH_SIZE * hidden_dim,
      dev_w_router, dev_b_router, hidden_dim, n_experts,
      BATCH_SIZE, compute_stream);

    getp_router_topk_softmax_batch(
        dev_s->router_score, n_experts, p->experts_per_token,
        ext->ext_topk_v + (size_t)(device_index - base_device_index) * BATCH_SIZE * p->experts_per_token,
        ext->ext_topk_i + (size_t)(device_index - base_device_index) * BATCH_SIZE * p->experts_per_token,
        BATCH_SIZE, compute_stream
#if GETP_EPLB
        , eplb_on ? eplb_relabel + (size_t)l * n_experts : nullptr
#else
        , nullptr
#endif
#if GETP_DEAD_ROWS
        , ext->mask_on
#endif
        );
    HIP_CHECK(hipEventRecord(event_router_topk, compute_stream));
#if GETP_AGSKIP
    // This list only reads the top-k of OUR OWN SLICE, available since :5542 - it waits for no peer.
    // For destination slice sl it lists the rows of ours that sl owns an expert for: exactly what sl will pull.
    if (ag_on) {
      getp_rowskip_build_all_recv(ext->ext_topk_i, p->experts_per_token, BATCH_SIZE,
                                  EXPERT_PARALLELISM, own_slice, own_slice,
                                  rs_elo, rs_ehi, rr_rank, rr_idx, rr_cnt, compute_stream);
#if GETP_XPULL
      // the peers read our rows in place (xpull_ag_kernel): no packing
      HIP_CHECK(hipEventRecord(g_ev_ag[device_index][l % 128], compute_stream));
#else
      getp_agskip_gather_kernel<<<dim3((unsigned)EXPERT_PARALLELISM, (unsigned)BATCH_SIZE), 256,
                                  0, compute_stream>>>(
          ext->ext_t_bf16 + (size_t)own_slice * BATCH_SIZE * hidden_dim, ag_send, rr_idx, rr_cnt,
          own_slice, BATCH_SIZE, hidden_dim);
#endif
    }
#endif

    // ext_t_bf16 was written by rmsnorm on compute_stream before event_rmsnorm.
    // The hidden-state and top-k exchange is a PULL after the barrier, not a PUSH before it.
    // The push (a peer copy into our buffer on the PEER's memory_stream, then a host barrier)
    // left a rare window in which we read the previous step's ext_t_bf16/topk: measured by
    // hashing 11 points per layer across 2 eight-GPU runs, about 4 events per run, always on a
    // matching (step, layer) pair of EP ranks, first diverging at a_in after the scatter while
    // that rank's own topk and attention still matched. Acquiring on the receiving side and
    // fine-grained memory both failed to fix it. Pulling on the receiving device's own queue
    // plus an event is the pattern the e_agg path already used, and it measures clean.
#if GETP_AGSKIP && GETP_XPULL
    if (ag_on) {
      // No drain: every rank has recorded its 'lists built' event once past this host barrier;
      // then one kernel pulls everything this rank needs from the seven peers.
      sync_point.wait();
      XPeers pp = {};
      for (int q = 0; q < EXPERT_PARALLELISM; ++q) {
        const int pd = base_device_index + q;
        if (q == own_slice) continue;
        RunStateExt *pe = ext_get(pd);
        pp.t[q] = pe->ext_t_bf16; pp.idx[q] = g_rr_idx[pd]; pp.cnt[q] = g_rr_cnt[pd];
        pp.ti[q] = pe->ext_topk_i; pp.tv[q] = pe->ext_topk_v;
        HIP_CHECK(hipStreamWaitEvent(compute_stream, g_ev_ag[pd][l % 128], 0));
      }
      xpull_ag_kernel<<<dim3((unsigned)EXPERT_PARALLELISM, (unsigned)BATCH_SIZE), 256, 0,
                        compute_stream>>>(ext->ext_t_bf16, ext->ext_topk_i, ext->ext_topk_v, pp,
                                          own_slice, BATCH_SIZE, hidden_dim,
                                          p->experts_per_token);
    } else
#endif
#if GETP_AGSKIP
    if (ag_on) {
      // Identical to the EP > 2 branch of sync_workers_dev, with one row-count read inserted in
      // BETWEEN: the stream has just drained so rr_cnt is final, and the barrier right after
      // publishes it to every rank.
      HIP_CHECK(hipStreamSynchronize(compute_stream));
      HIP_CHECK(hipMemcpy(g_ag_cnt[own_slice], rr_cnt, sizeof(int) * EXPERT_PARALLELISM,
                          hipMemcpyDeviceToHost));
      sync_point.wait();
    } else
#endif
    {
#if GETP_DEVSYNC
    sync_workers_dev(compute_stream, memory_stream, sync_point, device_index,
                     EXPERT_PARALLELISM, base_device_index, 2 * (int)l + 0);
#else
    sync_workers(compute_stream, sync_point);
#endif
    }
    HIP_CHECK(hipSetDevice(device_index));
#if GETP_XPULL
    const bool xp = ag_on;   // the all-gather already happened in xpull_ag_kernel
#else
    const bool xp = false;
#endif
    for (int delta = 1, j = 0; delta < EXPERT_PARALLELISM && !xp; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      GPUWorker *peer_worker = &workers[i];
      int peer_device_index = peer_worker->device_index;
      RunStateExt *peer_ext = ext_get(peer_device_index);
      if (peer_device_index == device_index) continue;
      const size_t ps = (size_t)(peer_device_index - base_device_index);
#if GETP_AGSKIP
      if (ag_on) {
        // Only the rows we own an expert for. The count has been known since before the sp1 barrier.
        const int nrow = g_ag_cnt[ps][own_slice];
        if (nrow > 0) {
          HIP_CHECK(hipMemcpyPeerAsync(
              ag_ridx + ps * BATCH_SIZE, device_index,
              g_ag_idx[peer_device_index] + (size_t)own_slice * BATCH_SIZE, peer_device_index,
              sizeof(int) * (size_t)nrow, GETP_CS(j)));
          HIP_CHECK(hipMemcpyPeerAsync(
              ag_recv + ps * BATCH_SIZE * hidden_dim, device_index,
              g_ag_send[peer_device_index] + (size_t)own_slice * BATCH_SIZE * hidden_dim,
              peer_device_index,
              sizeof(__hip_bfloat16) * (size_t)nrow * (size_t)hidden_dim, GETP_CS(j)));
        }
      } else
#endif
      HIP_CHECK(hipMemcpyPeerAsync(
          ext->ext_t_bf16 + ps * BATCH_SIZE * hidden_dim, device_index,
          peer_ext->ext_t_bf16 + ps * BATCH_SIZE * hidden_dim, peer_device_index,
          sizeof(__hip_bfloat16) * (size_t)BATCH_SIZE * (size_t)hidden_dim, GETP_CS(j)));
      HIP_CHECK(hipMemcpyPeerAsync(
          ext->ext_topk_i + ps * BATCH_SIZE * p->experts_per_token, device_index,
          peer_ext->ext_topk_i + ps * BATCH_SIZE * p->experts_per_token, peer_device_index,
          sizeof(int) * (size_t)BATCH_SIZE * (size_t)p->experts_per_token, GETP_CS(j)));
      HIP_CHECK(hipMemcpyPeerAsync(
          ext->ext_topk_v + ps * BATCH_SIZE * p->experts_per_token, device_index,
          peer_ext->ext_topk_v + ps * BATCH_SIZE * p->experts_per_token, peer_device_index,
          sizeof(float) * (size_t)BATCH_SIZE * (size_t)p->experts_per_token, GETP_CS(j)));
      HIP_CHECK(hipEventRecord(events_pull[j], GETP_CS(j)));
      ++j;
    }
    for (int delta = 1, j = 0; delta < EXPERT_PARALLELISM && !xp; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      if (workers[i].device_index == device_index) continue;
      HIP_CHECK(hipStreamWaitEvent(compute_stream, events_pull[j], 0));
      ++j;
    }
#if GETP_AGSKIP
    if (ag_on && !xp) {
      for (int delta = 1; delta < EXPERT_PARALLELISM; ++delta) {
        const int pd = workers[(device_index + delta) % EXPERT_PARALLELISM].device_index;
        if (pd == device_index) continue;
        const size_t ps = (size_t)(pd - base_device_index);
        const int nrow = g_ag_cnt[ps][own_slice];
        if (nrow <= 0) continue;
        getp_agskip_scatter_kernel<<<dim3((unsigned)nrow), 256, 0, compute_stream>>>(
            ext->ext_t_bf16 + ps * BATCH_SIZE * hidden_dim,
            ag_recv + ps * BATCH_SIZE * hidden_dim, ag_ridx + ps * BATCH_SIZE, hidden_dim);
      }
    }
#endif
    getp_map_global_to_local_batch(
        ext->ext_topk_i, ext->ext_topk_v, ext->local_ids, ext->local_wts,
        ext->n_local, p->experts_per_token, worker->expert_start,
        worker->expert_end, EXPERT_PARALLELISM * BATCH_SIZE, compute_stream);

    int experts_per_device = worker->expert_end - worker->expert_start;
#if GETP_ROWSKIP
    if (rs_on) {
#if GETP_RS_MERGE
      // One launch for all seven slices, instead of seven sitting 60 us apart.
      getp_rowskip_build_all_send(ext->n_local, BATCH_SIZE, EXPERT_PARALLELISM, own_slice,
                                  rs_rank, rs_idx, rs_cnt, compute_stream);
#if GETP_AGSKIP
      // Already built before sync point 1 for the all-gather direction; rr_* is intact, so reuse it.
      if (!ag_on)
#endif
      getp_rowskip_build_all_recv(ext->ext_topk_i, p->experts_per_token, BATCH_SIZE,
                                  EXPERT_PARALLELISM, own_slice, own_slice,
                                  rs_elo, rs_ehi, rr_rank, rr_idx, rr_cnt, compute_stream);
#else
      // SENDER side: one list per slice that is NOT ours - those are the slices we will send.
      for (int sd = 0; sd < EXPERT_PARALLELISM; ++sd) {
        if (sd == own_slice) continue;
        getp_rowskip_build_send(ext->n_local, BATCH_SIZE, sd,
                                rs_rank, rs_idx, rs_cnt, compute_stream);
      }
      // RECEIVER side: read OUR OWN slice but write into each peer's bucket. The sender derives
      // it from its n_local, the receiver from topk plus the peer's expert range - the same set.
      for (int d = 1; d < EXPERT_PARALLELISM; ++d) {
        GPUWorker *pw = &workers[(device_index + d) % EXPERT_PARALLELISM];
        const int ps = pw->device_index - base_device_index;
        if (ps == own_slice) continue;
        getp_rowskip_build_recv(ext->ext_topk_i, p->experts_per_token, BATCH_SIZE, own_slice,
                                pw->expert_start, pw->expert_end,
                                rr_rank, rr_idx, rr_cnt, ps, compute_stream);
      }
#endif
    }
#endif

    build_moe_buckets_local_pos(
        ext->local_ids, ext->local_wts, ext->n_local,
        EXPERT_PARALLELISM * BATCH_SIZE, p->experts_per_token,
        experts_per_device, ext->e_counts, ext->e_offsets, ext->e_dev,
        ext->w_dev, ext->pair_pos, compute_stream);
#if GETP_EPLB
    if (eplb_count)
      eplb_count_kernel<<<1, 64, 0, compute_stream>>>(eplb_acc + (size_t)l * experts_per_device,
                                                      ext->e_counts, experts_per_device,
                                                      MATMUL_MLP1_BLOCK_ROWS);
#endif

    int total_blocks_mlp1 = build_moe_block_schedule(
      ext->e_offsets, experts_per_device, MATMUL_MLP1_BLOCK_ROWS,
      ext->blk_counts, ext->blk_offsets,
      EXPERT_PARALLELISM * BATCH_SIZE * p->experts_per_token, compute_stream);

#if !GETP_SMALL_NOSCATTER
  moe_scatter_acts_to_expert_frombf16(
      ext->a_in, ext->ext_t_bf16, ext->e_dev,
      ext->e_offsets + experts_per_device, EXPERT_PARALLELISM * BATCH_SIZE * p->experts_per_token,
      hidden_dim, compute_stream);
#endif

  // (the 35,4 MB memset is gone: moe_gather_pairs now overwrites this whole region)

  __hip_bfloat16 *w1_base = dev_w->w_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim * hidden_dim;
  __hip_bfloat16 *b1_base = dev_w->b_mlp1 + 1ll * l * experts_per_device * 2 * p->intermediate_dim;

#if GETP_SMALL_NOSCATTER
  // RAW: the hipMemcpyPeerAsync pulling ext_t_bf16 back on memory_stream was tied into
  // compute_stream by hipStreamWaitEvent BEFORE map_global_to_local above; our own slice was
  // written by rmsnorm(l) earlier on compute_stream.
  // The copy kernel (just removed) read this same memory, at this same point.
  //
  // WAR - this is where the real change is, and it is NOT obvious: mlp1 reads ext_t_bf16 far
  // later than the copy kernel did (49 ms later, rather than 1,4 ms), while LAYER l+1's pull
  // OVERWRITES the peer slices of this same buffer, on memory_stream - and memory_stream never
  // waits on our compute_stream (sync_workers_dev only waits on the PEERS' events). Safety
  // comes from a transitive chain
  // of happens-before edges, with EXPERT_PARALLELISM = 2:
  //   pullT(l+1) [memory_D]  after  E_P(2l+2)            (hipStreamWaitEvent)
  //   E_P(2l+2)              after  vecadd_P(l)          (same compute_P)
  //   vecadd_P(l)            after  event e_agg_P(l)     (hipStreamWaitEvent)
  //   e_agg copy [memory_P]  after  E_D(2l+1)            (hipStreamWaitEvent)
  //   E_D(2l+1)              after gather_D(l) and mlp1_D(l)  (same compute_D)
  // => mlp1(l) is fully done before pull(l+1) starts writing. For n_dev > 2, and for
  // GETP_DEVSYNC=0, sync_workers* goi hipStreamSynchronize(compute_stream)
  // right at the sync point, so the conclusion is even stronger.
  launch_mlp1_swiglu_bf16_bucketed_outbf16(
      ext->gate_up_bf16, ext->ext_t_bf16, w1_base, b1_base, ext->e_offsets,
      ext->blk_offsets, hidden_dim, p->intermediate_dim, experts_per_device,
      total_blocks_mlp1, p->swiglu_limit, compute_stream, ext->e_dev);
#else
  launch_mlp1_swiglu_bf16_bucketed_outbf16(
      ext->gate_up_bf16, ext->a_in, w1_base, b1_base, ext->e_offsets,
      ext->blk_offsets, hidden_dim, p->intermediate_dim, experts_per_device,
      total_blocks_mlp1, p->swiglu_limit, compute_stream);
#endif

  __hip_bfloat16 *w2_base = dev_w->w_mlp2 + 1ll * l * experts_per_device * hidden_dim * p->intermediate_dim;
  __hip_bfloat16 *b2_base = dev_w->b_mlp2 + 1ll * l * experts_per_device * hidden_dim;

#if MATMUL_MLP2_BLOCK_ROWS != MATMUL_MLP1_BLOCK_ROWS
  // mlp2 decodes blk_offsets with its own BM, so when the two BMs differ it needs its own
  // schedule. Overwriting the same buffer is safe: everything is on compute_stream, so this
  // kernel cannot run before mlp1 has finished reading.
  int total_blocks_mlp2 = build_moe_block_schedule(
      ext->e_offsets, experts_per_device, MATMUL_MLP2_BLOCK_ROWS,
      ext->blk_counts, ext->blk_offsets,
      EXPERT_PARALLELISM * BATCH_SIZE * p->experts_per_token, compute_stream);
#else
  const int total_blocks_mlp2 = total_blocks_mlp1;
#endif

  launch_mlp2_partial_bf16_bucketed_frombf16(
      ext->z_partial, ext->gate_up_bf16, w2_base, b2_base, ext->w_dev,
      ext->e_offsets, ext->blk_offsets, ext->e_dev, p->intermediate_dim,
      hidden_dim, experts_per_device, total_blocks_mlp2, compute_stream);

    // The own_slice slice goes straight into dev_x; the other slices still go to ext_e_agg
    // because peers will pull them. That removes one vecadd entirely (17,7 MB read + 17,7 MB written).
    // The two buffers alternate by layer parity - see the note on RunStateExt::ext_e_agg2.
    // Measured on 8 GPUs over 1024 steps: the same binary run twice self-matched on 49,5 % of
    // lines with one buffer and 89,8 % with two; throughput unchanged.
    float *eagg_out = (l & 1) ? ext->ext_e_agg2 : ext->ext_e_agg;
#if GETP_ROWSKIP
    const int *gp_rank = rs_on ? rs_rank : nullptr;
#else
    const int *gp_rank = nullptr;
#endif
#if GETP_EAGG_BF16
    // Gated on rs_on: EP = 2 (20b) never enables it, so md5 720709b86d36 stays intact.
    const int gp_bf16 = rs_on ? 1 : 0;
#else
    const int gp_bf16 = 0;
#endif
    (void)gp_bf16;
#if GETP_GP_OWN_LAST && GETP_DEVSYNC
    const bool gp_own_last = !ag_on && EXPERT_PARALLELISM == 2;
#endif
#if GETP_GPSLICE
    // The rank that pulls from us first is D-1, and it wants its own slice. Emit that slice
    // separately first and fire an early event, so the pull chain need not wait for all 6144 tokens.
    const int gp_first = (own_slice + EXPERT_PARALLELISM - 1) % EXPERT_PARALLELISM;
    if (ag_on) {
      moe_gather_pairs_acc(eagg_out, dev_x, own_slice * BATCH_SIZE,
                           own_slice * BATCH_SIZE + BATCH_SIZE, ext->z_partial, ext->pair_pos,
                           hidden_dim, p->experts_per_token, EXPERT_PARALLELISM * BATCH_SIZE,
                           compute_stream, gp_rank, gp_first * BATCH_SIZE, BATCH_SIZE, gp_bf16);
      HIP_CHECK(hipEventRecord(g_ev_eagg_first[device_index][(int)(l % 128)], compute_stream));
      if (gp_first > 0)
        moe_gather_pairs_acc(eagg_out, dev_x, own_slice * BATCH_SIZE,
                             own_slice * BATCH_SIZE + BATCH_SIZE, ext->z_partial, ext->pair_pos,
                             hidden_dim, p->experts_per_token, EXPERT_PARALLELISM * BATCH_SIZE,
                             compute_stream, gp_rank, 0, gp_first * BATCH_SIZE, gp_bf16);
      if (gp_first + 1 < EXPERT_PARALLELISM)
        moe_gather_pairs_acc(eagg_out, dev_x, own_slice * BATCH_SIZE,
                             own_slice * BATCH_SIZE + BATCH_SIZE, ext->z_partial, ext->pair_pos,
                             hidden_dim, p->experts_per_token, EXPERT_PARALLELISM * BATCH_SIZE,
                             compute_stream, gp_rank,
                             (gp_first + 1) * BATCH_SIZE,
                             (EXPERT_PARALLELISM - gp_first - 1) * BATCH_SIZE, gp_bf16);
    } else
#endif
#if GETP_GP_OWN_LAST && GETP_DEVSYNC
    // EP = 2: only the peer's slice here; our own rows follow the sync point below.
    if (gp_own_last)
      moe_gather_pairs_acc(eagg_out, dev_x, own_slice * BATCH_SIZE,
                           own_slice * BATCH_SIZE + BATCH_SIZE, ext->z_partial, ext->pair_pos,
                           hidden_dim, p->experts_per_token, EXPERT_PARALLELISM * BATCH_SIZE,
                           compute_stream, gp_rank, (1 - own_slice) * BATCH_SIZE, BATCH_SIZE,
                           gp_bf16);
    else
#endif
    moe_gather_pairs_acc(eagg_out, dev_x,
                         own_slice * BATCH_SIZE,
                         own_slice * BATCH_SIZE + BATCH_SIZE,
                         ext->z_partial, ext->pair_pos, hidden_dim,
                         p->experts_per_token, EXPERT_PARALLELISM * BATCH_SIZE,
                         compute_stream, gp_rank, 0, -1, gp_bf16);

#if GETP_EVAGG
    const int ev_k = (int)(l % 128);
    if (ag_on) {
      // No drain. The barrier is only there to be sure every thread has ISSUED its event:
      // waiting on an event that has not been recorded returns at once and we would read the
      // previous step's e_agg.
      HIP_CHECK(hipEventRecord(g_ev_eagg[device_index][ev_k], compute_stream));
      sync_point.wait();
    } else
#endif
    {
#if GETP_DEVSYNC
    sync_workers_dev(compute_stream, memory_stream, sync_point, device_index,
                     EXPERT_PARALLELISM, base_device_index, 2 * (int)l + 1);
#if GETP_GP_OWN_LAST
    // Our own rows, into dev_x, while the peer pulls its slice. vecadd below comes after them on
    // compute_stream.
    if (gp_own_last)
      moe_gather_pairs_acc(eagg_out, dev_x, own_slice * BATCH_SIZE,
                           own_slice * BATCH_SIZE + BATCH_SIZE, ext->z_partial, ext->pair_pos,
                           hidden_dim, p->experts_per_token, EXPERT_PARALLELISM * BATCH_SIZE,
                           compute_stream, gp_rank, own_slice * BATCH_SIZE, BATCH_SIZE, gp_bf16);
#endif
#else
    sync_workers(compute_stream, sync_point);
#endif
    }
#if GETP_ROWSKIP
    // Free at EP > 2: sync_workers_dev has just called hipStreamSynchronize, so the stream is drained.
    // With EVAGG there is no drain any more, but this read is no longer needed either: agskip
    // already put those same numbers into g_ag_cnt[own_slice][*] before the sp1 barrier.
#if GETP_EVAGG
    if (rs_on && !ag_on)
#else
    if (rs_on)
#endif
      HIP_CHECK(hipMemcpy(rs_hcnt, rr_cnt, sizeof(int) * EXPERT_PARALLELISM,
                          hipMemcpyDeviceToHost));
#if GETP_EVAGG
    if (ag_on && !xp)
      for (int q = 0; q < EXPERT_PARALLELISM; ++q) rs_hcnt[q] = g_ag_cnt[own_slice][q];
#endif
#endif
#if GETP_XPULL
    if (xp) {
      // Wait for each peer's gather, then add every peer's partial sums in the old delta order.
      static thread_local int *xp_order = nullptr;
      int horder[MAXIMUM_GPU], np = 0;
      XPeers pp = {};
      for (int delta = 1; delta < EXPERT_PARALLELISM; ++delta) {
        const int pd = workers[(device_index + delta) % EXPERT_PARALLELISM].device_index;
        if (pd == device_index) continue;
        const int ps = pd - base_device_index;
        horder[np++] = ps;
        RunStateExt *pe = ext_get(pd);
        pp.e[ps] = reinterpret_cast<const __hip_bfloat16 *>((l & 1) ? pe->ext_e_agg2 : pe->ext_e_agg);
        HIP_CHECK(hipStreamWaitEvent(compute_stream, g_ev_eagg[pd][ev_k], 0));
      }
      if (xp_order == nullptr) {
        HIP_CHECK(hipMalloc(&xp_order, sizeof(int) * MAXIMUM_GPU));
        HIP_CHECK(hipMemcpy(xp_order, horder, sizeof(int) * np, hipMemcpyHostToDevice));
      }
      xpull_rs_kernel<<<dim3((unsigned)((hidden_dim / 8 + 127) / 128), (unsigned)BATCH_SIZE), 128, 0,
                        compute_stream>>>(dev_x, pp, rr_rank, xp_order, np, own_slice, BATCH_SIZE,
                                          hidden_dim);
    }
#endif

    for (int delta = 1, j = 0; delta < EXPERT_PARALLELISM && !xp; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      GPUWorker *peer_worker = &workers[i];
      int peer_device_index = peer_worker->device_index;
      RunStateExt *peer_ext = ext_get(peer_device_index);
      if (peer_device_index != device_index) {
#if GETP_EVAGG
        // Instead of a barrier: wait only for THIS peer to finish writing, not for all eight.
        if (ag_on)
#if GETP_GPSLICE
          // delta == 1 is the peer we pull from first: it fired an early event for our slice alone.
          HIP_CHECK(hipStreamWaitEvent(GETP_CS(j),
                                       (delta == 1 ? g_ev_eagg_first : g_ev_eagg)
                                           [peer_device_index][ev_k], 0));
#else
          HIP_CHECK(hipStreamWaitEvent(GETP_CS(j), g_ev_eagg[peer_device_index][ev_k], 0));
#endif
#endif
#if GETP_EAGG_BF16
        // The same number of ROWS, half the BYTES. gp_bf16 implies rs_on, so the rowskip size formula applies.
        if (gp_bf16) {
          const size_t ebs = (size_t)BATCH_SIZE * (size_t)hidden_dim;
          __hip_bfloat16 *ebdst =
              reinterpret_cast<__hip_bfloat16 *>(ext->peer_e_agg) + (size_t)j * ebs;
          const __hip_bfloat16 *ebsrc = reinterpret_cast<const __hip_bfloat16 *>(
              (l & 1) ? peer_ext->ext_e_agg2 : peer_ext->ext_e_agg)
              + (size_t)(device_index - base_device_index) * ebs;
          HIP_CHECK(hipMemcpyPeerAsync(
              ebdst, device_index, ebsrc, peer_device_index,
              sizeof(__hip_bfloat16)
                  * (size_t)rs_hcnt[peer_device_index - base_device_index]
                  * (size_t)hidden_dim,
              GETP_CS(j)));
        } else
#endif
        HIP_CHECK(hipMemcpyPeerAsync(
            ext->peer_e_agg + (size_t)j * BATCH_SIZE * hidden_dim, device_index,
            ((l & 1) ? peer_ext->ext_e_agg2 : peer_ext->ext_e_agg)
                + (size_t)(device_index - base_device_index) * BATCH_SIZE * hidden_dim,
            peer_device_index,
#if GETP_ROWSKIP
            rs_on ? (sizeof(float) * (size_t)rs_hcnt[peer_device_index - base_device_index]
                     * (size_t)hidden_dim)
                  : (sizeof(float) * BATCH_SIZE * hidden_dim),
#else
            sizeof(float) * BATCH_SIZE * hidden_dim,
#endif
            GETP_CS(j)));
        HIP_CHECK(hipEventRecord(events_e_agg[j], GETP_CS(j)));
        ++j;
      }
    }
    for (int delta = 1, j = 0; delta < EXPERT_PARALLELISM && !xp; ++delta) {
      int i = (device_index + delta) % EXPERT_PARALLELISM;
      GPUWorker *peer_worker = &workers[i];
      int peer_device_index = peer_worker->device_index;
      if (peer_device_index != device_index) {
        HIP_CHECK(hipStreamWaitEvent(compute_stream, events_e_agg[j]));
#if GETP_ROWSKIP
        if (rs_on) {
          const int ps = peer_device_index - base_device_index;
#if GETP_EAGG_BF16
          if (gp_bf16)
            getp_rowskip_scatter_add_bf16(
                dev_x,
                reinterpret_cast<const __hip_bfloat16 *>(ext->peer_e_agg)
                    + (size_t)j * BATCH_SIZE * hidden_dim,
                rr_idx + (size_t)ps * BATCH_SIZE, rr_cnt + ps, hidden_dim, rs_hcnt[ps],
                compute_stream);
          else
#endif
          getp_rowskip_scatter_add(dev_x,
                                   ext->peer_e_agg + (size_t)j * BATCH_SIZE * hidden_dim,
                                   rr_idx + (size_t)ps * BATCH_SIZE, rr_cnt + ps,
                                   hidden_dim, rs_hcnt[ps], compute_stream);
        } else
#endif
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
#undef GETP_CS
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
    // Layers are laid out even ones first, then odd ones (layer_toff); each also carries its padding.
    const size_t base_off = layer_toff * (size_t)BATCH_SIZE * (size_t)kv_dim +
                            (size_t)(is_even ? idx_half : even_layers + idx_half) *
                                GETP_KV_LAYER_PAD(BATCH_SIZE, kv_dim);

    const size_t slot_off = base_off + GETP_KV_OFF(tslot, 0, 0, BATCH_SIZE, kv_dim, cache_tcap);
    __hip_bfloat16 *k_slot = GETP_KV_PTR_K(dev_s->key_cache, slot_off);
    __hip_bfloat16 *v_slot = GETP_KV_PTR_V(dev_s->value_cache, slot_off);
    float *k_slot_scale = nullptr, *v_slot_scale = nullptr;
    const float *k_layer_scale = nullptr, *v_layer_scale = nullptr;
#if GETP_KV8_K
    k_slot_scale = getp_kv8_scales(dev_s->key_cache, p, BATCH_SIZE) + GETP_KV_SIDX(slot_off);
    k_layer_scale = getp_kv8_scales(dev_s->key_cache, p, BATCH_SIZE) + GETP_KV_SIDX(base_off);
#endif
#if GETP_KV8_V
    v_slot_scale = getp_kv8_scales(dev_s->value_cache, p, BATCH_SIZE) + GETP_KV_SIDX(slot_off);
    v_layer_scale = getp_kv8_scales(dev_s->value_cache, p, BATCH_SIZE) + GETP_KV_SIDX(base_off);
#endif

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
#if GETP_ROPE_FUSED
    // RoPE now lives in the qkv epilogue (and with GETP_ROPE_FUSED_KV so does the
    // bf16 conversion into the cache slot). dev_cos_vals/dev_sin_vals are generated
    // before the layer loop on the SAME stream, so the ordering still holds; the live range
    // of the two tables is unchanged - apply_rotary_emb used to read them at EXACTLY
    // this point, right after qkv and before the layer's MoE.
    getp_matmul_qkv_fused_bf16(dev_s->q, k_step, v_step, ext->pre_qkv_bf16,
                               dev_w_qkv, dev_b_qkv, hidden_dim, head_dim,
                               p->n_attn_heads, p->n_kv_heads, batch_size,
                               nullptr, dev_cos_vals, dev_sin_vals,
                               k_slot, v_slot, cache_tcap, k_slot_scale, v_slot_scale);
#if !GETP_ROPE_FUSED_KV
    kv_store_pair_fp32_to_bf16(k_slot, v_slot, k_step, v_step,
                               batch_size * kv_dim, nullptr);
#endif
#else
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
#endif

    {
      const __hip_bfloat16 *k_layer = GETP_KV_PTR_K(dev_s->key_cache, base_off);
      const __hip_bfloat16 *v_layer = GETP_KV_PTR_V(dev_s->value_cache, base_off);
      const float *q_ptr = dev_s->q;
      const float *attn_sinks_layer =
          dev_w->attn_sinks + (size_t)l * p->n_attn_heads;
      getp_flash_attn_decode_bf16(
          ext->attn_o_bf16, dev_s->tb,
          k_layer, v_layer, q_ptr, ext->mask_on, attn_sinks_layer, head_dim,
          p->n_attn_heads, p->n_kv_heads, pos, p->seq_len, p->sliding_window,
          (int)l, batch_size, cache_tcap, nullptr, k_layer_scale, v_layer_scale);
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

    build_moe_buckets_local_pos(
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
      ext->blk_counts, ext->blk_offsets,
      EXPERT_PARALLELISM * BATCH_SIZE * p->experts_per_token, nullptr);

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
