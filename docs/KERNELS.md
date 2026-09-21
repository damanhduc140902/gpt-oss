# Core kernels: GEMM, attention, mixture-of-experts

Matrix multiplication, multi-head attention and the mixture-of-experts block are where essentially all
of the time goes, and all three are implemented from scratch in a single file,
[`src/hip/forward.hip`](../src/hip/forward.hip), against the project's constraint of standard C/C++ plus
`hip` and `omp` — no rocBLAS, no hipBLAS. The target is one GCD of an AMD MI250 (`gfx90a`, 64-lane
waves, 104 compute units, 64 KB of LDS per CU), and the workload is a large batch of _requests_ decoded
one token at a time: 1536 requests per GPU for `gpt-oss-20b`, 768 for `gpt-oss-120b`
([`src/getp/run.cpp`](../src/getp/run.cpp)). That single fact shapes everything below. The GEMMs see a
tall, thin left-hand operand and can be tuned per call site; attention sees exactly one query token per
request, which looks like a pure memory-traffic problem and was treated as one until a drain test said
otherwise — see [Multi-head attention](#multi-head-attention).

## Matrix multiply

Every matrix multiply in the engine computes $\text{C}=\text{A}\times\text{B}^{\top}$, where the sizes
of $\text{C}$, $\text{A}$ and $\text{B}$ are respectively $\text{M}\times\text{N}$,
$\text{M}\times\text{K}$ and $\text{N}\times\text{K}$. $\text{M}$ is mostly the batch size the system
operates at; $\text{N}$ and $\text{K}$ are fixed by the model architecture and vary a great deal between
layers. Storing $\text{B}$ transposed means both operands are K-major, so the reduction axis is
contiguous for both and every global load can be a 16-byte `uint4`.

### Seven GEMMs, one per call site

There is no general-purpose GEMM here. There are seven matrix-multiply kernels, one for each place in
the model where a matrix multiply happens, and each is launched from exactly one live site. Shape is fixed at
compile time at every site except the router's, whose launcher inspects $\text{M}$ and $\text{N}$ at
runtime to pick between two kernels.

| Kernel                                                       | Role                        | $\text{M}\times\text{N}\times\text{K}$ at 20b |
| ------------------------------------------------------------ | --------------------------- | --------------------------------------------- |
| `new_matmul_qkv_fused_kernel`                                | fused Q/K/V projection      | 1536 × 5120 × 2880                            |
| `matmul_attn_o_bf16_kernel_tuned`                            | attention output projection | 1536 × 2880 × 4096                            |
| `matmul_kernel_nosplit` / `matmul_kernel_splitk_store_tuned` | router                      | 1536 × 32 × 2880                              |
| `mlp1_swiglu_bf16_bucketed_kernel_outbf16_tuned`             | expert gate/up + SwiGLU ‡   | M × (2 × 2880) × 2880                         |
| `mlp2_partial_bf16_bucketed_splitk_kernel_inbf16_tuned`      | expert down projection      | M × 2880 × 2880                               |
| `matmul_logits_argmax_bf16_kernel_tuned`                     | unembedding + argmax        | 1536 × 201088 × 2880                          |

‡ MLP1's $\text{N}$ counts _weight rows_, not output channels. gpt-oss interleaves the gate and up rows,
so the kernel consumes 2 × 2880 rows and writes 2880 SwiGLU outputs per token. Sizes that follow the
output axis — the grid, and the LDS tiling in [The fused expert GEMMs](#the-fused-expert-gemms) — use 2880.

The QKV kernel's $\text{N}=5120$ is $4096+512+512$: one GEMM produces Q, K and V together and the
epilogue fans the result out, so the activation tile is staged and read once instead of three times.
With `GETP_ROPE_FUSED` and `GETP_ROPE_FUSED_KV` (both default 1) the fan-out also does the rotation and
the bf16 rounding: `WN = 64 = head_dim` and every region boundary is a multiple of 64, so a wave sits
wholly inside one head of one region and the `dim + head_dim/2` partner is in the same lane — no shuffle.

```c
if (in_q) {                                     // q: bf16 under GETP_Q_BF16 (default 1)
  unsigned short *dst = reinterpret_cast<unsigned short *>(q_out) + (size_t)global_y * q_len;
  dst[gx1] = f32_to_bf16bits(q_scale * o1);     // pre-scaled by 1/sqrt(head_dim)
  dst[gx2] = f32_to_bf16bits(q_scale * o2);
} else {                                        // k: straight into this step's KV cache slot
  unsigned short *dst = reinterpret_cast<unsigned short *>(k_slot_bf16) + (size_t)global_y * k_len;
  dst[gx1 - q_len] = f32_to_bf16bits(o1);
  dst[gx2 - q_len] = f32_to_bf16bits(o2);
}
```

The three destinations on this path are therefore `q_out`, `k_slot_bf16` and `v_slot_bf16`, not a
separate K and V tensor. The V region is a second arm that does the same store without the rotation.
Pre-scaling q is its own switch, `GETP_Q_BF16` (default 1, and it requires `GETP_ROPE_FUSED`); with it
off, q goes out as fp32 and everything else here is unchanged.

The old fp32 fan-out to `q_out`/`k_out`/`v_out` is still present as the `else` arm, but nothing reaches
it: the launcher aborts on a shape that would disable fusion rather than fall through, because with RoPE
fused there is no separate rotary kernel left — the `getp_apply_rotary_emb` and
`kv_store_pair_fp32_to_bf16` calls are compiled out on this path.

For the two expert GEMMs, $\text{M}$ is not the batch size but the number of routed rows assigned to one
expert; see [Mixture-of-experts](#mixture-of-experts).

### Tiles

All seven kernels launch 256 threads — four 64-lane waves — per workgroup: every shipped tile has `(BM/WM) * (BN/WN) = 4`, MLP2 included since it moved to a 128-row block tile over a 64 × 32 wave tile. All accumulate in fp32 from bf16 inputs.
threads, six waves, because its 96-row block tile over a 32-row wave tile gives
`(96/32) * (64/32) = 6`. All accumulate in fp32
from bf16 inputs. What varies is the tile. Six of the seven take their tile from `#ifndef`-guarded
`#define`s sitting next to the kernel — `MATMUL_MLP1_*`, `MATMUL_MLP2_*`, `MATMUL_ROUTER_*`,
`MATMUL_ATTN_O_*` and `MATMUL_LOGITS_*` — and those are overridable from the compiler command line. The
values below are the shipped defaults. The QKV kernel is the exception: there is no `MATMUL_QKV_*`
macro anywhere in the file and `-D` cannot move its tile.

| Kernel         | BM × BN × BK   | Wave tile | BN_AGG | Effective N per workgroup (BNt) | LDS per workgroup |
| -------------- | -------------- | --------- | ------ | ------------------------------- | ----------------- |
| QKV †          | 128 × 128 × 32 | 64 × 64   | 1      | 128                             | 16 640 B          |
| Attention out  | 128 × 128 × 32 | 64 × 64   | 1      | 128                             | 16 640 B          |
| Logits         | 128 × 128 × 32 | 64 × 64   | 1      | 128                             | 19 456 B          |
| MLP1 (gate/up) | 128 × 64 × 32  | 64 × 32   | 2      | 128                             | 24 960 B          |
| MLP2 (down)    | 128 × 64 × 32  | 64 × 32   | 3      | 192                             | 20 736 B          |
| Router         | 32 × 32 × 32   | 16 × 16   | 4      | 128                             | 20 480 B          |

These are the defaults a plain `runfast` build compiles; none of them has to be passed on the command
line. The logits figure includes the 1 024 B of `smax`/`sidx` that the fused argmax keeps alongside the
tile.

† Not tunable by `-D`. `new_matmul_qkv_fused_kernel` takes its shape from `struct GetpQkvTile`
([`src/hip/forward.hip`](../src/hip/forward.hip)) — `BM = 128; BN = 128; BK = 32; WM = 64; WN = 64;
THREADS = 256;` — which the launcher and the kernel body both read, bound together by a `static_assert`
in the kernel body. Retuning means editing that struct and recompiling; the LDS padding follows from
`GetpQkvLdsTile` on its own.

The pattern is the one blocktiling predicts. Where $\text{M}$, $\text{N}$ and $\text{K}$ are all in the
thousands, a 128 × 128 block tile with a 64 × 64 wave tile gives the best ratio of matrix-core work to
LDS traffic. The two MoE kernels share a block height again. MLP1 keeps BM = 128 over a 64 × 32 wave tile;
MLP2 runs BM = 128 over a 64 × 32 wave tile at `BN_AGG = 3`. The figure that matters for both is
matrix-core work per LDS read, and it is fixed by the wave tile and `BN_AGG`, not by the block height:
one 16-wide k step costs `nIterM = WM/16` A reads and `BN_AGG × nIterN` B reads for
`nIterM × BN_AGG × nIterN` matrix-core instructions, and BM appears in none of those counts. At the
32 × 32 wave tile MLP2 shipped with until this pass that was 2 A + 6 B reads for 12 MFMA, 1.50 per
`ds_read_b64` against MLP1's 2.67 — the whole of why it reached 34 % of MFMA peak where MLP1 reaches
55 %. Moving BM from 64 to 96 changed the wave count per workgroup (4 to 6) and nothing in that ratio.
What BM = 128 buys is the 64 × 32 wave tile — 128 divides by 64, 96 does not — and with `nIterM = 4`
the same k step costs 4 A + 6 B reads for 24 MFMA, 2.40 per read. 128 was held to produce wrong output
for most of this work; the split-K section below records why it does not. The router drops all the way to
32 × 32 × 32 with a 16 × 16 wave tile because its $\text{N}$ is only the expert count — 32 for 20b, 128
for 120b — and a wider tile would mask off most of its own output.

Inside the k-loop MLP2 now reads its LDS fragments one 16-wide k step ahead into a second register
set, and pins the order the scheduler emits with `__builtin_amdgcn_sched_group_barrier`: the six
fragment reads of the first sub-step, then one `ds_read` after every three MFMAs for the second, then the
remaining MFMAs, closed by `__builtin_amdgcn_sched_barrier(0)` so nothing sinks below the barrier. Left
to itself the compiler placed each second-sub-step read directly before the MFMA that consumed it, and the
ISA read `s_waitcnt, mfma, s_waitcnt, mfma` eight times per k step — each MFMA paying a full LDS latency.
Pinned, the same arithmetic in the same order runs 1 272 → 1 172 µs per call at 1024 steps (−7.9 %),
bit-exact. The identical treatment on MLP1 did not help — every variant landed within ±3 % of the
unpinned kernel — so MLP1 keeps the compiler's schedule. The epilogue was also rewritten to load its
six bias values and sixteen row weights once, up front: the old form compiled to 8 100 instructions
(192 `atomic_cmpswap` loops for a split-K branch that never runs, 400 scalar loads behind per-row
`continue`s), the new one to 1 569 — no measurable speed change, since that code never executed, but a
kernel one can read.

The logits GEMM is the extreme case in the other direction. At $\text{N}=201088$ its grid is
$\lceil 201088/128 \rceil = 1571$ column tiles wide, so occupancy is never in question and the design
question is purely how to avoid writing the result (see [Fused epilogues](#fused-epilogues)).

Depth (BK) is 32 everywhere. The attention-out and logits kernels used to run at 16, trading a shorter
K loop per LDS round trip for a smaller LDS footprint; once their staging went k-contiguous the shorter
loop stopped paying for itself, and both now stage a BK = 32 tile — 16 640 B and 19 456 B.

One caution if you retune with `-D`. Three of the seven kernels still derive the wave's _column_ tile
index from `BM / WM` rather than `BN / WN`: `waveIdx % (BM / WM)` in `new_matmul_qkv_fused_kernel`, and
`wave % (BM / WM)` in both router kernels. That is correct only while `BM/WM == BN/WN`, which holds for
every shipped configuration. Of the three, only the router pair is reachable by `-D`, and there the
`static_assert(WARPS_PER_BLOCK == (BM/WM)*(BN/WN))` both kernels carry still passes for a divergent
setting, so it will not catch the mistake. `new_matmul_qkv_fused_kernel` is not a template and has no
`WARPS_PER_BLOCK` at all; its only shape assert is the one binding its own literals to `GetpQkvTile`,
which checks that the two agree and says nothing about `BM/WM == BN/WN` — so the same defect is latent
there, waiting on whoever retunes that struct and the kernel body together. MLP1, MLP2 and the logits
kernel compute the index from a named `splitN = BN / WN`, and `matmul_attn_o_bf16_kernel_tuned` divides by
`BN / WN` inline — it carries a comment marking the fix as the same latent defect QKV still has.

One trap still sits in a launcher rather than a kernel. `getp_matmul_router_bf16` does not read
`MATMUL_ROUTER_*` at all: it re-declares the tile as `constexpr int BM = 32, BN = 32, BK = 32;` and
computes both the grid and the _dynamic_ LDS request from those literals, while
`matmul_kernel_nosplit` and `matmul_kernel_splitk_store_tuned` size their `extern __shared__` arrays from
the macros. Overriding `-DMATMUL_ROUTER_BLOCK_ROWS=64` therefore leaves the kernels reading past the end
of a shared-memory block allocated for a 32-row tile.

The QKV launcher used to carry the same class of defect — its own `const int BK = 16;` while the kernel
body used 32, harmless only because the variable was dead. That copy is gone:
`getp_matmul_qkv_fused_bf16` now reads the shape from `struct GetpQkvTile`, and the kernel body carries
a `static_assert` binding its `BM/BN/BK/WM/WN/BLOCK_SIZE` to that same struct, so the two cannot drift
apart again.

### BN_AGG: widening N without widening the wave

Every kernel except the three 128 × 128 projections carries a second N factor. `GETP_BN_AGG` (default 4)
and the per-kernel `BN_AGG` template argument multiply the workgroup's column extent:
`BNt = BN * BN_AGG`. The block still stages a BM × BK activation tile, but it stages BNt columns of
weight, and each wave keeps `BN_AGG` independent accumulator sets spaced BN apart, rather than covering
the extra columns with more waves or a wider wave tile.

The point is A-fragment reuse. In the router loop, one `av` is read from LDS and fed to four matrix-core
instructions against four different `bv`s:

```c
const unsigned short* sx_ptr = &sx[basek * BM + row];
bf16x4 av = {sx_ptr[0], sx_ptr[BM], sx_ptr[2*BM], sx_ptr[3*BM]};

#pragma unroll
for (int g = 0; g < GETP_BN_AGG; ++g) {
  const unsigned short* sw_ptr = &sw[basek * BNt + c + g * BN];
  bf16x4 bv = {sw_ptr[0], sw_ptr[BNt], sw_ptr[2*BNt], sw_ptr[3*BNt]};

  d_acc[g] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, d_acc[g], 0, 0, 0);
}
```

For the router this is what rescues an otherwise hopeless shape. At BNt = 128 the entire router output
width fits in one block column for both models (32 ≤ 128, and 128 = 128), so `grid.x` is 1, and the LDS
read of the activation is amortised over four matrix-core instructions instead of one.

MLP1 applies the same idea along a different axis. gpt-oss stores the mlp1 weight interleaved — row
`2c` is the gate row for output column `c`, row `2c+1` the up row — so MLP1 keeps two accumulator banks,
`dg_acc` and `du_acc`, and issues the gate and up instructions back to back from one shared `av`. The
gate/up pair costs one A read, not two.

`GETP_BN_AGG` itself reaches only the router pair; MLP1 and MLP2 pass literal 2 and 3 as template
arguments at their launch sites.

### Matrix cores

Every GEMM issues the same single form of matrix-core instruction:

```c
dg_acc[g][im][in] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, gv_array[in], dg_acc[g][im][in], 0, 0, 0);
du_acc[g][im][in] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, uv_array[in], du_acc[g][im][in], 0, 0, 0);
```

The `_1k` suffix is the CDNA2 form, taking four bf16 per lane; the benchmark build passes
`--offload-arch=gfx90a` (see [Building these kernels](#building-these-kernels), which is not what a bare
`make` gives you).
Fragment types are a two-VGPR `bf16x4` for A and B and a four-VGPR `f32x4` for C/D. The lane mapping is
the standard one and is visible in the index arithmetic: `xMF = lane & 15` selects the row (A) or column
(B), `yMF = lane >> 4` times four selects the K offset, and the epilogues invert it with
`xD = lane & 15`, `yD = 4 * (lane >> 4)`. The 32 × 32 × 8 shape is not used anywhere, and neither is any
fp16, fp8 or XF32 variant. `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` is now the only matrix-core
instruction the engine issues at all: the attention kernel uses it too, for both $\text{QK}^{\top}$ and
$\text{PV}$. `__builtin_amdgcn_mfma_f32_4x4x4bf16_1k` is still in the file, but only inside the two
retired attention kernels, which nothing instantiates.

Counted per workgroup per block-K tile:

| Kernel        | Matrix-core instructions per K-tile | Accumulator VGPRs per lane | Prefetch VGPRs per lane |
| ------------- | ----------------------------------- | -------------------------- | ----------------------- |
| MLP1          | 256 (128 gate + 128 up)             | 128                        | 24                      |
| MLP2          | 192                                 | 96                         | 20                      |
| Router        | 32                                  | 16                         | 16                      |
| QKV           | 128 (BK = 32, two K steps)          | 64                         | 16                      |
| Attention out | 128 (BK = 32, two K steps)          | 64                         | 16                      |
| Logits        | 128 (BK = 32, two K steps)          | 64                         | 16                      |

MLP1's 256 and MLP2's 192 are the per-workgroup totals at their new shapes: MLP1 has four waves issuing
64 instructions each, MLP2 six waves issuing 24 each.

The claim that all computation instructions are issued by the matrix cores holds inside the K loop. The
epilogues are ordinary VALU work, and that is deliberate — the epilogue is where the fusion lives.

### Fused epilogues

Fusing work into the epilogue removes whole tensors from HBM, which at these shapes matters far more
than the arithmetic does.

- **Logits** never writes a logits row at all. Each block reduces its own 128 × 128 tile to one
  (max, index) pair per row in the LDS arrays `smax`/`sidx`, then commits each row's pair into that
  request's cell with a packed 64-bit atomic — so the 1571 column blocks covering one row merge against
  each other rather than against a materialised logits vector. That removes a
  1536 × 201088 × 4 B = 1.24 GB write every step, at the cost of supporting only greedy decoding.
- **MLP1** clamps both branches, applies SwiGLU and writes bf16 directly, so the intermediate never
  exists in fp32 and MLP2 reads it at half the width with no conversion pass in between.
- **MLP2** applies the routing weight, which makes the final MoE gather a plain sum.
- **QKV** fans out to three destinations and does the rotation and the rounding on the way: q is rotated, scaled by `1/sqrt(head_dim)` and written as bf16; K is rotated and written as bf16 straight into this step's KV cache slot; V is written as bf16 into its own slot unrotated. That retires both `getp_apply_rotary_emb` launches and `kv_store_pair_fp32_to_bf16` from the live path. The fp32 fan-out shown above is the `GETP_ROPE_FUSED 0` fallback arm.
- **Attention out** adds into the fp32 residual instead of writing a tensor of its own.
  `matmul_attn_o_bf16_kernel_tuned` takes an `ACC` template flag, and the live launcher is
  `getp_matmul_attn_o_bf16_acc`, which instantiates it with `ACC = true`. That removes a 17.7 MB write
  and a 17.7 MB read per layer. The non-accumulating `getp_matmul_attn_o_bf16` is still defined and no
  longer called.
- **The MoE gather** does the same on the way out. `moe_gather_pairs_kernel` adds its own-device slice
- **The MoE gather** does the same on the way out. `moe_gather_pairs_kernel` adds its own-device slice
  straight into the residual and _overwrites_ the peer slices rather than accumulating into them, which
  is why the 35 MB per-layer memset of `ext_e_agg` is gone. At `EXPERT_PARALLELISM > 2` — the 120B
  configuration; `GETP_ROWSKIP` self-disables below it, so 20B keeps the plain overwrite — the kernel
  goes further and skips any row whose token routes no expert to this device. Such a row is all zeros,
  so it is not written at all, and the surviving rows are compacted to the head of their slice. Nothing
  reads the cells that were skipped: the peer copies, and scatter-adds, only the first `cnt` rows, which
  on 120B is 43 % of them.
  every cell is written exactly once — which is why the 35 MB per-layer memset of `ext_e_agg` is gone.

Operand precision is handled three ways. The router takes fp32 activations and converts to bf16 during the global-to-LDS store, so the conversion rides along with a load that has to happen anyway. The logits kernel used to do the same; with `GETP_LOGITS_XBF16` (default 1) a separate `logits_x_to_bf16_kernel` converts the whole activation matrix once per step and the GEMM stages that bf16 copy through `xb` instead — half the bytes per staging load, which is what moved the group swizzle's optimum from G = 2 to G = 6. The fp32 convert-on-store path for logits survives as the `#else` arm. QKV and attention-out want bf16 already in memory, and their producers write it directly: `getp_rmsnorm_bf16` emits bf16 into `pre_qkv_bf16`, and the attention kernel rounds in the lane and writes `attn_o_bf16` itself. The separate `tensor_fp32_to_bf16` pass over each of those tensors is gone from the live path; the remaining calls to it sit in the dispatcher's retired `#else` arms and in the never-called `getp_forward_20b`.

### Double buffering, and the register pipeline underneath it

All seven kernels run the same software pipeline. The global loads for tile _k+1_ are issued into
`uint4`/`float4` registers at the top of the loop body; the matrix-core instructions for tile _k_ run
out of LDS while those loads are in flight; then a barrier, the register-to-LDS store, and a second
barrier. Two `__syncthreads()` per K step, in every one of them. This is what actually overlaps HBM
latency with compute, and it costs only the prefetch registers listed above.

LDS double buffering proper — two staging buffers alternating on `(bk / BK) & 1` — exists only in the
two router kernels, which is why their LDS budget is `2 * BK * (BM + BNt) * 2 B = 20480 B` while the
others allocate a single buffer. The extra buffer buys less than it appears to, because the register
prefetch already covers the HBM latency and the mid-loop barrier is kept regardless.

All five non-router GEMMs now stage LDS **k-contiguously**. Four consecutive k values of one row are
exactly the fragment one lane needs for `v_mfma_f32_16x16x16bf16_1k`, so keeping them adjacent turns an
A or B fragment read into a single `ds_read_b64` instead of four `ds_read_u16`, and turns the staging
store of one 16-byte HBM load into two `ds_write_b64` instead of eight scattered `ds_write_b16`.

Four of them — MLP1, MLP2, QKV and attention-out — use a `[k/4][row][4]` cell layout, where the cell is
four `unsigned short` = 8 bytes. Padding there is counted in _cells_, not in halves, and each of the four
carries a `constexpr` predicate plus a `static_assert` checking that the 16 lanes of a `ds_write_b64`
phase land on 16 distinct even banks; at BK = 32 the answer is 2 cells, 16 bytes, for all four. MLP1 and
MLP2 request their LDS dynamically, so that arithmetic lives in a plan struct — `Mlp1LdsPlan`,
`Mlp2LdsPlan` — which the kernel body and its launcher both instantiate, and neither can therefore
disagree with the other about how much shared memory the tile needs. QKV (`GetpQkvLdsTile`) and
attention-out (`attn_o_lds_pad_slots`) declare theirs statically, so there is nothing to keep in sync.
The old `MATMUL_MLP2_LDS_PAD` counted the old unit and is now an `#error` if it is
defined at all, rather than a silently ignored `-D`.

The logits kernel is k-contiguous too but arranges it the other way, `[row][k]`, with `KPAD = BK +
MATMUL_LOGITS_LDS_KPAD` shorts per row — so its pad is 4 _halves_, and its bank argument is that
`(BK + PAD)/4` must be odd rather than anything about even banks. Its write path keeps a 2-way conflict
that no pad can remove; the read path, which is the hot one, is clean.

Only the two double-buffered router kernels still stage k-major and unpadded, straight out of
`extern __shared__ unsigned short sm[]`.

Occupancy is steered with `__launch_bounds__`, whose second argument in HIP is the minimum waves per
SIMD, not CUDA's blocks per multiprocessor — and it does not grant occupancy, it _divides the
register budget_, capping each wave at 512/N VGPRs. The conversion to workgroups per CU is the one the
file states next to the logits kernel, `4 * CTA / WARPS_PER_BLOCK`. `NUM_CTA_MLP1 = 2` asks for two
waves per SIMD — eight per CU, that is two 4-wave workgroups — which is exactly what 24 960 B of LDS
permits on a 64 KB CU. `NUM_CTA_MLP2` is 2 for the 128 × 192 × 32 `Mlp2LdsPlan` tile. The workgroup is four waves, so a CTA of 2
asks for `4 * 2 / 4 = 2` workgroups per CU, and what it buys is the register cap: 512/2 = 256 VGPRs per
lane, of which the `3 × 4 × 2` f32x4 accumulators alone take 96. A CTA of 3 (170 VGPRs) measured the same
at 128 steps; 2 is the value the shipped 1024-step number was taken with. The history is worth one
sentence: the 96-row, six-wave tile shipped before this ran best at 3, which is not a power of two and so
was never on the grid of the sweep that tried 2, 4 and 8.

### The only runtime decision is split-K

Two launchers make a runtime choice, and both use the same rule: query the CU count (falling back to
104), and if the two-dimensional grid holds fewer than `4 × CU` = 416 workgroups, split the reduction
axis so that the idle CUs get work.

```c
const int target_cta = cu * 4;
int splits = (grid_xy >= target_cta) ? 1 : (target_cta + grid_xy - 1) / grid_xy;
splits = min(8, max(1, splits));
const int max_splits_by_K = max(1, (K + BK - 1) / BK);
splits = min(splits, max_splits_by_K);
```

For the router this always fires at the shipped batch sizes. With `gx = 1`, 20b gives
`gy = ceil(1536/32) = 48` and so `ceil(416/48) = 9`, clamped to 8; 120b gives `gy = 24`, so 18, again
clamped to 8. `matmul_kernel_nosplit` is therefore dead code in both shipped configurations — with
`gx = 1` it takes $\lceil \text{M}/32 \rceil \ge 416$, that is $\text{M} \ge 13281$, to run. The split-K router allocates an
$\text{M} \times \text{N} \times \text{splits}$ fp32 scratch with `hipMallocAsync` and finishes with
`reduce_splitk_with_bias`, which sums the slices and adds the router bias.

MLP2's split-K is a different mechanism despite the shared heuristic: no scratch allocation and no
reduction kernel. Partials go straight into the pre-allocated `z_partial`, zeroed by a `hipMemsetAsync`
when `splits > 1`, the epilogue does an `atomicAdd` into it, and the `blockIdx.z == 0` slice contributes
the bias exactly once. It no longer engages: `MATMUL_MLP2_MAX_SPLITS = 1` pins `splits` to 1, and the
reason is the one story in this file that took the longest to find. With BM = 128 the row-tile count
is about 48 on 20b, and a layer whose experts hold only 3072 pairs has 24 tiles × 15 = 360 workgroups,
under the 416 threshold — so 4 of the 360 MLP2 launches in a step went split-K while the 96-row build
never did. Their `atomicAdd` partial sums land in whatever order the blocks were scheduled, the
rounding drift crosses a bf16 boundary a few layers later, and the argmax token flips in about 3 % of
positions at 16 steps — which read as "BM = 128 is wrong" for days. Pinned, the kernel is bit-exact
with the 96-row build, and split-K had no speed to sell: 62 865 tok/s with it against 62 899 without.
The `total_pairs` read that sizes the memset is a synchronous `hipMemcpy` now; the old
`hipMemcpyAsync` into a stack `int` only worked because ROCm copies synchronously into pageable memory.

One constraint worth recording: both split-K paths compute `kChunk = ceil(K / splits)` without forcing
any alignment, while the loads are 16-byte vectors guarded by `kk + 7 < kEnd`. At the shipped
`splits = 8` over $\text{K}=2880$ this is exact, `kChunk = 360`. At `splits = 7`, `kChunk = 412` would
silently zero-fill four k values at the tail of every chunk and misalign the bf16 weight load by 8
bytes. `splits = 7` needs `grid_xy` in 60..69, which the shipped batch sizes do not produce.

## Multi-head attention

Attention is one kernel template, `flash_attn_decode_mfma16_kernel`, instantiated twice by
`getp_flash_attn_decode_bf16`: with `APPLY_MASK = true` for the sliding-window layers and
`APPLY_MASK = false` for the full ones. Its name is honest — there is no prefill kernel in this engine.

Two older kernels, `flash_attn_decode_even_layer_bf16_matrix_core_kernel` and
`flash_attn_decode_odd_layer_bf16_matrix_core_kernel`, are still in the file, and both still open with
the same line, `/* Ignore sliding window mechanism, only consider odd layer */`, which is true only of
the second. Neither is in the binary: they are templates that nothing instantiates while `FLASH_MFMA16`
is 1, which is its default — only the `#else` arms of the dispatcher name them. They are kept so the two
designs can still be A/B'd, and the per-parity switches `FLASH_MFMA16_EVEN` and `FLASH_MFMA16_ODD` exist
so a numerical difference can be bisected onto the masked or the unmasked path. Nothing below describes
them unless it says so.

Prompt tokens are teacher-forced through the ordinary decode step, in
[`src/getp/run.cpp`](../src/getp/run.cpp):

```c
if (pos < nums_prompt_tokens[b]) next[b] = prompts_tokens[b][pos];
```

Every forward pass advances every request in the batch by exactly one position. Prompt processing
differs from generation only in which token is fed back; the GPU-sampled token is discarded while the
prompt lasts. So attention is always a one-query-token problem.

The obvious conclusion from that — the kernel must be bound by KV traffic, not by arithmetic — turned
out to be wrong, and the current kernel exists because of it. A drain test on the previous design (keep
the HBM loads, the LDS staging and the barriers, replace the entire computation with a single add) put
56 % of kernel time in the compute half. The cost was not the KV stream; it was what the old score
product made the wave do around each matrix-core instruction.

### Grouped-query attention as a matrix-core enabler

The launch is `dim3 block(64 * (8 / FLASH_HEADS_PER_WAVE))` with `dim3 grid(n_kv_heads, batch_size)` —
one thread block per (KV head, request), which is 8 × 1536 = 12288 blocks per layer per step for 20b.
With `head_dim = 64`, `n_attn_heads = 64` and `n_kv_heads = 8`, the group size is `kv_mul = 8`, and at
the default `FLASH_HEADS_PER_WAVE = 8` the block is a single 64-lane wave that owns all eight query
heads of the group: `hg_self = kv_h * KV_MUL + waveIdx * HPW + (ln < HPW ? ln : 0)`, where the head
rides the _N_ axis of the matrix core, one head per lane column `ln = lane & 15`. The clamp is not
decoration: columns `ln >= HPW` correspond to no head at all, so they are fed a zero Q — finite scores,
no NaN — and never written back.

The obvious payoff is traffic. The 16-key tiles of K and of V staged in LDS are consumed by all eight
query heads, so K and V are fetched from HBM once per group rather than once per query head — an
eightfold reduction on the dominant cost.

The subtler payoff is that GQA is what makes the matrix core usable at all. With one query token per
request, per-query-head attention is a GEMV ($\text{M}=1$) and a matrix core is the wrong instrument.
Grouping eight query heads is what fills the _N_ axis of a 16 × 16 × 16 MFMA: the eight heads of one KV
group become eight of the sixteen B columns, and the M axis is spent on the 16 keys of a tile rather
than on queries. Half the N columns idle, and that is the price of the shape;
`FLASH_HEADS_PER_WAVE = 4` restores the older two-wave split, where only a quarter of the columns are
useful, at twice the MFMA count and twice the loader threads.

![attention score computation](assets/attention.png)

_The diagram is out of date in one respect: it shows the retired split of two waves holding four query
heads each, with the heads as rows of the product. At the shipped `FLASH_HEADS_PER_WAVE = 8` a single
wave holds all eight heads of a KV group, and the heads are the columns._

### Both products on the matrix core

The instruction is `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`, issued twice per 16-element chunk of the
head dimension: first $\text{S}^{\top} = \text{K}\cdot\text{Q}^{\top}$, with the keys of the tile on the
M axis and the query heads on the N axis, then
$\text{O}^{\top} \mathrel{+}= \text{V}^{\top}\cdot\text{P}^{\top}$. With `head_dim = 64` covered by
`NCHUNK = HEAD_DIM / TILE = 4` chunks, a 16-key tile costs 8 matrix-core instructions in total.

The payoff of putting keys on M is that the MFMA's D layout — lane holds `D[4*(lane>>4)+m][lane&15]` —
is identical to its B-operand layout. So $\text{P}^{\top}$ feeds the second product exactly where it
already sits: no transpose, no staging through LDS, and not one cross-lane instruction. The only
cross-lane traffic left per tile is the two `__shfl_xor` that fold the tile maximum across the four
k-groups. That, and not the KV stream, is what the rewrite bought.

Q never touches LDS. Each lane holds `qB[NCHUNK]` bf16x4 registers — its own column of
Q never touches LDS. Each lane holds `qB[NCHUNK]` bf16x4 registers — its own column of
$\text{Q}^{\top}$. With `GETP_Q_BF16` (default 1) the qkv epilogue has already scaled q by
`inv_sqrt_d` and rounded it to bf16, so the kernel copies those bits straight out of `q` and its own
`inv_sqrt_d` argument goes unused; at `GETP_Q_BF16 = 0` it falls back to scaling and rounding from
fp32 once at kernel entry. There is no tiling over the head dimension beyond those four chunks:
`head_dim == 64` is asserted outright.

### The online softmax

Rescaling is done once per 16-key tile, and the output accumulator is `f32x4 acc_o[NCHUNK]` — 16 floats
per lane, one four-wide register per head-dim chunk, held in exactly the layout the second MFMA writes.

```c
float m_tile = fmaxf(fmaxf(st[0], st[1]), fmaxf(st[2], st[3]));
m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 16));
m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 32));

const float m_new = fmaxf(m_run, m_tile);
const float alpha = __expf(m_run - m_new);
```

This is FlashAttention with a tile-wide inner step: a running maximum `m_run`, a running denominator
`l_run`, and an output accumulator rescaled by `alpha` whenever the maximum moves. Nothing of size
$\text{seq}$ is ever materialised, which is the main reason the kernel needs only ~4.3 KB of LDS; the
other is that the epilogue's transpose scratch `ot` aliases the `ks`/`vs` region instead of holding its
own 2 KB, guarded by a `static_assert` that it fits and by a `__syncthreads()` before the first touch.

The two `__shfl_xor` are mandatory rather than an optimisation. Each lane's `st` covers only its own
k-group of the tile, and a group with no valid key would carry `m_tile = -INFINITY` and turn the
subsequent `__expf` into NaN; after the fold every group shares a maximum that is guaranteed finite.
Out-of-range keys are handled the same way — rows of `ks` past the end of the sequence are zero-filled,
which would score 0 and contaminate the sum, so they are forced to `-INFINITY` before the maximum is
taken.

$\text{P}^{\top} = \exp(\text{S}^{\top} - m_{new})$ is rounded to bf16 in place so that it can be the B
operand of the second product, and V is the A operand of it, so both products run on the matrix core;
the VALU is left with the exponentials and the `alpha` rescale of the accumulator. `l_run` is kept per
k-group and folded across groups once in the epilogue, because all four groups share the same `m_new`.
Normalisation is deferred: the division by `l` happens once, in the epilogue, folded into a single
scale factor.

Instruction counting is what the whole shape is for, and it inverts the old one. Per 16-key tile the
wave now runs 8 matrix-core instructions, 2 cross-lane operations and 5 `__expf`, against 4 matrix-core
instructions, roughly 128 cross-lane operations and roughly 80 `__expf` in the retired design. The
softmax is no longer the bottleneck because there is barely any of it left per key.

Numerically the path is: Q scaled and rounded to bf16 in the qkv epilogue rather than here — `GETP_Q_BF16` (default on) has the epilogue write `f32_to_bf16bits(q_scale * o1)`, and the attention kernel loads those bf16 bit patterns directly; `q_scale` is 2^-3, so scaling before the rounding leaves the mantissa untouched and the result bit-identical to rounding first — K and V stored bf16 in the cache,
cache, scores accumulated in fp32 inside the matrix core, `P` rounded to bf16 before the PV product,
output accumulated in fp32, then rounded to bf16 in the lane and written straight to `attn_o_bf16`. The
one new rounding is the one on `P`; `FLASH_P_HILO` (default 0) exists to split `P` into a high and a
residual bf16 and pay one extra MFMA per chunk — four per tile — should that rounding ever prove to be
the cause of a numerical difference.

One initialiser in the _retired_ kernels is worth knowing about before it is discovered by accident,
because the pattern reads like a bug. Their running maximum is declared

```c
float m_values[MFMA_M] = {-INFINITY};
```

which is C aggregate initialisation: element 0 becomes `-INFINITY` and elements 1, 2 and 3 become
`0.0f`. Three of the four query heads in each wave therefore start with a running maximum of zero rather
than −∞. It is harmless as written, because `l_values` starts at 0 and `out_values` at 0: the first real
key gives `alpha = exp(0 - m_new)`, a finite factor applied to accumulators that are still zero, and
after that first update `m` holds a true maximum. Both `l` and `out` pick up the same surplus factor,
which the final division by `l` divides straight back out. It is called out here only so that a
bit-exactness investigation does not start in the wrong place — the same declaration copied into a
kernel that starts from a non-zero accumulator, or that drops the `l`-rescale, would be a real bug. The
shipped kernel has no exposure to it: its running maximum is the single scalar `float m_run =
-INFINITY;`.

### Sinks and the alternating window

gpt-oss adds a learned attention sink per head. It is handled in the epilogue as one extra virtual key
with logit `attn_sinks[hg_self]` and no value — the running maximum, the rescale and the denominator are
updated exactly as for a real key, and the division by `l` is folded into the same factor:

```c
const float sink  = attn_sinks[hg_self];
const float m_fin = fmaxf(m_run, sink);
const float a     = __expf(m_run - m_fin);
const float l     = l_run * a + __expf(sink - m_fin);
const float scale = a / l;
```

`scale` is then applied to the four-wide accumulator, so the sink rescale and the normalisation together
cost one multiply per element. This reproduces the CPU reference, which softmaxes over `pos+2` entries
while summing values only over the real keys.

Sliding-window attention is expressed by _not looking_, rather than by masking. The dispatcher computes
`apply_mask = (sliding_window > 0 && ((layer_id & 1) == 0))`, matching the reference's `l % 2 == 0`
rule, and passes it as the `APPLY_MASK` template argument. The masked instantiation sets
`t_start = MAX(0, pos - sliding_window + 1)` and iterates at most 128 keys; the unmasked one sets
`t_start = MAX(0, pos + 1 - cache_tcap)`, so even the full layers are bounded by the ring buffer rather
than by `pos`. No additive `-INFINITY` mask is ever materialised on the GPU. The two behaviours differ
in exactly one place: that ternary on `APPLY_MASK`. Both instantiations take the identical parameter
list.

### Cache layout and tiling

The KV cache is one bf16 allocation per device, time-major inside each layer: the element address runs
`layer_offset + (t % cache_tcap)`, then batch, then KV head, then head dim. Time capacity differs by
layer class.

| Layer class    | Slots (`cache_tcap`)           | Why                                            |
| -------------- | ------------------------------ | ---------------------------------------------- |
| Even (sliding) | 128, equal to `sliding_window` | a ring sized to exactly what the layer can see |
| Odd (full)     | 1024, equal to `seq_len / 2`   | a ring sized to the memory budget              |

For 20b that is 12 × 128 + 12 × 1024 = 13 824 time slots per device across 24 layers, at 1.5 MiB per
slot for K and the same for V (1536 requests × 512 elements × 2 B). Note the consequence for the full
layers: they hold half the advertised `seq_len` of 2048, so history beyond 1024 positions is dropped
there too.

Putting time outermost makes the per-step write one contiguous `BATCH_SIZE * kv_dim` block — 1.5 MiB. With `GETP_ROPE_FUSED_KV` (default 1) the qkv epilogue writes that block itself, two bf16 per lane, and `kv_store_pair_fp32_to_bf16` is no longer launched: its wide 8-byte-per-lane `uint2` store was traded away to remove the 12.6 MB per layer of fp32 k/v round-trip traffic that path required.
perfectly coalesced by `kv_store_pair_fp32_to_bf16`. The cost is on the read side: consecutive timesteps
for one (request, KV head) are 1.5 MiB apart, so each 16-key tile touches 16 separate 128-byte rows. The
trade is deliberate. Writes happen once per step for every request in the batch, while reads are
amortised over the eight query heads of the group.

Sequence tiling is `TILE = 16`. LDS is one padded region, `kvo[2 * TILE * (HEAD_DIM + FLASH_LDS_PAD)]`,
reinterpreted as `ks` and `vs`: 16 rows of 68 `unsigned short` for K plus the same for V, 4352 bytes per
block at the default `FLASH_LDS_PAD = 4`. Both are kept as raw bf16 _bit patterns_, because both are now
matrix-core operands. The pad of 4 makes a row 136 bytes = 34 banks, so the 16 lanes of a `ds_read_b64`
phase cover the 32 banks exactly once with no conflict; the cost is that a row is then only 8-byte
aligned, so the loader (`put16`) writes two `b64` stores where one `b128` would otherwise do. With
`nbLoadsKV = TILE * (HEAD_DIM/8) / BLOCK_SIZE = 2` at the default 64-thread block, each thread issues
two 16-byte K loads and two V loads per tile.

Double buffering here is register-staged, not LDS-staged: the next tile is prefetched into `regK` and
`regV` before the compute loop and written into the single LDS buffer after it, bracketed by two
`__syncthreads()`. Register state per lane is 16 floats of output accumulator (`acc_o[NCHUNK]`, four
head-dim chunks), the two scalars `m_run` and `l_run`, four `bf16x4` Q registers, and 16 VGPRs of K/V
prefetch — which is what lets a 64-thread block with 4352 B of LDS reach high occupancy, and occupancy
is what hides the KV read latency.

Finished requests cost almost nothing. `if (!mask_on[b]) return;` retires the whole block after a single
global read of the per-request active flag, before any Q, K, V or sink traffic. The flag array is
uploaded each step and cleared when a request emits an end token.

Two limits on generality. The kernel is hard-wired to `head_dim = 64` and `kv_mul = 8`, so the shipped
`gpt-oss-7m` test configuration (head_dim 32, 2 query heads, 1 KV head) would trip the assert. And the
20b/120b split in the dispatch is nominal: `getp_forward_20b` exists but is never called, so both model
sizes run the identical attention path.

## Mixture-of-experts

The MoE block is where a naive implementation loses the most. Routing sends each token to 4 of 32
experts (20b) or 4 of 128 (120b), and the resulting distribution is strongly non-uniform: some experts
receive many tokens, many receive few or none. A kernel launch per expert would leave most of the GPU
idle for most of the block. The approach here is to turn routing into a bucketed row layout once, then
run **one** launch of each expert GEMM across all experts at the same time.

### Routing

Each MoE block starts from the post-attention RMSNorm activation, shape `[B, 2880]`, and scores it
against the router weight with `getp_matmul_router_bf16`. That is an ordinary
$\text{C}=\text{A}\times\text{B}^{\top}$ with $\text{N}$ = expert count and $\text{K}$ = 2880, using the
32 × 32 × 32 tile and `GETP_BN_AGG = 4` described above, with the router bias folded into the epilogue
of the split-K reduction.

Top-k selection is `router_topk_softmax_batch_kernel`: one block per token, scores cached in LDS, and
K sequential selection passes, each a block-wide argmax followed by poisoning the winner with
`scores[topi[sel]] = -INFINITY`. Comparisons use an explicit relative tie-break, and on a near-tie the
_lower_ expert index wins:

```c
float thr = eps * fmaxf(fabsf(v), fabsf(best));
if (v > best + thr || (fabsf(v - best) <= thr && i < besti)) {
  best = v;
  besti = i;
}
```

with `eps = 1e-6f`. This matters because the logits come out of a bf16 matmul, where near-exact
collisions are common; the CPU reference sorts with an unstable comparator, so its ties are arbitrary,
and a deterministic rule here is what makes this kernel's output independent of thread scheduling. It
was not by itself enough to make the engine reproducible — that came from making the expert exchange
a pull, after which two full 8-GPU runs agree on 100 % of output lines, against 50 % to 95 % run pair
by run pair for the push build. The one comparison that
provably _was_ order-dependent has been removed: the logits argmax used to be a relative-tolerance test
inside a CAS loop, which is not a transitive relation, so with thousands of column blocks racing for one
cell the winner depended on arrival order. It is now a single `atomicMax` on a key that packs value and
index monotonically (`pack_val_idx`), which has a total order. Softmax runs _after_
selection and only over the K selected values, matching the reference. `GETP_ROUTER_TOPK_MAXK` (4) sizes
the fixed-length scratch arrays and the shared-memory request; it is a compile-time ceiling on
`experts_per_token`, and all shipped configurations use exactly 4.

The block is sized `min(1024, max(64, 2^ceil(log2(n_experts))))` under `GETP_SMALL_TOPK` (default on):
64 threads for 20b's 32 experts and 128 for 120b's 128, against the fixed 1024 of the `#else` arm.
Shrinking it is bit-exact — every reduction stage with stride >= `n_experts` compares against a
never-filled `(-INFINITY, n_experts)` slot, which the tie-break rule above always rejects, so those
stages are the identity — and it is worth 2.5 ms/step, almost all of it `__syncthreads` barriers across
16 waves and the 7.4 serial block waves that 16-wave occupancy forced.


### From (token, expert) pairs to buckets

The "sorting" is a counting sort over (token, expert) pairs, and it never moves a routing decision
twice.

| Kernel                                       | Does                                                                                                                               |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `map_global_to_local_batch_kernel`           | filters each token's four global expert ids against `[expert_start, expert_end)`, rebases them to local ids, pads the rest with −1 |
| `moe_count_local_kernel`                     | histograms the local expert ids                                                                                                    |
| `exclusive_scan_small_kernel`                | one 256-thread Hillis–Steele scan producing `e_offsets[0..E]`                                                                      |
| `moe_fill_local_pos_kernel`                  | re-counts with `atomicAdd` and writes each pair to slot `offsets[lid] + idx`                                                       |
| `moe_scatter_acts_to_expert_frombf16_kernel` | physically gathers the activations, one 256-thread block per pair                                                                  |

The product is three slot-indexed arrays plus one inverse map: `e_dev[pos]` is the token index,
`w_dev[pos]` is that token's routing weight for that expert, and `pair_pos[b*K+k]` is the slot, used
later for the gather (non-local choices get −1). Each expert's input is then a contiguous
`[n_e, 2880]` matrix at `a_in + offsets[e]*H` — precisely what a blocked GEMM wants.

The single-block scan caps experts per device at 256. Both models need 16 — 32 experts at EP = 2, 128 at
EP = 8 — so there is headroom.

### One kernel for all experts

`build_moe_block_schedule` computes `blk_counts[e] = ceil(n_e / BM)` for the caller's `BM` and
exclusive-scans it, so `blk_offsets[E]` is the total number of 128-row tiles over all experts. Both
expert GEMMs then launch exactly once, with `grid.y` set not to that total — reading it back would
drain the stream — but to the host-computable upper bound `(max_pairs + BM − 1) / BM + E`, which the
per-expert rounding can at most reach; workgroups past the real total find `base_all >= n` and exit
before touching memory. Inside the kernel, each workgroup recovers its own (expert, tile) pair by
binary search:

```c
int by = blockIdx.y;
int lo = 0, hi = E;
while (lo + 1 < hi) { int mid = (lo + hi) >> 1; int off = blk_offs[mid]; if (by < off) hi = mid; else lo = mid; }
int e = lo;
int blk = by - blk_offs[e];
```

Four iterations at 16 experts per device. There is no host-side loop over experts, no launch per expert,
and the grid does not depend on how skewed the routing is. An expert with zero tokens contributes zero
blocks, so `blk_offsets[e] == blk_offsets[e+1]`, and the search — which advances `lo` whenever
`by >= offs[mid]` — maintains `blk_offs[lo] <= by < blk_offs[lo+1]` and therefore can never land on an
empty expert. Load imbalance costs at most one partially filled tile per non-empty expert — 128 rows for
MLP1 and 128 for MLP2 — instead of a serialized launch per expert.

MLP2 decodes `blk_offsets` with its own `BM`, so the two kernels can share a schedule only while their
block heights agree. They do again, 128 and 128, so the `#else` arm of
`#if MATMUL_MLP2_BLOCK_ROWS != MATMUL_MLP1_BLOCK_ROWS` runs and one scan serves both launches. Change
either height and the `#if` arm builds MLP2 a second schedule into the same buffers, which is safe
because everything is on `compute_stream` and MLP1's launch has already read them.

### The fused expert GEMMs

**MLP1** computes gate, up and the activation in one pass. It reads the interleaved weight rows `2*col`
and `2*col+1` directly, keeps the two accumulator banks, issues two matrix-core instructions per
activation fragment, pulls the gate and up biases in a single 32-bit load of the adjacent bf16 pair, and
finishes with:

```c
float gval = fminf(fmaxf(dg_acc[g][im][in][i] + bg_loc[g][in], swiglu_limit_neg), swiglu_limit);
float uval = fminf(fmaxf(du_acc[g][im][in][i] + bu_loc[g][in], swiglu_limit_neg), swiglu_limit);

const float sgm = 1.f / (1.f + __expf(-swiglu_a * gval));
const float out = (gval * sgm) * (uval + 1.f);

output_row[col] = f32_to_bf16bits(out);
```

with `swiglu_a = 1.702f` and `swiglu_limit = 7.0` from the checkpoint configuration. Writing bf16 here
is what lets MLP2 consume the intermediate at half the width with no conversion pass. One deviation from
the CPU reference is worth recording: the reference clamps the gate only from above, while the kernel
clamps it symmetrically. For gate values far below −7 the silu terms differ by at most
|silu(−7)| ≈ 4.7e-5, and after the `(up + 1)` factor the output by at most about 3.7e-4 — negligible,
but real.

**MLP2** is the down projection, and it is where the routing weight is applied: the epilogue scales
every output element by `w_by_bucket[pos]`. That is why the token index is passed to the kernel but
never read in its body, and why the final gather is a pure sum. `moe_gather_pairs_kernel` adds at most
four `z_partial` rows per token through `pair_pos`, skipping −1.

Both expert GEMMs run the K loop 2880 / 32 = 90 times per output tile. Along the output axis MLP1
covers its 2880 channels in $\lceil 2880/128 \rceil = 23$ column tiles — the 2 × 2880 in the GEMM table
above is the weight-row count, not this one — and MLP2 covers its 2880 in exactly 2880 / 192 = 15.

### Costs worth knowing

The design is not free, and the costs concentrate in two places.

**Host synchronization.** Each MoE layer used to pay three device-to-host reads: `h_off` inside
`build_moe_buckets_local_pos`, `total_pairs` before the activation scatter, and `total_blocks` inside
`build_moe_block_schedule` — each a drain of `compute_stream` in the middle of the layer. None remains.
The first only ever produced `cap_pairs`, which was assigned and never read on any live path. The other
two sized launch grids from data-dependent bucket counts; both grids are now sized from the
host-constant worst case `EXPERT_PARALLELISM × BATCH_SIZE × experts_per_token` — the scatter reads
`e_offsets[E]` on the device and returns for `pos` beyond it, and the block schedule returns
`(max_pairs + BM − 1) / BM + E` (see above). The surplus workgroups exit at their first branch; what
the device gains is a queue that never empties between kernels, measured at +1.0 % of throughput at
1024 steps (63 444 against 62 793 tok/s, two interleaved pairs) with bit-identical output. It also
retires a hazard: the old `total_pairs` copy was `hipMemcpyAsync` into a stack `int` with nothing
waiting on it, correct only because the block-schedule call in between happened to end in a
`hipStreamSynchronize`. Two reads of that shape are left. One is in
`launch_mlp2_partial_bf16_bucketed_frombf16`, now a synchronous `hipMemcpy`, on a branch that needs
`splits > 1` and that `MATMUL_MLP2_MAX_SPLITS = 1` never takes. The other is `GETP_ROWSKIP`'s
per-layer read of the per-rank non-zero row counts, which does run - on 120b and any
`EXPERT_PARALLELISM > 2`, since the path disables itself at EP <= 2. It costs nothing only because
the worker sync immediately before it has already called `hipStreamSynchronize`; see
[PARALLELISM.md](PARALLELISM.md) for why the byte count has to reach the host at all.

**Worst-case allocation.** All pair-indexed buffers are sized for
`EXPERT_PARALLELISM × BATCH_SIZE × experts_per_token`, the case where every token routes all four of its
choices to local experts.

| Buffer                | 20b (12 288 slots) | 120b (24 576 slots) |
| --------------------- | ------------------ | ------------------- |
| `z_partial` (fp32)    | 141.6 MB           | 283.1 MB            |
| `a_in` (bf16)         | 70.8 MB            | 141.6 MB            |
| `gate_up_bf16` (bf16) | 70.8 MB            | 141.6 MB            |

Expected live occupancy is roughly 6144 pairs for 20b (16 of 32 experts local) and 3072 for 120b (16 of
128), so most of that is insurance against a routing skew that would otherwise overrun the buckets.

## Building these kernels

Every figure on this page — tiles, LDS budgets, occupancy, and the throughput numbers quoted in
[`SERVING.md`](SERVING.md) — assumes the `runfast` target in the [`Makefile`](../Makefile):

```make
CFLAGS = --std=c++17 -lm $(INCLUDES)     # + --offload-arch=gfx90a, unless CC is g++

runfast: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -O3 -o run $(CPP_FILES)
```

That is what `./run.sh build` selects: its default mode is `fast`. `rundebug` and `runomp` carry the
same `$(CFLAGS)`, at `-g` and at `-O3 -fopenmp -march=native`.

A bare `make` does not. `run` is the default target _and_ the only recipe that never expands
`$(CFLAGS)`:

```make
run: $(CPP_FILES) tokenizer-bin
	$(CC) $(INCLUDES) -g -O0 -o run $(CPP_FILES)
```

so it compiles at `-O0`, without `--offload-arch=gfx90a` and without `--std=c++17`. It still produces a
binary — hipcc falls back to its own default architecture — but none of the reasoning above survives it.
The tile choices are argued against a `gfx90a` GCD (64-lane waves, 64 KB of LDS, 104 CUs), the `_1k`
matrix-core form is CDNA2, and `-O0` leaves the register pipeline that the double-buffering section
describes entirely at the compiler's mercy. Build with `runfast`, or with `./run.sh build`, before
measuring anything.

There is no CPU-only build, despite two places in the tooling that suggest otherwise. The Makefile picks
`CC := $(shell command -v hipcc 2>/dev/null || echo g++)`, and the `build` path in
[`run.sh`](../run.sh) prints
`warning: hipcc not found, falling back to g++ (CPU only)`. But [`src/run.cpp`](../src/run.cpp)
includes `getp/run.cpp` unconditionally, and [`src/getp/run.cpp`](../src/getp/run.cpp) includes
`hip/forward.hip` the same way — there is no `#ifdef` on either. Every `__global__` kernel on this page
is therefore part of the one translation unit that the fallback `g++` would have to compile, and it
cannot. The warning describes a configuration that does not exist; without hipcc the build fails
outright rather than degrading to a CPU path.

## Where to look in the code

| Concept                                                                                                                                                                                                    | File                                                                                                       |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| All GEMM, attention and MoE kernels                                                                                                                                                                        | [`src/hip/forward.hip`](../src/hip/forward.hip)                                                            |
| Tile macros: `MATMUL_*`, `NUM_CTA_*`, `GETP_BN_AGG` (no `MATMUL_QKV_*` exists)                                                                                                                             | [`src/hip/forward.hip`](../src/hip/forward.hip), defined next to each kernel                               |
| Attention switches: `FLASH_MFMA16` (and the per-parity `FLASH_MFMA16_EVEN` / `_ODD`), `FLASH_HEADS_PER_WAVE`, `FLASH_LDS_PAD`, `FLASH_P_HILO`; `FLASH_DECODE_TILE_T` now sizes only the retired kernels    | [`src/hip/forward.hip`](../src/hip/forward.hip)                                                            |
| LDS padding, one macro per staging layout: `MATMUL_MLP1_LDS_PAD_SLOTS`, `MATMUL_MLP2_LDS_PAD_SLOTS`, `GETP_QKV_LDS_PAD_SLOTS`, `MATMUL_ATTN_O_LDS_PAD_SLOTS` (cells) and `MATMUL_LOGITS_LDS_KPAD` (halves) | [`src/hip/forward.hip`](../src/hip/forward.hip)                                                            |
| Batch size, expert parallelism, per-device request slicing                                                                                                                                                 | [`src/getp/run.cpp`](../src/getp/run.cpp)                                                                  |
| KV cache sizing and per-layer ring capacities                                                                                                                                                              | [`src/getp/transformer.cpp`](../src/getp/transformer.cpp)                                                  |
| MoE scratch buffers: `z_partial`, `a_in`, `gate_up_bf16`, `pair_pos`                                                                                                                                       | [`src/getp/state_ext.cpp`](../src/getp/state_ext.cpp), [`include/state_ext.hpp`](../include/state_ext.hpp) |
| CPU reference for sinks, SwiGLU and top-k                                                                                                                                                                  | [`src/run.cpp`](../src/run.cpp)                                                                            |
| Build targets and flags; `runfast` is the one to measure with                                                                                                                                              | [`Makefile`](../Makefile), [`run.sh`](../run.sh)                                                           |
| Model dimensions quoted above                                                                                                                                                                              | [`tools/model_export/`](../tools/model_export/) `config.json`                                              |
| Expert × Data parallelism and the collectives                                                                                                                                                              | [`PARALLELISM.md`](PARALLELISM.md)                                                                         |
