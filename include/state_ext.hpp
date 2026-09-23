#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdlib>

struct RunStateExt {
  int *mask_on;

  int *local_ids;
  float *local_wts;
  int *n_local;
  int *e_counts;
  int *e_offsets;
  int *e_dev;
  float *w_dev;
  int *pair_pos;
  float *z_partial;

  int *ext_topk_i;
  float *ext_topk_v;
  float *ext_router_score;
  // float *ext_gate_up;
  float *ext_t;
  float *ext_e_agg;
  // A second copy of ext_e_agg, alternated by layer parity. A peer pulls its slice out of this
  // buffer with hipMemcpyPeerAsync on memory_stream, and nothing stops the next layer's gather
  // from overwriting it while that copy is still in flight. Two buffers plus the per-layer
  // barrier are enough: gather(L+2) only runs after every peer has passed the layer L+1
  // barrier, i.e. after layer L's copy has completed.
  float *ext_e_agg2;
  float *peer_e_agg;

  int *blk_counts;
  int *blk_offsets;
  __hip_bfloat16 *a_in;
  __hip_bfloat16 *gate_up_bf16;
  __hip_bfloat16 *pre_qkv_bf16;
  __hip_bfloat16 *attn_o_bf16;
  __hip_bfloat16 *ext_t_bf16;

  // Pre-allocated temporary buffers for collective operations
  float *allreduce_tmp;
  size_t allreduce_tmp_size;
};

void ext_create(int n_devices);
void ext_alloc_device(int device_index, int batch_size, int expert_parallelism,
                      Config *p);
void ext_free_all(int n_devices);
RunStateExt *ext_get(int device_index);
