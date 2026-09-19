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
  // Ban thu hai cua ext_e_agg, dung xen ke theo parity lop. Peer keo lat cat
  // cua no tu buffer nay bang hipMemcpyPeerAsync tren memory_stream, va khong
  // co gi chan gather cua lop ke tiep ghi de len khi ban sao chua xong. Hai
  // buffer + barrier moi lop la du: gather(L+2) chi chay sau khi moi peer da
  // qua barrier cua lop L+1, tuc sau khi ban sao cua lop L da hoan tat.
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
