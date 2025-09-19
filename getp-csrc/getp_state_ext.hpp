#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdlib>

struct RunStateExt {
  int *local_ids;
  float *local_wts;
  int *n_local;
  int *e_counts;
  int *e_offsets;
  int *e_dev;
  float *w_dev;
  int *pair_pos;
  float *z_partial;
  int *blk_counts;
  int *blk_offsets;
  __hip_bfloat16 *a_in;
  __hip_bfloat16 *gate_up_bf16;
};

void ext_create(int n_devices);
void ext_alloc_device(int device_index, int batch_size, int k_per_tok,
                      int n_experts, int hidden_dim, int intermediate_dim);
void ext_free_all(int n_devices);
RunStateExt *ext_get(int device_index);
