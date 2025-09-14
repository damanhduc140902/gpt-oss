#pragma once
#include <hip/hip_runtime.h>

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

  int *ext_topk_i;
  float *ext_topk_v;
  float *ext_router_score;
  float *ext_gate_up;
  float *ext_t;
  float *ext_e_agg;
};

void ext_create(int n_devices);
void ext_alloc_device(int device_index, int batch_size, int expert_parallelism, Config *p);
void ext_free_all(int n_devices);
RunStateExt *ext_get(int device_index);
