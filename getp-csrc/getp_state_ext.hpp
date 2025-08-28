#pragma once
#include <hip/hip_runtime.h>
#include <cstdlib>

struct RunStateExt {
  int   *local_ids;   // [BATCH_SIZE * K]
  float *local_wts;   // [BATCH_SIZE * K]
  int   *n_local;     // [BATCH_SIZE]
};

void ext_create(int n_devices);
void ext_alloc_device(int device_index, int batch_size, int k_per_tok);
void ext_free_all(int n_devices);
RunStateExt* ext_get(int device_index);
