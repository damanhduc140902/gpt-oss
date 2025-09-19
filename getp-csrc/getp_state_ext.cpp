#include "getp_state_ext.hpp"

#ifndef HIP_CHECK
#define HIP_CHECK(expr) do { hipError_t _e = (expr); if (_e != hipSuccess) { \
  fprintf(stderr, "HIP error %d (%s) at %s:%d\n", _e, hipGetErrorString(_e), __FILE__, __LINE__); \
} } while(0)
#endif

static RunStateExt* g_ext = nullptr;

void ext_create(int n_devices) {
  g_ext = (RunStateExt*)malloc(sizeof(RunStateExt) * n_devices);
  for (int i = 0; i < n_devices; ++i) {
    g_ext[i] = {};
  }
}

void ext_alloc_device(int device_index, int batch_size, int k_per_tok,
                      int n_experts, int hidden_dim, int intermediate_dim) {
  HIP_CHECK(hipSetDevice(device_index));
  size_t pairs_cap = (size_t)batch_size * k_per_tok;

  HIP_CHECK(hipMalloc(&g_ext[device_index].local_ids, sizeof(int) * pairs_cap));
  HIP_CHECK(hipMalloc(&g_ext[device_index].local_wts, sizeof(float) * pairs_cap));
  HIP_CHECK(hipMalloc(&g_ext[device_index].n_local, sizeof(int) * batch_size));
  HIP_CHECK(hipMalloc(&g_ext[device_index].e_counts, sizeof(int) * n_experts));
  HIP_CHECK(hipMalloc(&g_ext[device_index].e_offsets, sizeof(int) * (n_experts + 1)));
  HIP_CHECK(hipMalloc(&g_ext[device_index].e_dev, sizeof(int) * pairs_cap));
  HIP_CHECK(hipMalloc(&g_ext[device_index].w_dev, sizeof(float) * pairs_cap));
  HIP_CHECK(hipMalloc(&g_ext[device_index].pair_pos, sizeof(int) * pairs_cap));
  HIP_CHECK(hipMalloc(&g_ext[device_index].z_partial, sizeof(float) * pairs_cap * hidden_dim));
  HIP_CHECK(hipMalloc(&g_ext[device_index].blk_counts, sizeof(int) * n_experts));
  HIP_CHECK(hipMalloc(&g_ext[device_index].blk_offsets, sizeof(int) * (n_experts + 1)));

  HIP_CHECK(hipMalloc(&g_ext[device_index].a_in, sizeof(__hip_bfloat16) * pairs_cap * hidden_dim));
  HIP_CHECK(hipMalloc(&g_ext[device_index].gate_up_bf16, sizeof(__hip_bfloat16) * pairs_cap * intermediate_dim));
}

void ext_free_all(int n_devices) {
  for (int i = 0; i < n_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    if (g_ext[i].local_ids) HIP_CHECK(hipFree(g_ext[i].local_ids));
    if (g_ext[i].local_wts) HIP_CHECK(hipFree(g_ext[i].local_wts));
    if (g_ext[i].n_local) HIP_CHECK(hipFree(g_ext[i].n_local));
    if (g_ext[i].e_counts) HIP_CHECK(hipFree(g_ext[i].e_counts));
    if (g_ext[i].e_offsets) HIP_CHECK(hipFree(g_ext[i].e_offsets));
    if (g_ext[i].e_dev) HIP_CHECK(hipFree(g_ext[i].e_dev));
    if (g_ext[i].w_dev) HIP_CHECK(hipFree(g_ext[i].w_dev));
    if (g_ext[i].pair_pos) HIP_CHECK(hipFree(g_ext[i].pair_pos));
    if (g_ext[i].z_partial) HIP_CHECK(hipFree(g_ext[i].z_partial));
    if (g_ext[i].blk_counts) HIP_CHECK(hipFree(g_ext[i].blk_counts));
    if (g_ext[i].blk_offsets) HIP_CHECK(hipFree(g_ext[i].blk_offsets));
    if (g_ext[i].a_in) HIP_CHECK(hipFree(g_ext[i].a_in));
    if (g_ext[i].gate_up_bf16) HIP_CHECK(hipFree(g_ext[i].gate_up_bf16));
  }
  free(g_ext);
  g_ext = nullptr;
}


RunStateExt* ext_get(int device_index) { return &g_ext[device_index]; }
