#include "getp_state_ext.hpp"

#ifndef HIP_CHECK
#define HIP_CHECK(expr) do { hipError_t _e = (expr); if (_e != hipSuccess) { \
  fprintf(stderr, "HIP error %d (%s) at %s:%d\n", _e, hipGetErrorString(_e), __FILE__, __LINE__); \
} } while(0)
#endif

static RunStateExt* g_ext = nullptr;
static int g_ext_n = 0;

void ext_create(int n_devices) {
  g_ext_n = n_devices;
  g_ext = (RunStateExt*)malloc(sizeof(RunStateExt) * n_devices);
  for (int i = 0; i < n_devices; ++i) {
    g_ext[i].local_ids = nullptr;
    g_ext[i].local_wts = nullptr;
    g_ext[i].n_local   = nullptr;
  }
}

void ext_alloc_device(int device_index, int batch_size, int k_per_tok) {
  HIP_CHECK(hipSetDevice(device_index));
  HIP_CHECK(hipMalloc(&g_ext[device_index].local_ids, sizeof(int)   * (size_t)batch_size * k_per_tok));
  HIP_CHECK(hipMalloc(&g_ext[device_index].local_wts, sizeof(float) * (size_t)batch_size * k_per_tok));
  HIP_CHECK(hipMalloc(&g_ext[device_index].n_local,   sizeof(int)   * (size_t)batch_size));
}

void ext_free_all(int n_devices) {
  for (int i = 0; i < n_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    if (g_ext[i].local_ids) HIP_CHECK(hipFree(g_ext[i].local_ids));
    if (g_ext[i].local_wts) HIP_CHECK(hipFree(g_ext[i].local_wts));
    if (g_ext[i].n_local)   HIP_CHECK(hipFree(g_ext[i].n_local));
  }
  free(g_ext);
  g_ext = nullptr; g_ext_n = 0;
}

RunStateExt* ext_get(int device_index) {
  return &g_ext[device_index];
}
