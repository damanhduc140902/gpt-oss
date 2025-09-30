#include "getp_state_ext.hpp"
#include "getp_transformer.cpp"
#include <cstring>

#ifndef HIP_CHECK
#define HIP_CHECK(expr) do { hipError_t _e = (expr); if (_e != hipSuccess) { \
  fprintf(stderr, "HIP error %d (%s) at %s:%d\n", _e, hipGetErrorString(_e), __FILE__, __LINE__); \
} } while(0)
#endif

static RunStateExt* g_ext = nullptr;

void ext_create(int n_devices) {
  g_ext = (RunStateExt*)malloc(sizeof(RunStateExt) * n_devices);
  for (int i = 0; i < n_devices; ++i) {
    memset(&g_ext[i], 0, sizeof(RunStateExt));
  }
}

void ext_alloc_device(int device_index, int batch_size, int expert_parallelism, Config *p) {
  int experts_per_token = p->experts_per_token;
  int n_experts = p->n_experts;
  int hidden_dim = p->hidden_dim;
  int intermediate_dim = p->intermediate_dim;

  HIP_CHECK(hipSetDevice(device_index));

  HIP_CHECK(hipMalloc(&g_ext[device_index].mask_on, sizeof(int) * batch_size));

  HIP_CHECK(hipMalloc(&g_ext[device_index].local_ids, sizeof(int)   * (size_t)expert_parallelism * batch_size * experts_per_token));
  HIP_CHECK(hipMalloc(&g_ext[device_index].local_wts, sizeof(float) * (size_t)expert_parallelism * batch_size * experts_per_token));
  HIP_CHECK(hipMalloc(&g_ext[device_index].n_local,   sizeof(int)   * (size_t)expert_parallelism * batch_size));
  HIP_CHECK(hipMalloc(&g_ext[device_index].e_counts,  sizeof(int)   * (size_t)n_experts));
  HIP_CHECK(hipMalloc(&g_ext[device_index].e_offsets, sizeof(int)   * ((size_t)n_experts + 1)));
  HIP_CHECK(hipMalloc(&g_ext[device_index].e_dev,     sizeof(int)   * (size_t)expert_parallelism * batch_size * experts_per_token));
  HIP_CHECK(hipMalloc(&g_ext[device_index].w_dev,     sizeof(float) * (size_t)expert_parallelism * batch_size * experts_per_token));
  HIP_CHECK(hipMalloc(&g_ext[device_index].pair_pos,  sizeof(int)   * (size_t)expert_parallelism * batch_size * experts_per_token));
  HIP_CHECK(hipMalloc(&g_ext[device_index].z_partial, sizeof(float) * (size_t)expert_parallelism * batch_size * experts_per_token * hidden_dim));

  if (expert_parallelism > 1) {
    HIP_CHECK(hipMalloc(&g_ext[device_index].ext_t_bf16,
      sizeof(__hip_bfloat16) *
      (size_t)expert_parallelism * batch_size * hidden_dim));
    HIP_CHECK(hipMalloc(&g_ext[device_index].ext_topk_i,       sizeof(int)   * (size_t)expert_parallelism * batch_size * experts_per_token));
    HIP_CHECK(hipMalloc(&g_ext[device_index].ext_topk_v,       sizeof(float) * (size_t)expert_parallelism * batch_size * experts_per_token));
    HIP_CHECK(hipMalloc(&g_ext[device_index].ext_router_score, sizeof(float) * (size_t)expert_parallelism * batch_size * n_experts));
    HIP_CHECK(hipMalloc(&g_ext[device_index].ext_t,            sizeof(float) * (size_t)expert_parallelism * batch_size * hidden_dim));
    HIP_CHECK(hipMalloc(&g_ext[device_index].ext_e_agg,        sizeof(float) * (size_t)expert_parallelism * batch_size * hidden_dim));
    HIP_CHECK(hipMalloc(&g_ext[device_index].peer_e_agg,       sizeof(float) * (size_t)expert_parallelism * batch_size * hidden_dim));
  } else {
    g_ext[device_index].ext_topk_i = nullptr;
    g_ext[device_index].ext_topk_v = nullptr;
    g_ext[device_index].ext_router_score = nullptr;
    g_ext[device_index].ext_t = nullptr;
    g_ext[device_index].ext_e_agg = nullptr;
    g_ext[device_index].peer_e_agg = nullptr;
    g_ext[device_index].ext_t_bf16 = nullptr;
  }

  HIP_CHECK(hipMalloc(&g_ext[device_index].blk_counts,  sizeof(int) * (size_t)n_experts));
  HIP_CHECK(hipMalloc(&g_ext[device_index].blk_offsets, sizeof(int) * (size_t)(n_experts + 1)));

  HIP_CHECK(hipMalloc(&g_ext[device_index].a_in,         sizeof(__hip_bfloat16) * (size_t)expert_parallelism * batch_size * experts_per_token * hidden_dim));
  HIP_CHECK(hipMalloc(&g_ext[device_index].gate_up_bf16, sizeof(__hip_bfloat16) * (size_t)expert_parallelism * batch_size * experts_per_token * intermediate_dim));
  HIP_CHECK(hipMalloc(&g_ext[device_index].pre_qkv_bf16, sizeof(__hip_bfloat16) * (size_t)batch_size * hidden_dim));
  HIP_CHECK(hipMalloc(&g_ext[device_index].attn_o_bf16,
                      sizeof(__hip_bfloat16) * (size_t)batch_size *
                          (size_t)p->head_dim * (size_t)p->n_attn_heads));

  // Pre-allocate temporary buffer for collective operations
  // Allocate enough for typical allreduce operations
  size_t max_allreduce_size = (size_t)batch_size * (size_t)hidden_dim * 2;
  g_ext[device_index].allreduce_tmp_size = max_allreduce_size;
  HIP_CHECK(hipMalloc(&g_ext[device_index].allreduce_tmp, 
                      sizeof(float) * max_allreduce_size));
}


void ext_free_all(int n_devices) {
  for (int i = 0; i < n_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    
    if (g_ext[i].mask_on) HIP_CHECK(hipFree(g_ext[i].mask_on));


    if (g_ext[i].local_ids) HIP_CHECK(hipFree(g_ext[i].local_ids));
    if (g_ext[i].local_wts) HIP_CHECK(hipFree(g_ext[i].local_wts));
    if (g_ext[i].n_local)   HIP_CHECK(hipFree(g_ext[i].n_local));
    if (g_ext[i].e_counts)  HIP_CHECK(hipFree(g_ext[i].e_counts));
    if (g_ext[i].e_offsets) HIP_CHECK(hipFree(g_ext[i].e_offsets));
    if (g_ext[i].e_dev)     HIP_CHECK(hipFree(g_ext[i].e_dev));
    if (g_ext[i].w_dev)     HIP_CHECK(hipFree(g_ext[i].w_dev));
    if (g_ext[i].pair_pos)  HIP_CHECK(hipFree(g_ext[i].pair_pos));
    if (g_ext[i].z_partial) HIP_CHECK(hipFree(g_ext[i].z_partial));

    if (g_ext[i].ext_topk_i)        HIP_CHECK(hipFree(g_ext[i].ext_topk_i));
    if (g_ext[i].ext_topk_v)        HIP_CHECK(hipFree(g_ext[i].ext_topk_v));
    if (g_ext[i].ext_router_score)  HIP_CHECK(hipFree(g_ext[i].ext_router_score));
    // if (g_ext[i].ext_gate_up)       HIP_CHECK(hipFree(g_ext[i].ext_gate_up));
    if (g_ext[i].ext_t)             HIP_CHECK(hipFree(g_ext[i].ext_t));
    if (g_ext[i].ext_e_agg)         HIP_CHECK(hipFree(g_ext[i].ext_e_agg));
    if (g_ext[i].peer_e_agg)        HIP_CHECK(hipFree(g_ext[i].peer_e_agg));

    if (g_ext[i].blk_counts)  HIP_CHECK(hipFree(g_ext[i].blk_counts));
    if (g_ext[i].blk_offsets) HIP_CHECK(hipFree(g_ext[i].blk_offsets));

    if (g_ext[i].a_in)          HIP_CHECK(hipFree(g_ext[i].a_in));
    if (g_ext[i].gate_up_bf16)  HIP_CHECK(hipFree(g_ext[i].gate_up_bf16));
    if (g_ext[i].pre_qkv_bf16)  HIP_CHECK(hipFree(g_ext[i].pre_qkv_bf16));
    if (g_ext[i].attn_o_bf16)   HIP_CHECK(hipFree(g_ext[i].attn_o_bf16));
    if (g_ext[i].ext_t_bf16) HIP_CHECK(hipFree(g_ext[i].ext_t_bf16));
    if (g_ext[i].allreduce_tmp) HIP_CHECK(hipFree(g_ext[i].allreduce_tmp));

  }
  free(g_ext);
  g_ext = nullptr;
}


RunStateExt* ext_get(int device_index) { return &g_ext[device_index]; }
