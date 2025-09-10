#pragma once
#include "getp_transformer.hpp"

#include <cassert>
#include <hip/hip_runtime.h>

#include <cmath>
#include <iostream>

#include "getp_eval.cpp"

#define HIP_CHECK(expression)                                                  \
  {                                                                            \
    const hipError_t status = expression;                                      \
    if (status != hipSuccess) {                                                \
      std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
    }                                                                          \
  }

__global__ void init_mask_kernel(float *mask, int seq_len, int sliding_window) {
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= seq_len || j >= seq_len) return;
  mask[i * seq_len + j] =
      (sliding_window > 0 && i - j >= sliding_window ? -INFINITY : 0);
}

void free_device_run_state(RunState *s);

DeviceTransformer::~DeviceTransformer() {
  HIP_CHECK(hipSetDevice(device_index));
  if (dev_data) HIP_CHECK(hipFree(dev_data));
  if (dev_experts) HIP_CHECK(hipFree(dev_experts));
  if (dev_linear_bf16) HIP_CHECK(hipFree(dev_linear_bf16));
  free_device_run_state(&state);
}

void getp_memcpy_fp32_to_bf16(__hip_bfloat16 *dst, float *src, int n_elements) {
  __hip_bfloat16 *tmp = reinterpret_cast<__hip_bfloat16 *>(
      malloc(sizeof(__hip_bfloat16) * n_elements));
  for (int i = 0; i < n_elements; ++i) tmp[i] = __float2bfloat16(src[i]);
  HIP_CHECK(hipMemcpy(dst, tmp, sizeof(__hip_bfloat16) * n_elements,
                      hipMemcpyHostToDevice));
  free(tmp);
}

static void upload_weights(TransformerWeights *w,
                           DeviceTransformerWeights *dev_w, Config *cfg,
                           float **_dev_data, __hip_bfloat16 **_dev_experts,
                           __hip_bfloat16 **_dev_linear_bf16,
                           GPUWorker *worker, int device_index) {
  int head_dim = cfg->head_dim;
  int n_layers = cfg->n_layers;
  int n_experts = cfg->n_experts;
  int intermediate_dim = cfg->intermediate_dim;
  int hidden_dim = cfg->hidden_dim;

  float *weights_begin = w->token_embedding_table;
  float *weights_end = w->b_mlp2 + 1ll * n_layers * n_experts * cfg->hidden_dim;
  ssize_t weights_size = (weights_end - weights_begin) * sizeof(float);

  float *experts_begin = w->w_mlp1;
  float *experts_end = weights_end;
  ssize_t experts_size = (experts_end - experts_begin) * sizeof(float);

  HIP_CHECK(hipSetDevice(device_index));
  {
    HIP_CHECK(hipMalloc(_dev_data, weights_size - experts_size));
    HIP_CHECK(hipMemcpy(*_dev_data, w->token_embedding_table,
                        weights_size - experts_size, hipMemcpyHostToDevice));
    float *ptr = *_dev_data;
    dev_w->token_embedding_table = ptr;
    ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
    dev_w->out = ptr;
    ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
    dev_w->rms_attn_w = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim;
    dev_w->rms_ffn_w = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim;
    dev_w->rms_out_w = ptr;
    ptr += 1ll * cfg->hidden_dim;
    dev_w->w_qkv = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim *
           (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
    dev_w->b_qkv = ptr;
    ptr += 1ll * n_layers *
           (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
    dev_w->w_o = ptr;
    ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;
    dev_w->b_o = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim;
    dev_w->attn_sinks = ptr;
    ptr += 1ll * n_layers * cfg->n_attn_heads;
    dev_w->w_router = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim * n_experts;
    dev_w->b_router = ptr;
    ptr += 1ll * n_layers * n_experts;
  }

  size_t elems_w_qkv =
      1ll * n_layers * hidden_dim *
      (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  size_t elems_b_qkv =
      1ll * n_layers *
      (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  size_t elems_w_o =
      1ll * n_layers * hidden_dim * (head_dim * cfg->n_attn_heads);
  size_t elems_b_o = 1ll * n_layers * hidden_dim;
  size_t elems_w_router = 1ll * n_layers * hidden_dim * n_experts;
  size_t elems_b_router = 1ll * n_layers * n_experts;
  size_t elems_out = 1ll * cfg->vocab_size * hidden_dim;

  size_t linear_elems = elems_w_qkv + elems_b_qkv + elems_w_o + elems_b_o +
                        elems_w_router + elems_b_router + elems_out;

  HIP_CHECK(hipMalloc(_dev_linear_bf16, linear_elems * sizeof(__hip_bfloat16)));
  __hip_bfloat16 *p16 = *_dev_linear_bf16;

  dev_w->w_qkv_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->w_qkv, elems_w_qkv);
  p16 += elems_w_qkv;
  dev_w->b_qkv_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->b_qkv, elems_b_qkv);
  p16 += elems_b_qkv;
  dev_w->w_o_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->w_o, elems_w_o);
  p16 += elems_w_o;
  dev_w->b_o_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->b_o, elems_b_o);
  p16 += elems_b_o;
  dev_w->w_router_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->w_router, elems_w_router);
  p16 += elems_w_router;
  dev_w->b_router_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->b_router, elems_b_router);
  p16 += elems_b_router;
  dev_w->out_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->out, elems_out);
  p16 += elems_out;

  int experts_per_device = worker->expert_end - worker->expert_start;
  HIP_CHECK(hipMalloc(_dev_experts, (experts_size / sizeof(float)) *
                                        sizeof(__hip_bfloat16) / n_experts * 
                                        experts_per_device));
  __hip_bfloat16 *ptr16 = *_dev_experts;

  dev_w->w_mlp1 = ptr16;
  for (int l = 0; l < n_layers; ++l) {
    getp_memcpy_fp32_to_bf16(
        ptr16,
        w->w_mlp1 + 1ll * l * n_experts * 2 * intermediate_dim * hidden_dim +
            1ll * worker->expert_start * 2 *
                intermediate_dim * hidden_dim,
        experts_per_device * 2 * intermediate_dim * hidden_dim);
    ptr16 += experts_per_device * 2 * intermediate_dim * hidden_dim;
  }
  dev_w->b_mlp1 = ptr16;
  for (int l = 0; l < n_layers; ++l) {
    getp_memcpy_fp32_to_bf16(
        ptr16,
        w->b_mlp1 + 1ll * l * n_experts * 2 * intermediate_dim +
            1ll * worker->expert_start * 2 * intermediate_dim,
        experts_per_device * 2 * intermediate_dim);
    ptr16 += experts_per_device * 2 * intermediate_dim;
  }
  dev_w->w_mlp2 = ptr16;
  for (int l = 0; l < n_layers; ++l) {
    getp_memcpy_fp32_to_bf16(
        ptr16,
        w->w_mlp2 + 1ll * l * n_experts * hidden_dim * intermediate_dim +
            1ll * worker->expert_start * hidden_dim * intermediate_dim,
        experts_per_device * hidden_dim * intermediate_dim);
    ptr16 += experts_per_device * hidden_dim * intermediate_dim;
  }
  dev_w->b_mlp2 = ptr16;
  for (int l = 0; l < n_layers; ++l) {
    getp_memcpy_fp32_to_bf16(
        ptr16,
        w->b_mlp2 + 1ll * l * n_experts * hidden_dim +
            1ll * worker->expert_start * hidden_dim,
        experts_per_device * hidden_dim);
    ptr16 += experts_per_device * hidden_dim;
  }
}

void init_device_run_state(RunState *s, Config *p) {
  int kv_dim = p->head_dim * p->n_kv_heads;
  HIP_CHECK(hipMalloc(&s->x, BATCH_SIZE * p->hidden_dim * sizeof(float))); // (batch, hidden_dim)
  HIP_CHECK(hipMalloc(&s->t, BATCH_SIZE * p->hidden_dim * sizeof(float))); // (batch, hidden_dim)
  HIP_CHECK(hipMalloc(&s->tb, BATCH_SIZE * p->head_dim * p->n_attn_heads * sizeof(float))); // (batch, n_attn_heads, head_dim)
  HIP_CHECK(hipMalloc(&s->tb2, BATCH_SIZE * p->hidden_dim * sizeof(float)));  // (batch, hidden_dim)
  HIP_CHECK(hipMalloc(&s->router_score, BATCH_SIZE * p->n_experts * sizeof(float))); // (batch, n_experts)
  HIP_CHECK(hipMalloc(&s->topk_v, BATCH_SIZE * p->experts_per_token * sizeof(float))); // (batch, experts_per_token)
  HIP_CHECK(hipMalloc(&s->topk_i, BATCH_SIZE * p->experts_per_token * sizeof(int))); // (batch, experts_per_token)
  HIP_CHECK(hipMalloc(&s->mlp1_out, 2 * p->intermediate_dim * sizeof(float))); // no used?
  HIP_CHECK(hipMalloc(&s->gate, p->intermediate_dim * sizeof(float)));  // no used?
  HIP_CHECK(hipMalloc(&s->up, p->intermediate_dim * sizeof(float))); // no used?
  HIP_CHECK(hipMalloc(&s->gate_up, BATCH_SIZE * p->experts_per_token * p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->e_agg, BATCH_SIZE * p->hidden_dim * sizeof(float))); // (batch, hidden_dim)
  HIP_CHECK(hipMalloc(&s->qkv, p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float))); // no used
  HIP_CHECK(hipMalloc(&s->q, BATCH_SIZE * p->n_attn_heads * p->head_dim * sizeof(float))); // (batch, n_attn_heads, head_dim)
  HIP_CHECK(hipMalloc(&s->key_cache, BATCH_SIZE * p->n_layers * p->seq_len * kv_dim * sizeof(float))); // (layer, seq_len, batch, n_kv_heads, head_dim)
  HIP_CHECK(hipMalloc(&s->value_cache, BATCH_SIZE * p->n_layers * p->seq_len * kv_dim * sizeof(float))); // (layer, seq_len, batch, n_kv_heads, head_dim)
  HIP_CHECK(hipMalloc(&s->att, BATCH_SIZE * p->n_attn_heads * (p->seq_len + 1) * sizeof(float))); // (batch, n_attn_heads, seq_len + 1)
  HIP_CHECK(hipMalloc(&s->logits, BATCH_SIZE * p->vocab_size * sizeof(float))); // (batch, vocab)
  if (p->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&s->mask, p->seq_len * p->seq_len * sizeof(float)));
    dim3 blockDim(32, 32);
    dim3 gridDim((p->seq_len + blockDim.x - 1) / blockDim.x,
                 (p->seq_len + blockDim.y - 1) / blockDim.y);
    init_mask_kernel<<<gridDim, blockDim>>>(s->mask, p->seq_len,
                                            p->sliding_window);
  } else {
    s->mask = NULL;
  }
}

void free_device_run_state(RunState *s) {
  HIP_CHECK(hipFree(s->x));
  HIP_CHECK(hipFree(s->t));
  HIP_CHECK(hipFree(s->tb));
  HIP_CHECK(hipFree(s->tb2));
  HIP_CHECK(hipFree(s->router_score));
  HIP_CHECK(hipFree(s->topk_v));
  HIP_CHECK(hipFree(s->topk_i));
  HIP_CHECK(hipFree(s->mlp1_out));
  HIP_CHECK(hipFree(s->gate));
  HIP_CHECK(hipFree(s->up));
  HIP_CHECK(hipFree(s->gate_up));
  HIP_CHECK(hipFree(s->e_agg));
  HIP_CHECK(hipFree(s->qkv));
  HIP_CHECK(hipFree(s->q));
  HIP_CHECK(hipFree(s->att));
  HIP_CHECK(hipFree(s->logits));
  HIP_CHECK(hipFree(s->key_cache));
  HIP_CHECK(hipFree(s->value_cache));
  if (s->mask) HIP_CHECK(hipFree(s->mask));
}

void upload_transformer(Transformer *transformer,
                        DeviceTransformer *dev_transformer, 
                        GPUWorker *worker) {
  dev_transformer->config = transformer->config;
  upload_weights(
      &transformer->weights, &dev_transformer->weights, &transformer->config,
      &dev_transformer->dev_data, &dev_transformer->dev_experts,
      &dev_transformer->dev_linear_bf16, 
      worker, dev_transformer->device_index);
  init_device_run_state(&dev_transformer->state, &dev_transformer->config);
  HIP_CHECK(hipStreamCreate(&dev_transformer->memory_stream));
  HIP_CHECK(hipStreamCreate(&dev_transformer->compute_stream));
}

void cleanup(Transformer *transformer, DeviceTransformer *dev_transformer) {
  free_device_run_state(&dev_transformer->state);
  HIP_CHECK(hipStreamDestroy(dev_transformer->memory_stream));
  HIP_CHECK(hipStreamDestroy(dev_transformer->compute_stream));
}
