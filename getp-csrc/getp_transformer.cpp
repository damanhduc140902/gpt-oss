#pragma once

#include "getp_eval.cpp"
#include "getp_transformer.hpp"
#include <cmath>
#include <iostream>
#include <hip/hip_runtime.h>

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

__global__ void init_mask_kernel(float *mask, int seq_len, int sliding_window) {
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= seq_len || j >= seq_len) return;

  mask[i * seq_len + j] = (sliding_window > 0 && i - j >= sliding_window ? -INFINITY : 0);
}

void free_device_run_state(RunState *s);

DeviceTransformer::~DeviceTransformer() {
  HIP_CHECK(hipSetDevice(device_index));
  if (dev_data) {
    HIP_CHECK(hipFree(dev_data));
  }
  if (dev_experts) {
    HIP_CHECK(hipFree(dev_experts));
  }
  free_device_run_state(&state);
}

void getp_memcpy_fp32_to_bf16(__hip_bfloat16 *dst, float *src, int n_elements) {
  __hip_bfloat16 *tmp = reinterpret_cast<__hip_bfloat16 *>(malloc(sizeof(__hip_bfloat16) * n_elements));
  for (int i = 0; i < n_elements; ++i) {
    tmp[i] = __float2bfloat16(src[i]);
  }
  HIP_CHECK(hipMemcpy(dst, tmp, sizeof(__hip_bfloat16) * n_elements, hipMemcpyHostToDevice));
  free(tmp);
}

static void upload_weights(TransformerWeights *w, DeviceTransformerWeights *dev_w, Config *cfg, 
                           float **_dev_data, __hip_bfloat16 **_dev_experts, int device_index) {
  int head_dim = cfg->head_dim;
  int n_layers = cfg->n_layers;
  int n_experts = cfg->n_experts;
  int intermediate_dim = cfg->intermediate_dim;
  int hidden_dim = cfg->hidden_dim;
  // token_embedding_table points to the beginning of the memory region consisting of the TransformerWeights
  // b_mlp2 points the "almost" ending of the memory region
  float *weights_begin = w->token_embedding_table;
  float *weights_end = w->b_mlp2 + 1ll * n_layers * n_experts * cfg->hidden_dim;
  ssize_t weights_size = (weights_end - weights_begin) * sizeof(float);

  float *experts_begin = w->w_mlp1;
  float *experts_end = weights_end;
  ssize_t experts_size = (experts_end - experts_begin) * sizeof(float);

  HIP_CHECK(hipSetDevice(device_index));

  // if (device_index == 0) {
  //   HIP_CHECK(hipMalloc(_dev_data, weights_size - experts_size));
  //   HIP_CHECK(hipMemcpy(*_dev_data, w->token_embedding_table, weights_size - experts_size, hipMemcpyHostToDevice));
  //   // getp_memcpy_fp32_to_bf16(*_dev_data, w->token_embedding_table, (weights_size - experts_size) / sizeof(float));
  
  //   float *ptr = *_dev_data;
  //   dev_w->token_embedding_table = ptr;
  //   ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
  //   dev_w->out = ptr; // unembedding
  //   ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
  //   dev_w->rms_attn_w = ptr;
  //   ptr += 1ll * n_layers * cfg->hidden_dim;
  //   dev_w->rms_ffn_w = ptr;
  //   ptr += 1ll * n_layers * cfg->hidden_dim;
  //   dev_w->rms_out_w = ptr;
  //   ptr += 1ll * cfg->hidden_dim;
  //   // hey it's qkvqkv, not qqkkvv
  //   dev_w->w_qkv = ptr;
  //   ptr += 1ll * n_layers * cfg->hidden_dim *
  //          (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  //   dev_w->b_qkv = ptr;
  //   ptr += 1ll * n_layers *
  //          (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  //   dev_w->w_o = ptr;
  //   ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;
  //   dev_w->b_o = ptr;
  //   ptr += 1ll * n_layers * cfg->hidden_dim;
  //   dev_w->attn_sinks = ptr;
  //   ptr += 1ll * n_layers * cfg->n_attn_heads;
  //   dev_w->w_router = ptr;
  //   ptr += 1ll * n_layers * cfg->hidden_dim * n_experts;
  //   dev_w->b_router = ptr;
  //   ptr += 1ll * n_layers * n_experts;
  // }
  { // replicate non-expert weights to every device (TP=1 base; shard later when TP>1)
    HIP_CHECK(hipMalloc(_dev_data, weights_size - experts_size));
    HIP_CHECK(hipMemcpy(*_dev_data, w->token_embedding_table, weights_size - experts_size, hipMemcpyHostToDevice));
  
    float *ptr = *_dev_data;
    dev_w->token_embedding_table = ptr;
    ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
    dev_w->out = ptr; // unembedding
    ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
    dev_w->rms_attn_w = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim;
    dev_w->rms_ffn_w = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim;
    dev_w->rms_out_w = ptr;
    ptr += 1ll * cfg->hidden_dim;
    // qkvqkv layout
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
  

  int n_devices;
  HIP_CHECK(hipGetDeviceCount(&n_devices));

  HIP_CHECK(hipMalloc(_dev_experts, (experts_size / sizeof(float)) * sizeof(__hip_bfloat16) / n_devices));
  __hip_bfloat16 *ptr = *_dev_experts;

  dev_w->w_mlp1 = ptr;
  for (int l = 0; l < n_layers; ++l) {
    getp_memcpy_fp32_to_bf16(ptr, w->w_mlp1 + 
              1ll * l * n_experts * 2 * intermediate_dim * hidden_dim + 
              1ll * device_index * (n_experts / n_devices) * 2 * intermediate_dim * hidden_dim, 
              (n_experts / n_devices) * 2 * intermediate_dim * hidden_dim);
    ptr += (n_experts / n_devices) * 2 * intermediate_dim * hidden_dim;
  }

  dev_w->b_mlp1 = ptr;
  for (int l = 0; l < n_layers; ++l) {
    getp_memcpy_fp32_to_bf16(ptr, w->b_mlp1 + 
              1ll * l * n_experts * 2 * intermediate_dim + 
              1ll * device_index * (n_experts / n_devices) * 2 * intermediate_dim, 
              (n_experts / n_devices) * 2 * intermediate_dim);
    ptr += (n_experts / n_devices) * 2 * intermediate_dim;
  }

  dev_w->w_mlp2 = ptr;
  for (int l = 0; l < n_layers; ++l) {
    getp_memcpy_fp32_to_bf16(ptr, w->w_mlp2 + 
              1ll * l * n_experts * hidden_dim * intermediate_dim + 
              1ll * device_index * (n_experts / n_devices) * hidden_dim * intermediate_dim, 
              (n_experts / n_devices) * hidden_dim * intermediate_dim);
    ptr += (n_experts / n_devices) * hidden_dim * intermediate_dim;
  }

  dev_w->b_mlp2 = ptr;
  for (int l = 0; l < n_layers; ++l) {
    getp_memcpy_fp32_to_bf16(ptr, w->b_mlp2 + 
              1ll * l * n_experts * hidden_dim + 
              1ll * device_index * (n_experts / n_devices) * hidden_dim, 
              (n_experts / n_devices) * hidden_dim);
    ptr += (n_experts / n_devices) * hidden_dim;
  }
}

void init_device_run_state(RunState *s, Config *p) {
  int kv_dim = p->head_dim * p->n_kv_heads;
  // s->x = reinterpret_cast<float *>(calloc(p->hidden_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->x, p->hidden_dim * sizeof(float)));
  // s->t = reinterpret_cast<float *>(calloc(p->hidden_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->t, p->hidden_dim * sizeof(float)));
  // s->tb = reinterpret_cast<float *>(
  //     calloc(p->head_dim * p->n_attn_heads, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb, p->head_dim * p->n_attn_heads * sizeof(float)));
  // s->tb2 = reinterpret_cast<float *>(calloc(p->hidden_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb2, p->hidden_dim * sizeof(float)));

  // s->router_score =
  //     reinterpret_cast<float *>(calloc(p->n_experts, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->router_score, p->n_experts * sizeof(float)));
  // s->topk_v =
  //     reinterpret_cast<float *>(calloc(p->experts_per_token, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->topk_v, p->experts_per_token * sizeof(float)));
  // s->topk_i =
  //     reinterpret_cast<int *>(calloc(p->experts_per_token, sizeof(int)));
  HIP_CHECK(hipMalloc(&s->topk_i, p->experts_per_token * sizeof(int)));

  // s->mlp1_out =
  //     reinterpret_cast<float *>(calloc(2 * p->intermediate_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->mlp1_out, 2 * p->intermediate_dim * sizeof(float)));
  // s->gate =
  //     reinterpret_cast<float *>(calloc(p->intermediate_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate, p->intermediate_dim * sizeof(float)));
  // s->up = reinterpret_cast<float *>(calloc(p->intermediate_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->up, p->intermediate_dim * sizeof(float)));
  // s->gate_up =
  //     reinterpret_cast<float *>(calloc(p->intermediate_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up, p->intermediate_dim * sizeof(float)));
  // s->e_agg = reinterpret_cast<float *>(calloc(p->hidden_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->e_agg, p->hidden_dim * sizeof(float)));

  // s->qkv = reinterpret_cast<float *>(calloc(
  //     p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads), sizeof(float)));
  HIP_CHECK(hipMalloc(&s->qkv, p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads) * sizeof(float)));
  // s->q = reinterpret_cast<float *>(
  //     calloc(p->n_attn_heads * p->head_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->q, p->n_attn_heads * p->head_dim * sizeof(float)));

  // s->key_cache = reinterpret_cast<float *>(
  //     calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
  // s->value_cache = reinterpret_cast<float *>(
  //     calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
  // s->att = reinterpret_cast<float *>(
  //     calloc(p->n_attn_heads * p->seq_len, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->att, p->n_attn_heads * p->seq_len * sizeof(float)));
  // s->logits = reinterpret_cast<float *>(calloc(p->vocab_size, sizeof(float)));
  HIP_CHECK(hipMalloc(&s->logits, p->vocab_size * sizeof(float)));
  // s->mask = p->sliding_window > 0 ? reinterpret_cast<float *>(calloc(
  //                                       p->seq_len * p->seq_len, sizeof(float)))
  //                                 : NULL;
  if (p->sliding_window > 0) {
    HIP_CHECK(hipMalloc(&s->mask, p->seq_len * p->seq_len * sizeof(float)));
    dim3 blockDim(32, 32);
    dim3 gridDim((p->seq_len + blockDim.x - 1) / blockDim.x, 
                 (p->seq_len + blockDim.y - 1) / blockDim.y);
    init_mask_kernel<<<gridDim, blockDim>>>(s->mask, p->seq_len, p->sliding_window);
  }
  else {
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
  if (s->mask)
    HIP_CHECK(hipFree(s->mask));
}

void upload_transformer(Transformer *transformer, DeviceTransformer *dev_transformer) {
  dev_transformer->config = transformer->config;
  upload_weights(&transformer->weights, &dev_transformer->weights, &transformer->config, 
    &dev_transformer->dev_data, &dev_transformer->dev_experts, dev_transformer->device_index);
  init_device_run_state(&dev_transformer->state, &dev_transformer->config);
}

void cleanup(Transformer *transformer, DeviceTransformer *dev_transformer) {
  free_device_run_state(&dev_transformer->state);
}