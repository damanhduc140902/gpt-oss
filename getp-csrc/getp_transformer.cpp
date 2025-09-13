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
                           __hip_bfloat16 **_dev_linear_bf16, GPUWorker *worker,
                           int device_index) {
  HIP_CHECK(hipSetDevice(device_index));

  const int L = cfg->n_layers;
  const int E = cfg->n_experts;
  const int H = cfg->hidden_dim;
  const int D = cfg->head_dim;
  const int I = cfg->intermediate_dim;

  const size_t emb_elems = (size_t)cfg->vocab_size * (size_t)H;
  const size_t rms_attn_elems = (size_t)L * (size_t)H;
  const size_t rms_ffn_elems = (size_t)L * (size_t)H;
  const size_t rms_out_elems = (size_t)H;
  const size_t sinks_elems = (size_t)L * (size_t)cfg->n_attn_heads;

  const size_t keep_fp32_elems =
      rms_attn_elems + rms_ffn_elems + rms_out_elems + sinks_elems;

  HIP_CHECK(hipMalloc(_dev_data, keep_fp32_elems * sizeof(float)));
  float *p32 = *_dev_data;

  dev_w->rms_attn_w = p32;
  HIP_CHECK(hipMemcpy(p32, w->rms_attn_w, rms_attn_elems * sizeof(float),
                      hipMemcpyHostToDevice));
  p32 += rms_attn_elems;

  dev_w->rms_ffn_w = p32;
  HIP_CHECK(hipMemcpy(p32, w->rms_ffn_w, rms_ffn_elems * sizeof(float),
                      hipMemcpyHostToDevice));
  p32 += rms_ffn_elems;

  dev_w->rms_out_w = p32;
  HIP_CHECK(hipMemcpy(p32, w->rms_out_w, rms_out_elems * sizeof(float),
                      hipMemcpyHostToDevice));
  p32 += rms_out_elems;

  dev_w->attn_sinks = p32;
  HIP_CHECK(hipMemcpy(p32, w->attn_sinks, sinks_elems * sizeof(float),
                      hipMemcpyHostToDevice));
  p32 += sinks_elems;

  dev_w->token_embedding_table = nullptr;
  dev_w->w_qkv = dev_w->b_qkv = nullptr;
  dev_w->w_o = dev_w->b_o = nullptr;
  dev_w->w_router = dev_w->b_router = nullptr;
  dev_w->out = nullptr;

  const size_t wqkv = (size_t)L * (size_t)H *
                      ((size_t)D * (size_t)cfg->n_attn_heads +
                       2ull * (size_t)D * (size_t)cfg->n_kv_heads);
  const size_t bqkv = (size_t)L * ((size_t)D * (size_t)cfg->n_attn_heads +
                                   2ull * (size_t)D * (size_t)cfg->n_kv_heads);
  const size_t wo =
      (size_t)L * (size_t)H * (size_t)D * (size_t)cfg->n_attn_heads;
  const size_t bo = (size_t)L * (size_t)H;
  const size_t wr = (size_t)L * (size_t)H * (size_t)E;
  const size_t br = (size_t)L * (size_t)E;
  const size_t wout = (size_t)cfg->vocab_size * (size_t)H;

  const size_t linear_bf16_elems =
      emb_elems + wqkv + bqkv + wo + bo + wr + br + wout;

  HIP_CHECK(
      hipMalloc(_dev_linear_bf16, linear_bf16_elems * sizeof(__hip_bfloat16)));
  __hip_bfloat16 *p16 = *_dev_linear_bf16;

  dev_w->token_embedding_table_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->token_embedding_table, emb_elems);
  p16 += emb_elems;

  dev_w->w_qkv_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->w_qkv, wqkv);
  p16 += wqkv;
  dev_w->b_qkv_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->b_qkv, bqkv);
  p16 += bqkv;
  dev_w->w_o_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->w_o, wo);
  p16 += wo;
  dev_w->b_o_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->b_o, bo);
  p16 += bo;
  dev_w->w_router_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->w_router, wr);
  p16 += wr;
  dev_w->b_router_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->b_router, br);
  p16 += br;
  dev_w->out_bf16 = p16;
  getp_memcpy_fp32_to_bf16(p16, w->out, wout);
  p16 += wout;

  const int experts_per_device = worker->expert_end - worker->expert_start;
  const size_t w1_per_layer = 2ull * (size_t)I * (size_t)H;
  const size_t b1_per_layer = 2ull * (size_t)I;
  const size_t w2_per_layer = (size_t)H * (size_t)I;
  const size_t b2_per_layer = (size_t)H;
  const size_t expert_layer_pack =
      w1_per_layer + b1_per_layer + w2_per_layer + b2_per_layer;
  const size_t expert_total =
      (size_t)L * (size_t)experts_per_device * expert_layer_pack;

  HIP_CHECK(hipMalloc(_dev_experts, expert_total * sizeof(__hip_bfloat16)));
  __hip_bfloat16 *e16 = *_dev_experts;

  dev_w->w_mlp1 = e16;
  for (int l = 0; l < L; ++l) {
    getp_memcpy_fp32_to_bf16(e16,
                             w->w_mlp1 + 1ll * l * E * 2 * I * H +
                                 1ll * worker->expert_start * 2 * I * H,
                             (size_t)experts_per_device * 2ull * I * H);
    e16 += (size_t)experts_per_device * 2ull * I * H;
  }
  dev_w->b_mlp1 = e16;
  for (int l = 0; l < L; ++l) {
    getp_memcpy_fp32_to_bf16(
        e16,
        w->b_mlp1 + 1ll * l * E * 2 * I + 1ll * worker->expert_start * 2 * I,
        (size_t)experts_per_device * 2ull * I);
    e16 += (size_t)experts_per_device * 2ull * I;
  }
  dev_w->w_mlp2 = e16;
  for (int l = 0; l < L; ++l) {
    getp_memcpy_fp32_to_bf16(
        e16,
        w->w_mlp2 + 1ll * l * E * H * I + 1ll * worker->expert_start * H * I,
        (size_t)experts_per_device * (size_t)H * (size_t)I);
    e16 += (size_t)experts_per_device * (size_t)H * (size_t)I;
  }
  dev_w->b_mlp2 = e16;
  for (int l = 0; l < L; ++l) {
    getp_memcpy_fp32_to_bf16(
        e16, w->b_mlp2 + 1ll * l * E * H + 1ll * worker->expert_start * H,
        (size_t)experts_per_device * (size_t)H);
    e16 += (size_t)experts_per_device * (size_t)H;
  }
}

void init_device_run_state(RunState *s, Config *p) {
  int kv_dim = p->head_dim * p->n_kv_heads;

  HIP_CHECK(
      hipMalloc(&s->x, (size_t)BATCH_SIZE * p->hidden_dim * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&s->t, (size_t)BATCH_SIZE * p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->tb, (size_t)BATCH_SIZE * p->head_dim *
                                  p->n_attn_heads * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&s->tb2, (size_t)BATCH_SIZE * p->hidden_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->router_score,
                      (size_t)BATCH_SIZE * p->n_experts * sizeof(float)));
  HIP_CHECK(hipMalloc(
      &s->topk_v, (size_t)BATCH_SIZE * p->experts_per_token * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->topk_i,
                      (size_t)BATCH_SIZE * p->experts_per_token * sizeof(int)));
  HIP_CHECK(
      hipMalloc(&s->mlp1_out, (size_t)2 * p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate, (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->up, (size_t)p->intermediate_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->gate_up, (size_t)BATCH_SIZE * p->experts_per_token *
                                       p->intermediate_dim * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&s->e_agg, (size_t)BATCH_SIZE * p->hidden_dim * sizeof(float)));
  s->qkv = nullptr;
  HIP_CHECK(hipMalloc(&s->q, (size_t)BATCH_SIZE * p->n_attn_heads *
                                 p->head_dim * sizeof(float)));

  const int even_layers = (p->n_layers + 1) / 2;
  const int even_tcap = (p->sliding_window > 0 ? p->sliding_window : 1);
  const int odd_tcap = 1;
  const size_t total_t = (size_t)even_layers * (size_t)even_tcap +
                         (size_t)(p->n_layers - even_layers) * (size_t)odd_tcap;

  HIP_CHECK(hipMalloc(&s->key_cache, (size_t)BATCH_SIZE * total_t *
                                         (size_t)kv_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&s->value_cache, (size_t)BATCH_SIZE * total_t *
                                           (size_t)kv_dim * sizeof(float)));

  s->att = nullptr;
  HIP_CHECK(hipMalloc(&s->logits,
                      (size_t)BATCH_SIZE * p->vocab_size * sizeof(float)));
  s->mask = NULL;
}

static inline size_t kv_total_t(const Config *p) {
  const int even_layers = (p->n_layers + 1) / 2;
  const int even_tcap =
      (p->sliding_window > 0 ? p->sliding_window : p->seq_len);
  return (size_t)even_layers * (size_t)even_tcap +
         (size_t)(p->n_layers - even_layers) * (size_t)p->seq_len;
}

void resize_kv_cache(DeviceTransformer *dev, int seq_len_eff) {
  HIP_CHECK(hipSetDevice(dev->device_index));
  dev->config.seq_len = seq_len_eff;
  const int kv_dim = dev->config.head_dim * dev->config.n_kv_heads;
  const size_t total_t = kv_total_t(&dev->config);
  HIP_CHECK(hipFree(dev->state.key_cache));
  HIP_CHECK(hipFree(dev->state.value_cache));
  HIP_CHECK(
      hipMalloc(&dev->state.key_cache,
                (size_t)BATCH_SIZE * total_t * (size_t)kv_dim * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&dev->state.value_cache,
                (size_t)BATCH_SIZE * total_t * (size_t)kv_dim * sizeof(float)));
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
  if (s->qkv) HIP_CHECK(hipFree(s->qkv));
  HIP_CHECK(hipFree(s->q));
  if (s->att) HIP_CHECK(hipFree(s->att));
  HIP_CHECK(hipFree(s->logits));
  HIP_CHECK(hipFree(s->key_cache));
  HIP_CHECK(hipFree(s->value_cache));
  if (s->mask) HIP_CHECK(hipFree(s->mask));
}

void upload_transformer(Transformer *transformer,
                        DeviceTransformer *dev_transformer, GPUWorker *worker) {
  dev_transformer->config = transformer->config;
  upload_weights(
      &transformer->weights, &dev_transformer->weights, &transformer->config,
      &dev_transformer->dev_data, &dev_transformer->dev_experts,
      &dev_transformer->dev_linear_bf16, worker, dev_transformer->device_index);
  init_device_run_state(&dev_transformer->state, &dev_transformer->config);
  HIP_CHECK(hipStreamCreate(&dev_transformer->memory_stream));
  HIP_CHECK(hipStreamCreate(&dev_transformer->compute_stream));
}

void cleanup(Transformer *transformer, DeviceTransformer *dev_transformer) {
  free_device_run_state(&dev_transformer->state);
  HIP_CHECK(hipStreamDestroy(dev_transformer->memory_stream));
  HIP_CHECK(hipStreamDestroy(dev_transformer->compute_stream));
}
