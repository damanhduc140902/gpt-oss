#pragma once

#include "getp_eval.cpp"
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

static void upload_weights(TransformerWeights *w, TransformerWeights *dev_w, Config *cfg) {
  int head_dim = cfg->head_dim;
  int n_layers = cfg->n_layers;
  int n_experts = cfg->n_experts;
  // token_embedding_table points to the beginning of the memory region consisting of the TransformerWeights
  // b_mlp2 points the "almost" ending of the memory region
  ssize_t weights_size = (w->b_mlp2 - w->token_embedding_table + 
    1ll * n_layers * n_experts * cfg->hidden_dim) * sizeof(float);
  float *dev_data;
  HIP_CHECK(hipMalloc(&dev_data, weights_size));
  HIP_CHECK(hipMemcpy(dev_data, w->token_embedding_table, weights_size, hipMemcpyHostToDevice));

  dev_w->token_embedding_table = dev_data;
  dev_data += 1ll * cfg->vocab_size * cfg->hidden_dim;
  dev_w->out = dev_data; // unembedding
  dev_data += 1ll * cfg->vocab_size * cfg->hidden_dim;
  dev_w->rms_attn_w = dev_data;
  dev_data += 1ll * n_layers * cfg->hidden_dim;
  dev_w->rms_ffn_w = dev_data;
  dev_data += 1ll * n_layers * cfg->hidden_dim;
  dev_w->rms_out_w = dev_data;
  dev_data += 1ll * cfg->hidden_dim;
  // hey it's qkvqkv, not qqkkvv
  dev_w->w_qkv = dev_data;
  dev_data += 1ll * n_layers * cfg->hidden_dim *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  dev_w->b_qkv = dev_data;
  dev_data += 1ll * n_layers *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  dev_w->w_o = dev_data;
  dev_data += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;
  dev_w->b_o = dev_data;
  dev_data += 1ll * n_layers * cfg->hidden_dim;
  dev_w->attn_sinks = dev_data;
  dev_data += 1ll * n_layers * cfg->n_attn_heads;
  dev_w->w_router = dev_data;
  dev_data += 1ll * n_layers * cfg->hidden_dim * n_experts;
  dev_w->b_router = dev_data;
  dev_data += 1ll * n_layers * n_experts;
  // hey it's gate_upgate_up, not gategateupup
  dev_w->w_mlp1 = dev_data;
  dev_data +=
      1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim;
  dev_w->b_mlp1 = dev_data;
  dev_data += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim;
  dev_w->w_mlp2 = dev_data;
  dev_data += 1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim;
  dev_w->b_mlp2 = dev_data;
  dev_data += 1ll * n_layers * n_experts * cfg->hidden_dim;
}

static void free_device_weights(TransformerWeights *dev_w) {
  // token_embedding_table points to the beginning of the memory region consisting of the TransformerWeights
  HIP_CHECK(hipFree(dev_w->token_embedding_table));
}

static void init_device_run_state(RunState *s, Config *p) {
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
  HIP_CHECK(hipMalloc(&s->topk_i, p->experts_per_token * sizeof(float)));

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
  }
  else {
    s->mask = NULL;
  }
  
  // TODO: initialize mask on device
  // for (int i = 0; i < p->seq_len; i++) {
  //   for (int j = 0; j < p->seq_len; j++) {
  //     if (p->sliding_window > 0 && i - j >= p->sliding_window) {
  //       s->mask[i * p->seq_len + j] = -INFINITY; // Sliding window mask
  //     }
  //   }
  // }

}

static void free_device_run_state(RunState *s) {
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

void upload_transformer(Transformer *transformer, Transformer *dev_transformer) {
  dev_transformer->config = transformer->config;
  upload_weights(&transformer->weights, &dev_transformer->weights, &transformer->config);
  init_device_run_state(&dev_transformer->state, &dev_transformer->config);
}

void cleanup(Transformer *transformer, Transformer *dev_transformer) {
  free_device_weights(&dev_transformer->weights);
  free_device_run_state(&dev_transformer->state);
}