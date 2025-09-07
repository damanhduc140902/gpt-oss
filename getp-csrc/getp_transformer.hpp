#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#define BATCH_SIZE            (32)
#define EXPERT_PARALLELISM    (8)

struct GPUWorker {
  int device_index;

  int expert_start;
  int expert_end;

  int request_start;
  int request_end;
};

struct DeviceTransformerWeights {
  // token_embedding_table - embedding.weight
  float *token_embedding_table;  // (vocab_size, hidden_dim) (in, out)
  // weights for rmsnorms
  float *rms_attn_w;  // (n_layers, hidden_dim) [attn.norm.scale]
  float *rms_ffn_w;   // (n_layers, hidden_dim) [mlp.norm.scale]
  // weights for attention [attn.qkv.weight & attn.qkv.bias]
  float *
      w_qkv;  // (n_layers, head_dim * n_attn_heads + 2 * head_dim * n_kv_heads,
              // hidden_dim) where w_q (head_dim * n_attn_heads, hidden_dim)
              // (out_features, in_features) w_k (head_dim * n_kv_heads,
              // hidden_dim)  (out_features, in_features) w_v (head_dim *
              // n_kv_heads, hidden_dim)  (out_features, in_features)
  float *w_o;    // (n_layers, hidden_dim, head_dim * n_attn_heads)
  float *b_qkv;  // (n_layers, head_dim * n_attn_heads + 2 * head_dim *
                 // n_kv_heads) (head_dim * n_attn_heads) (head_dim *
                 // n_kv_heads) (head_dim * n_kv_heads)
  float *b_o;         // (n_layers, hidden_dim)
  float *attn_sinks;  // (n_layers, n_attn_heads)
  // weights for router [mlp.gate.weight & mlp.gate.bias]
  float *w_router;  // (n_layers, hidden_dim, n_experts)
  float *b_router;  // (n_layers, n_experts)
  // weights for MoE [mlp.mlp1_weight & mlp.mlp1_bias & mlp.mlp2_weight &
  // mlp.mlp2_bias] NOTE: gate_up projects from hidden_dim to intermediate_dim,
  // the shape is kinda reverted because the original code use einsum to reduce
  // over hidden_dim
  __hip_bfloat16 *w_mlp1;  // gate_up_proj (n_layers, n_experts, 2 *
                           // intermediate_dim, hidden_dim)
  __hip_bfloat16
      *w_mlp2;  // down_proj (n_layers, n_experts, hidden_dim, intermediate_dim)
  __hip_bfloat16
      *b_mlp1;  // gate_up proj (n_layers, n_experts, 2 * intermediate_dim)
  __hip_bfloat16 *b_mlp2;  // down_proj (n_layers, n_experts, hidden_dim)
  // final norm [norm.scale]
  float *rms_out_w;  // (hidden_dim, )
  // classifier weights for the logits [unembedding.weight]
  float *out;  // (vocab_size, hidden_dim) (out, in)
  __hip_bfloat16 *w_qkv_bf16, *b_qkv_bf16;
  __hip_bfloat16 *w_o_bf16, *b_o_bf16;
  __hip_bfloat16 *w_router_bf16, *b_router_bf16;
  __hip_bfloat16 *out_bf16;
};

struct DeviceTransformer {
  Config config;
  DeviceTransformerWeights weights;
  RunState state;  // buffers for the "wave" of activations in the forward pass

  float *dev_data;  // exclude experts
  __hip_bfloat16 *dev_experts;

  __hip_bfloat16 *dev_linear_bf16;

  int device_index;

  ~DeviceTransformer();
};