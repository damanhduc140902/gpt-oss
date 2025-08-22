#include "getp_transformer.hpp"
#include <cstdlib>
#include <cstring>
#include <math.h>

float *getp_forward(Transformer *transformer, DeviceTransformer **dev_transformeres, int token, int pos) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  float *x = s->x;
  int head_dim = p->head_dim;
  int hidden_dim = p->hidden_dim;
  int kv_dim = p->head_dim * p->n_kv_heads;
  int kv_mul =
      p->n_attn_heads /
      p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int intermediate_dim = p->intermediate_dim;
  int n_experts = p->n_experts;

  // copy the token embedding into x
  float *content_row = w->token_embedding_table + token * hidden_dim;
  memcpy(x, content_row, hidden_dim * sizeof(*x));

  // forward all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {
    // s->t (hidden_dim, )
    rmsnorm(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);

    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k = s->key_cache + loff + pos * kv_dim;
    s->v = s->value_cache + loff + pos * kv_dim;

    // s->qkv = w->w_qkv * s->t = (head_dim * (n_attn_heads + 2 * n_kv_heads),
    // hidden_dim) * (hidden_dim, ) = head_dim * (n_attn_heads + 2 * n_kv_heads)
    float *w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                  (head_dim * p->n_attn_heads +
                                   2 * head_dim * p->n_kv_heads);
    float *b_qkv =
        w->b_qkv +
        1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
    matmul(s->qkv, s->t, w_qkv, hidden_dim,
           (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
    // add bias
    for (int i = 0; i < (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim; ++i) {
      s->qkv[i] += b_qkv[i];
    }
    // Separate q, k, v
    memcpy(s->q, s->qkv, head_dim * p->n_attn_heads * sizeof(float)); // gate
    memcpy(s->k, s->qkv + head_dim * p->n_attn_heads,
           head_dim * p->n_kv_heads * sizeof(float)); // gate
    memcpy(s->v, s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads,
           head_dim * p->n_kv_heads * sizeof(float)); // gate

    // RoPE relative positional encoding: complex-valued rotate q and k in each
    // head Adapted from
    // https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py#L85
    // RoPE with YaRN scaling adapted from Python code
    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;
    float *cos_vals =
        reinterpret_cast<float *>(malloc((head_dim / 2) * sizeof(float)));
    float *sin_vals =
        reinterpret_cast<float *>(malloc((head_dim / 2) * sizeof(float)));
    compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                    p->initial_context_length, ntk_beta, ntk_alpha, cos_vals,
                    sin_vals);
    apply_rotary_emb(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
    apply_rotary_emb(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);

    free(cos_vals);
    free(sin_vals);

    // multihead attention. iterate over all heads
    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < p->n_attn_heads; h++) {
      // get the query vector for this head
      float *q = s->q + h * head_dim;
      // attention scores for this head
      float *att = s->att + h * p->seq_len;
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        // GQA
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
        // calculate the attention score as the dot product of q and k
        double score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
          score += q[i] * k[i];
        }
        score /= sqrtf(head_dim);
        // Apply sliding window mask if enabled
        if (p->sliding_window > 0 && (l % 2 == 0)) {
          score += s->mask[pos * p->seq_len + t];
        }
        // save the score to the attention buffer
        att[t] = score;
      }
      // Add attention sink score
      att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];
      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 2);

      // weighted sum of the values
      float *tb = s->tb + h * head_dim;
      memset(tb, 0, head_dim * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        // GQA
        float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < head_dim; i++) {
          tb[i] += a * v[i];
        }
      }
    }
    // final matmul to get the output of the attention
    float *w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    float *b_o = w->b_o + 1ll * l * hidden_dim;
    matmul(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
    // add bias b_o
    for (int i = 0; i < hidden_dim; i++) {
      s->tb2[i] += b_o[i];
    }

    // residual connection back into x
    for (int i = 0; i < hidden_dim; i++) {
      x[i] += s->tb2[i];
    }

    // ffn rmsnorm
    rmsnorm(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);

    // MoE
    // Compute router_score
    float *w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
    float *b_router = w->b_router + 1ll * l * n_experts;
    matmul(s->router_score, s->t, w_router, hidden_dim,
           n_experts); // s->router_score now stores router_score (n_experts, )
    // add bias b_router
    for (int i = 0; i < n_experts; i++) {
      s->router_score[i] += b_router[i];
    }
    // Select top-k experts
    topk(s->topk_v, s->topk_i, s->router_score, n_experts,
         p->experts_per_token);
    // Normalize selected experts using softmax or sigmoid
    softmax(s->topk_v, p->experts_per_token); // expert

    // Route the tokens to their corresponding top-k experts
    memset(s->e_agg, 0, hidden_dim * sizeof(float));
    for (int e = 0; e < n_experts; e++) {
      float expert_w = 0;
      int in_topk = 0;
      // Check if expert i is in top-k experts
      for (int idx = 0; idx < p->experts_per_token; idx++) {
        if (s->topk_i[idx] == e) {
          in_topk = 1;
          expert_w = s->topk_v[idx];
          break;
        }
      }

      if (in_topk) {
        float *w_mlp1 = w->w_mlp1 + 1ll * (l * n_experts + e) *
                                        (2 * p->intermediate_dim) * hidden_dim;
        float *b_mlp1 =
            w->b_mlp1 + 1ll * (l * n_experts + e) * (2 * p->intermediate_dim);
        matmul(s->mlp1_out, s->t, w_mlp1, hidden_dim,
               2 * p->intermediate_dim); // (2 * intermediate_dim, )
        for (int i = 0; i < 2 * p->intermediate_dim; i++) {
          s->mlp1_out[i] += b_mlp1[i];
        }
        // Split mlp1_out into gate and up
        for (int j = 0; j < p->intermediate_dim; j++) {
          s->gate[j] = s->mlp1_out[2 * j];
          s->up[j] = s->mlp1_out[2 * j + 1];
        }

        // SwiGLU non-linearity
        const float alpha = 1.702f;
        for (int i = 0; i < p->intermediate_dim; i++) {
          float val = s->gate[i];
          float up_val = s->up[i];
          // Clamping
          if (val > p->swiglu_limit)
            val = p->swiglu_limit;
          if (up_val > p->swiglu_limit)
            up_val = p->swiglu_limit;
          if (up_val < -p->swiglu_limit)
            up_val = -p->swiglu_limit;
          // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
          val *= (1.0f / (1.0f + expf(-alpha * val)));
          // elementwise multiply with w_gate(x)
          val *= (up_val +
                  1.0f); // gpt-oss adds an extra bias of 1 to the up layer
          s->gate_up[i] = val;
        }

        // final matmul to get the output of the ffn
        float *w_mlp2 =
            w->w_mlp2 +
            1ll * (l * n_experts + e) * hidden_dim *
                p->intermediate_dim; // (out: hidden_dim, in: intermediate_dim)
        float *b_mlp2 = w->b_mlp2 + 1ll * (l * n_experts + e) * hidden_dim;
        matmul(s->tb2, s->gate_up, w_mlp2, p->intermediate_dim,
               hidden_dim); // (hidden_dim, )
        for (int i = 0; i < hidden_dim; i++) {
          s->tb2[i] += b_mlp2[i];
        }

        // aggregate topk experts using weighted sum
        for (int i = 0; i < hidden_dim; i++) {
          s->e_agg[i] += s->tb2[i] * expert_w;
        }
      }
    }

    // residual connection
    for (int i = 0; i < hidden_dim; i++) {
      x[i] += s->e_agg[i];
    }
  }
  // final rmsnorm
  rmsnorm(x, x, w->rms_out_w, hidden_dim);

  // classifier into logits
  matmul(s->logits, x, w->out, hidden_dim, p->vocab_size);
  return s->logits;
}