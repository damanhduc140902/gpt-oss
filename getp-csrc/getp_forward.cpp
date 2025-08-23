#include "getp_transformer.hpp"
#include <cmath>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

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

__global__ void square_reduction_kernel(float *x, float *sum, int size) {
  extern __shared__ float L[];
  int tid = threadIdx.x;
  int offset = 2 * blockDim.x * blockIdx.x;
  int stride = blockDim.x;
  L[tid] = 0;
  if (tid + offset < size) {
    float t = x[tid + offset];
    L[tid] += t * t;
  }
  if (tid + offset + stride < size) {
    float t = x[tid + offset + stride];
    L[tid] += t * t;
  }
  __syncthreads();
  for (stride = stride / 2; stride > 0; stride /= 2) {
    if (tid < stride) L[tid] += L[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(sum, L[0]);
  }
}

__global__ void rmsnorm_kernel(float *o, float *x, float *weight, float ss, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= size) return;
  o[i] = weight[i] * (ss * x[i]);
}

void getp_rmsnorm(float *o, float *x, float *weight, int size) {
  float ss;
  float *dev_ss;
  HIP_CHECK(hipMalloc(&dev_ss, sizeof(float)));
  HIP_CHECK(hipMemset(dev_ss, 0, sizeof(float)));

  // calculate sum of squares
  {
    dim3 blockDim(1024); // arbitrary
    dim3 gridDim((size + 2 * blockDim.x - 1) / (2 * blockDim.x));
    square_reduction_kernel<<<gridDim, blockDim, sizeof(float) * blockDim.x>>>
      (x, dev_ss, size);
  }
  HIP_CHECK(hipMemcpy(&ss, dev_ss, sizeof(float), hipMemcpyDeviceToHost));
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  {
    dim3 blockDim(1024); // arbitrary
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    rmsnorm_kernel<<<gridDim, blockDim>>>(o, x, weight, ss, size);
  }
  HIP_CHECK(hipFree(dev_ss));
}

__global__ void matmul_kernel(float *xout, float *x, float *w, float *b, int n, int d) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= d) return;
  float val = 0.f;
  for (int j = 0; j < n; ++j) {
    val += w[1ll * i * n + j] * x[j];
  }
  xout[i] = val + b[i];
}

void getp_matmul(float *xout, float *x, float *w, float *b, int n, int d) {
  dim3 blockDim(64);
  dim3 gridDim((d + blockDim.x - 1) / blockDim.x);
  matmul_kernel<<<gridDim, blockDim>>>(xout, x, w, b, n, d);
}

__global__ void compute_inv_freq_kernel(float base, int head_dim,
                                        float scaling_factor,
                                        float initial_context_length,
                                        float ntk_beta, float ntk_alpha,
                                        float *inv_freq_out // length head_dim/2
) {
  const int d_half = head_dim / 2;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= d_half) return;
  float freq = powf(base, ((float)(2 * i)) / (float)head_dim);
  float inv_freq;
  if (scaling_factor > 1.0f) {
    float low = d_half * 
                logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / 
                logf(base);
    float high = d_half * 
                 logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) /
                 logf(base);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;

    float ramp = ((float)i - low) / (high - low);
    if (ramp < 0) ramp = 0;
    if (ramp > 1) ramp = 1;

    float mask = 1.0f - ramp;
    inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
  } 
  else {
    inv_freq = 1.0f / freq;
  }
  inv_freq_out[i] = inv_freq;
}

__global__ void compute_cos_sin_kernel(float *cos_out, float *sin_out, float *inv_freq, float concentration, int pos, int d_half) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= d_half) return;
  float val = (float)pos * inv_freq[i];
  cos_out[i] = cosf(val) * concentration;
  sin_out[i] = sinf(val) * concentration;
}

void getp_compute_cos_sin(int pos, // position index
                          float base, int head_dim, float scaling_factor,
                          float initial_context_length, float ntk_beta,
                          float ntk_alpha,
                          float *cos_out, // shape: head_dim/2
                          float *sin_out  // shape: head_dim/2
) {
  int d_half = head_dim / 2;
  
  float concentration = scaling_factor > 1.0f ? 
                          0.1f * logf(scaling_factor) + 1.0f : 
                          1.0f;
  float *inv_freq;
  HIP_CHECK(hipMalloc(&inv_freq, d_half * sizeof(float)));

  // Compute inverse frequency
  {
    dim3 blockDim(256); // arbitrary
    dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x);
    compute_inv_freq_kernel<<<gridDim, blockDim>>>
      (base, head_dim, scaling_factor, initial_context_length, ntk_beta, ntk_alpha, inv_freq);
  }
  // Compute sin, cos
  {
    dim3 blockDim(256); // arbitrary
    dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x);
    compute_cos_sin_kernel<<<gridDim, blockDim>>>
      (cos_out, sin_out, inv_freq, concentration, pos, head_dim / 2);
  }

  HIP_CHECK(hipFree(inv_freq));
}

__global__ void apply_rotary_emb_kernel(float *x, float *cos, float *sin, int n_heads, int head_dim) {
  const int half = head_dim / 2;

  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (h >= n_heads || i >= half) return;

  float x1 = x[h * head_dim + i];         // first half
  float x2 = x[h * head_dim + half + i];  // second half

  float c = cos[i];
  float s = sin[i];

  float o1 = x1 * c - x2 * s;
  float o2 = x2 * c + x1 * s;

  x[h * head_dim + i] = o1;
  x[h * head_dim + half + i] = o2;
}

void getp_apply_rotary_emb(float *x, float *cos, float *sin, int n_heads,
                           int head_dim) {
  dim3 blockDim(16, 16); // arbitrary
  dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x, 
               (n_heads + blockDim.y - 1) / blockDim.y);
  apply_rotary_emb_kernel<<<gridDim, blockDim>>>(x, cos, sin, n_heads, head_dim);
}

__global__ void multihead_attention_kernel(float *key_cache, float *value_cache, float *query, float *mask, float *attn_sinks, float *attn, 
                                           int l, int head_dim, int n_attn_heads, int n_kv_heads, int pos, int seq_len, int apply_mask
) {
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int t = blockDim.x * blockIdx.x + threadIdx.x;

  if (h >= n_attn_heads || t > pos + 1) return;

  if (t == pos + 1) {
    attn[h * seq_len + t] = attn_sinks[l * n_attn_heads + h];
    return;
  }

  const int kv_dim = head_dim * n_kv_heads;
  const int kv_mul = n_attn_heads / n_kv_heads;
  const float *q = query + h * head_dim;
  const float *k = key_cache + t * kv_dim + (h / kv_mul) * head_dim;

  float score = 0;
  for (int i = 0; i < head_dim; ++i) {
    score += q[i] * k[i];
  }
  score = score / sqrtf((float)head_dim);

  if (apply_mask) {
    score += mask[pos * seq_len + t];
  }

  attn[h * seq_len + t] = score;
}

void getp_multihead_attention(float *key_cache, float *value_cache, float *query, float *mask, float *attn_sinks, float *attn,
                              int l, int head_dim, int n_attn_heads, int n_kv_heads, int pos, int seq_len, int sliding_window) {
  int apply_mask = (sliding_window > 0 && (l % 2 == 0) ? 1 : 0);
  dim3 blockDim(16, 16); // arbitrary
  dim3 gridDim((pos + 2 + blockDim.x - 1) / blockDim.x, 
               (n_attn_heads + blockDim.y - 1) / blockDim.y);
  multihead_attention_kernel<<<gridDim, blockDim>>>
    (key_cache, value_cache, query, mask, attn_sinks, attn, 
     l, head_dim, n_attn_heads, n_kv_heads, pos, seq_len, apply_mask);
}

__global__ void weighted_sum_kernel(float *tb, float *value_cache, float *attn, 
                                    int seq_len, int n_attn_heads, int n_kv_heads, int pos, int head_dim
) {
  int h = blockDim.y * blockIdx.y + threadIdx.y;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (h >= n_attn_heads || i >= head_dim) return;

  const int kv_dim = head_dim * n_kv_heads;
  const int kv_mul = n_attn_heads / n_kv_heads;

  float sum = 0;
  for (int t = 0; t <= pos; ++t) {
    float a = attn[h * seq_len + t];
    float v = value_cache[t * kv_dim + (h / kv_mul) * head_dim + i];
    sum += a * v;
  }
  tb[h * head_dim + i] = sum; 
}

void getp_weighted_sum(float *tb, float *value_cache, float *attn, 
                       int seq_len, int n_attn_heads, int n_kv_heads, int pos, int head_dim
) {
  dim3 blockDim(16, 16);
  dim3 gridDim((head_dim + blockDim.x - 1) / blockDim.x, 
               (n_attn_heads + blockDim.y - 1) / blockDim.y);
  weighted_sum_kernel<<<gridDim, blockDim>>>
    (tb, value_cache, attn, seq_len, n_attn_heads, n_kv_heads, pos, head_dim);
}

__global__ void max_reduction_kernel(float *in, float *max_in, int size) {
  extern __shared__ float L[];

  int tid = threadIdx.x;
  int offset = 2 * blockDim.x * blockIdx.x;
  int stride = blockDim.x;

  L[tid] = -INFINITY;
  if (tid + offset < size) {
    float t = in[tid + offset];
    L[tid] = MAX(L[tid], t);
  }
  if (tid + offset + stride < size) {
    float t = in[tid + offset + stride];
    L[tid] = MAX(L[tid], t);
  }
  __syncthreads();

  for (stride = stride / 2; stride > 0; stride /= 2) {
    if (tid < stride) L[tid] = MAX(L[tid], L[tid + stride]);
    __syncthreads();
  }

  if (tid == 0) {
    atomicMax(max_in, L[0]);
  }
}

__global__ void compute_exp_and_sum(float *x, float *sum, float max_val, int size) {
  extern __shared__ float L[];

  int tid = threadIdx.x;
  int offset = 2 * blockDim.x * blockIdx.x;
  int stride = blockDim.x;

  L[tid] = 0;
  if (tid + offset < size) {
    float t = x[tid + offset];
    t = expf(t - max_val);
    L[tid] += t;
    x[tid + offset] = t;
  }
  if (tid + offset + stride < size) {
    float t = x[tid + offset + stride];
    t = expf(t - max_val);
    L[tid] += t;
    x[tid + offset + stride] = t;
  }
  __syncthreads();

  for (stride = stride / 2; stride > 0; stride /= 2) {
    if (tid < stride) L[tid] += L[tid + stride];
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(sum, L[0]);
  }
}

__global__ void softmax_normalize_kernel(float *x, float sum, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= size) return;
  x[i] /= sum;
}

void getp_softmax(float *x, int size) {
  float max_val, sum;
  float *dev_max_val, *dev_sum;
  float *dev_data;

  HIP_CHECK(hipMalloc(&dev_data, sizeof(float) * 2));
  dev_max_val = dev_data;
  dev_sum = dev_data + 1;

  max_val = -INFINITY;
  HIP_CHECK(hipMemcpy(dev_max_val, &max_val, sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dev_sum, 0, sizeof(float)));

  // Get max
  {
    dim3 blockDim(1024); // arbitrary
    dim3 gridDim((size + 2 * blockDim.x - 1) / (2 * blockDim.x));
    max_reduction_kernel<<<gridDim, blockDim, sizeof(float) * blockDim.x>>>
      (x, dev_max_val, size);
  }
  HIP_CHECK(hipMemcpy(&max_val, dev_max_val, sizeof(float), hipMemcpyDeviceToHost));
  // Compute exp and sum
  {
    dim3 blockDim(1024); // arbitrary
    dim3 gridDim((size + 2 * blockDim.x - 1) / (2 * blockDim.x));
    compute_exp_and_sum<<<gridDim, blockDim, sizeof(float) * blockDim.x>>>
      (x, dev_sum, max_val, size);
  }
  HIP_CHECK(hipMemcpy(&sum, dev_sum, sizeof(float), hipMemcpyDeviceToHost));
  // Normalize
  {
    dim3 blockDim(1024); // arbitrary
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    softmax_normalize_kernel<<<gridDim, blockDim>>>(x, sum, size);
  }

  HIP_CHECK(hipFree(dev_data));
}

__global__ void vecadd_kernel(float *x, float *y, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= size) return;
  x[i] += y[i];
}

void getp_vecadd(float *x, float *y, int size) {
  dim3 blockDim(1024); // arbitrary
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
  vecadd_kernel<<<gridDim, blockDim>>>(x, y, size);
}

float *getp_forward(Transformer *transformer, DeviceTransformer **dev_transformeres, int token, int pos) {
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  TransformerWeights *dev_w = &dev_transformeres[0]->weights;
  RunState *dev_s = &dev_transformeres[0]->state;

  float *x = s->x;
  float *dev_x = dev_s->x;
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
  // copy to GPU
  HIP_CHECK(hipSetDevice(0));

  // Allocate precomputing sin, cos memory
  float *dev_cos_vals, *dev_sin_vals;
  HIP_CHECK(hipMalloc(&dev_cos_vals, sizeof(float) * (head_dim / 2)));
  HIP_CHECK(hipMalloc(&dev_sin_vals, sizeof(float) * (head_dim / 2)));

  // forward all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {
    HIP_CHECK(hipSetDevice(0));
    // s->t (hidden_dim, )
    // input or output of previous layer is stored in host memory
    // copy to device
    HIP_CHECK(hipMemcpy(dev_x, x, hidden_dim * sizeof(*x), hipMemcpyHostToDevice));
    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);
    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    dev_s->k = dev_s->key_cache + loff + pos * kv_dim;
    dev_s->v = dev_s->value_cache + loff + pos * kv_dim;

    // s->qkv = w->w_qkv * s->t = (head_dim * (n_attn_heads + 2 * n_kv_heads),
    // hidden_dim) * (hidden_dim, ) = head_dim * (n_attn_heads + 2 * n_kv_heads)
    float *dev_w_qkv = dev_w->w_qkv + 1ll * l * hidden_dim * 
                                          (head_dim * p->n_attn_heads + 
                                           2 * head_dim * p->n_kv_heads);
    float *dev_b_qkv = dev_w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 
                                                 2 * head_dim * p->n_kv_heads);
    getp_matmul(dev_s->qkv, dev_s->t, dev_w_qkv, dev_b_qkv, 
                hidden_dim, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
    // Separate q, k, v
    HIP_CHECK(hipMemcpy(dev_s->q, dev_s->qkv, 
                        head_dim * p->n_attn_heads * sizeof(float), 
                        hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(dev_s->k, dev_s->qkv + head_dim * p->n_attn_heads, 
                        head_dim * p->n_kv_heads * sizeof(float), 
                        hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(dev_s->v, dev_s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads, 
                        head_dim * p->n_kv_heads * sizeof(float), 
                        hipMemcpyDeviceToDevice));

    // RoPE relative positional encoding: complex-valued rotate q and k in each
    // head Adapted from
    // https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py#L85
    // RoPE with YaRN scaling adapted from Python code
    float ntk_beta = 32.0f;
    float ntk_alpha = 1.0f;
    getp_compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor, 
                         p->initial_context_length, ntk_beta, ntk_alpha, 
                        dev_cos_vals, dev_sin_vals);
    getp_apply_rotary_emb(dev_s->q, dev_cos_vals, dev_sin_vals, p->n_attn_heads, head_dim);
    getp_apply_rotary_emb(dev_s->k, dev_cos_vals, dev_sin_vals, p->n_kv_heads, head_dim);

    // multihead attention
    getp_multihead_attention(dev_s->key_cache + loff, dev_s->value_cache + loff, dev_s->q, dev_s->mask, dev_w->attn_sinks, dev_s->att, 
                             l, head_dim, p->n_attn_heads, p->n_kv_heads, pos, p->seq_len, p->sliding_window);
    for (int h = 0; h < p->n_attn_heads; ++h) {
      getp_softmax(dev_s->att + h * p->seq_len, pos + 2);
    }
    getp_weighted_sum(dev_s->tb, dev_s->value_cache + loff, dev_s->att, 
                      p->seq_len, p->n_attn_heads, p->n_kv_heads, pos, head_dim);

    // final matmul to get the output of the attention
    float *dev_w_o = dev_w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
    float *dev_b_o = dev_w->b_o + 1ll * l * hidden_dim;
    getp_matmul(dev_s->tb2, dev_s->tb, dev_w_o, dev_b_o, head_dim * p->n_attn_heads, hidden_dim);

    // residual connection back into x
    getp_vecadd(dev_x, dev_s->tb2, hidden_dim);

    // ffn rmsnorm
    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);

    // Copy back to host
    HIP_CHECK(hipMemcpy(s->t, dev_s->t, sizeof(float) * hidden_dim, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(x, dev_x, sizeof(float) * hidden_dim, hipMemcpyDeviceToHost));

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

  // Deallocate precomputing sin, cos
  HIP_CHECK(hipFree(dev_cos_vals));
  HIP_CHECK(hipFree(dev_sin_vals));

  // final rmsnorm
  rmsnorm(x, x, w->rms_out_w, hidden_dim);

  // classifier into logits
  matmul(s->logits, x, w->out, hidden_dim, p->vocab_size);
  return s->logits;
}