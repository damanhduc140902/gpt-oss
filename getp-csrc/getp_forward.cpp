#include "getp_transformer.hpp"
#include "profiler.hpp"
#include <cmath>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <malloc.h>

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
  PROFILE_FUNCTION();
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
  HIP_CHECK(hipDeviceSynchronize());
}

template <typename T>
__global__ void matmul_kernel(float *xout, float *x, T *w, T *b, int n, int d) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= d) return;
  float val = 0.f;
  for (int j = 0; j < n; ++j) {
    val += w[1ll * i * n + j] * x[j];
  }
  xout[i] = val + b[i];
}

template <>
__global__ void matmul_kernel<__hip_bfloat16>(float *xout, float *x, __hip_bfloat16 *w, __hip_bfloat16 *b, int n, int d) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= d) return;
  float val = 0.f;
  for (int j = 0; j < n; ++j) {
    val += __bfloat162float(w[1ll * i * n + j]) * x[j];
  }
  xout[i] = val + __bfloat162float(b[i]);
}

template <typename T>
void getp_matmul(float *xout, float *x, T *w, T *b, int n, int d) {
  PROFILE_FUNCTION();
  dim3 blockDim(64);
  dim3 gridDim((d + blockDim.x - 1) / blockDim.x);
  matmul_kernel<T><<<gridDim, blockDim>>>(xout, x, w, b, n, d);
  HIP_CHECK(hipDeviceSynchronize());
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
  PROFILE_FUNCTION();
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
  HIP_CHECK(hipDeviceSynchronize());
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
  PROFILE_FUNCTION();
  dim3 blockDim(16, 16); // arbitrary
  dim3 gridDim((head_dim / 2 + blockDim.x - 1) / blockDim.x, 
               (n_heads + blockDim.y - 1) / blockDim.y);
  apply_rotary_emb_kernel<<<gridDim, blockDim>>>(x, cos, sin, n_heads, head_dim);
  HIP_CHECK(hipDeviceSynchronize());
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
  PROFILE_FUNCTION();
  int apply_mask = (sliding_window > 0 && (l % 2 == 0) ? 1 : 0);
  dim3 blockDim(16, 16); // arbitrary
  dim3 gridDim((pos + 2 + blockDim.x - 1) / blockDim.x, 
               (n_attn_heads + blockDim.y - 1) / blockDim.y);
  multihead_attention_kernel<<<gridDim, blockDim>>>
    (key_cache, value_cache, query, mask, attn_sinks, attn, 
     l, head_dim, n_attn_heads, n_kv_heads, pos, seq_len, apply_mask);
  HIP_CHECK(hipDeviceSynchronize());
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
  PROFILE_FUNCTION();
  dim3 blockDim(16, 16);
  dim3 gridDim((head_dim + blockDim.x - 1) / blockDim.x, 
               (n_attn_heads + blockDim.y - 1) / blockDim.y);
  weighted_sum_kernel<<<gridDim, blockDim>>>
    (tb, value_cache, attn, seq_len, n_attn_heads, n_kv_heads, pos, head_dim);
  HIP_CHECK(hipDeviceSynchronize());
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
  PROFILE_FUNCTION();
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
  HIP_CHECK(hipDeviceSynchronize());
}

__global__ void vecadd_kernel(float *x, float *y, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= size) return;
  x[i] += y[i];
}

void getp_vecadd(float *x, float *y, int size) {
  PROFILE_FUNCTION();
  dim3 blockDim(1024); // arbitrary
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
  vecadd_kernel<<<gridDim, blockDim>>>(x, y, size);
  HIP_CHECK(hipDeviceSynchronize());
}

// MoE GPU kernels
__global__ void topk_kernel(float *router_score, float *topk_values, int *topk_indices, 
                           int n_experts, int experts_per_token) {
  // Simple parallel top-k selection using sorting network approach
  int tid = threadIdx.x;
  if (tid >= experts_per_token) return;
  
  // Initialize with first experts_per_token experts
  if (tid < n_experts) {
    topk_values[tid] = router_score[tid];
    topk_indices[tid] = tid;
  } else {
    topk_values[tid] = -INFINITY;
    topk_indices[tid] = -1;
  }
  
  __syncthreads();
  
  // For remaining experts, replace if larger
  for (int i = experts_per_token; i < n_experts; i++) {
    __syncthreads();
    float current_score = router_score[i];
    
    // Find minimum in current top-k
    if (tid == 0) {
      int min_idx = 0;
      float min_val = topk_values[0];
      for (int j = 1; j < experts_per_token; j++) {
        if (topk_values[j] < min_val) {
          min_val = topk_values[j];
          min_idx = j;
        }
      }
      
      // Replace if current score is larger
      if (current_score > min_val) {
        topk_values[min_idx] = current_score;
        topk_indices[min_idx] = i;
      }
    }
  }
}

__global__ void swiglu_kernel(float *gate_up, float *gate, float *up, 
                             int intermediate_dim, float swiglu_limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= intermediate_dim) return;
  
  float val = gate[i];
  float up_val = up[i];
  const float alpha = 1.702f;
  
  // Clamping
  if (val > swiglu_limit) val = swiglu_limit;
  if (up_val > swiglu_limit) up_val = swiglu_limit;
  if (up_val < -swiglu_limit) up_val = -swiglu_limit;
  
  // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
  val *= (1.0f / (1.0f + expf(-alpha * val)));
  // elementwise multiply with up + bias
  val *= (up_val + 1.0f); // gpt-oss adds an extra bias of 1 to the up layer
  
  gate_up[i] = val;
}

__global__ void split_gate_up_kernel(float *mlp1_out, float *gate, float *up, 
                                     int intermediate_dim) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= intermediate_dim) return;
  
  gate[i] = mlp1_out[2 * i];
  up[i] = mlp1_out[2 * i + 1];
}

__global__ void weighted_aggregate_kernel(float *e_agg, float *expert_out, 
                                         float expert_weight, int hidden_dim) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= hidden_dim) return;
  
  atomicAdd(&e_agg[i], expert_out[i] * expert_weight);
}

void getp_topk(float *topk_values, int *topk_indices, float *router_score, 
               int n_experts, int experts_per_token) {
  PROFILE_FUNCTION();
  dim3 blockDim(experts_per_token); 
  dim3 gridDim(1);
  topk_kernel<<<gridDim, blockDim>>>(router_score, topk_values, topk_indices, 
                                     n_experts, experts_per_token);
  HIP_CHECK(hipDeviceSynchronize());
}

void getp_swiglu(float *gate_up, float *gate, float *up, 
                 int intermediate_dim, float swiglu_limit) {
  PROFILE_FUNCTION();
  dim3 blockDim(256);
  dim3 gridDim((intermediate_dim + blockDim.x - 1) / blockDim.x);
  swiglu_kernel<<<gridDim, blockDim>>>(gate_up, gate, up, intermediate_dim, swiglu_limit);
  HIP_CHECK(hipDeviceSynchronize());
}

void getp_split_gate_up(float *mlp1_out, float *gate, float *up, int intermediate_dim) {
  PROFILE_FUNCTION();
  dim3 blockDim(256);
  dim3 gridDim((intermediate_dim + blockDim.x - 1) / blockDim.x);
  split_gate_up_kernel<<<gridDim, blockDim>>>(mlp1_out, gate, up, intermediate_dim);
  HIP_CHECK(hipDeviceSynchronize());
}

void getp_weighted_aggregate(float *e_agg, float *expert_out, float expert_weight, int hidden_dim) {
  PROFILE_FUNCTION();
  dim3 blockDim(256);
  dim3 gridDim((hidden_dim + blockDim.x - 1) / blockDim.x);
  weighted_aggregate_kernel<<<gridDim, blockDim>>>(e_agg, expert_out, expert_weight, hidden_dim);
  HIP_CHECK(hipDeviceSynchronize());
}

float *getp_forward(Transformer *transformer, DeviceTransformer **dev_transformeres, int token, int pos) {
  PROFILE_FUNCTION();
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  DeviceTransformerWeights *dev_w = &dev_transformeres[0]->weights;
  RunState *dev_s = &dev_transformeres[0]->state;

  int n_devices;
  HIP_CHECK(hipGetDeviceCount(&n_devices));

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
  // memcpy(x, content_row, hidden_dim * sizeof(*x));
  // copy to GPU
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemcpy(dev_x, content_row, hidden_dim * sizeof(*x), hipMemcpyHostToDevice));

  // Allocate precomputing sin, cos memory
  float *dev_cos_vals, *dev_sin_vals;
  HIP_CHECK(hipMalloc(&dev_cos_vals, sizeof(float) * (head_dim / 2)));
  HIP_CHECK(hipMalloc(&dev_sin_vals, sizeof(float) * (head_dim / 2)));

  // forward all the layers
  for (unsigned long long l = 0; l < p->n_layers; l++) {
    HIP_CHECK(hipSetDevice(0));
    // s->t (hidden_dim, )
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
    getp_matmul<float>(dev_s->qkv, dev_s->t, dev_w_qkv, dev_b_qkv, 
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
    getp_matmul<float>(dev_s->tb2, dev_s->tb, dev_w_o, dev_b_o, head_dim * p->n_attn_heads, hidden_dim);

    // residual connection back into x
    getp_vecadd(dev_x, dev_s->tb2, hidden_dim);

    // ffn rmsnorm
    getp_rmsnorm(dev_s->t, dev_x, dev_w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);

    // MoE GPU implementation with multi-GPU expert sharding
    // Compute router_score on device 0 (router weights are on device 0)
    float *dev_w_router = dev_w->w_router + 1ll * l * hidden_dim * n_experts;
    float *dev_b_router = dev_w->b_router + 1ll * l * n_experts;
    getp_matmul(dev_s->router_score, dev_s->t, dev_w_router, dev_b_router, 
                hidden_dim, n_experts);
    
    // Select top-k experts on GPU
    getp_topk(dev_s->topk_v, dev_s->topk_i, dev_s->router_score, 
              n_experts, p->experts_per_token);
    
    // Normalize selected experts using softmax
    getp_softmax(dev_s->topk_v, p->experts_per_token);
    
    // Initialize aggregation buffer to zero on device 0
    HIP_CHECK(hipMemset(dev_s->e_agg, 0, hidden_dim * sizeof(float)));
    
    // Copy topk results back to host to determine expert distribution
    HIP_CHECK(hipMemcpy(s->topk_v, dev_s->topk_v, p->experts_per_token * sizeof(float), 
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(s->topk_i, dev_s->topk_i, p->experts_per_token * sizeof(int), 
                        hipMemcpyDeviceToHost));

    // Process experts across multiple GPUs
    for (int device_id = 0; device_id < n_devices; device_id++) {
      HIP_CHECK(hipSetDevice(device_id));
      
      // Determine expert range for this device
      int experts_per_device = n_experts / n_devices;
      int expert_start = device_id * experts_per_device;
      int expert_end = expert_start + experts_per_device;
      
      // Get device transformer for this GPU
      DeviceTransformerWeights *curr_dev_w = &dev_transformeres[device_id]->weights;
      RunState *curr_dev_s = &dev_transformeres[device_id]->state;
      
      // Copy normalized input to current device if not device 0
      if (device_id != 0) {
        // Copy through host memory for cross-device transfer
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMemcpy(s->t, dev_s->t, sizeof(float) * hidden_dim, hipMemcpyDeviceToHost));
        HIP_CHECK(hipSetDevice(device_id));
        HIP_CHECK(hipMemcpy(curr_dev_s->t, s->t, sizeof(float) * hidden_dim, hipMemcpyHostToDevice));
      }
      
      // Initialize device aggregation buffer
      HIP_CHECK(hipMemset(curr_dev_s->e_agg, 0, hidden_dim * sizeof(float)));
      
      // Process experts assigned to this device
      for (int idx = 0; idx < p->experts_per_token; idx++) {
        int expert_id = s->topk_i[idx];
        float expert_weight = s->topk_v[idx];
        
        // Check if this expert belongs to current device
        if (expert_id >= expert_start && expert_id < expert_end) {
          // Local expert index within this device
          int local_expert_id = expert_id - expert_start;
          
          // Get expert weights pointers (local indexing)
          __hip_bfloat16 *dev_w_mlp1 = curr_dev_w->w_mlp1 + 1ll * (l * experts_per_device + local_expert_id) *
                                                                (2 * p->intermediate_dim) * hidden_dim;
          __hip_bfloat16 *dev_b_mlp1 = curr_dev_w->b_mlp1 + 1ll * (l * experts_per_device + local_expert_id) * 
                                                                (2 * p->intermediate_dim);
          
          // MLP layer 1: gate_up projection
          getp_matmul<__hip_bfloat16>(curr_dev_s->mlp1_out, curr_dev_s->t, dev_w_mlp1, dev_b_mlp1, 
                                      hidden_dim, 2 * p->intermediate_dim);
          
          // Split into gate and up
          getp_split_gate_up(curr_dev_s->mlp1_out, curr_dev_s->gate, curr_dev_s->up, 
                             p->intermediate_dim);
          
          // SwiGLU non-linearity
          getp_swiglu(curr_dev_s->gate_up, curr_dev_s->gate, curr_dev_s->up, 
                      p->intermediate_dim, p->swiglu_limit);
          
          // MLP layer 2: down projection
          __hip_bfloat16 *dev_w_mlp2 = curr_dev_w->w_mlp2 + 1ll * (l * experts_per_device + local_expert_id) * 
                                                                hidden_dim * p->intermediate_dim;
          __hip_bfloat16 *dev_b_mlp2 = curr_dev_w->b_mlp2 + 1ll * (l * experts_per_device + local_expert_id) * 
                                                                hidden_dim;
          getp_matmul<__hip_bfloat16>(curr_dev_s->tb2, curr_dev_s->gate_up, dev_w_mlp2, dev_b_mlp2, 
                                      p->intermediate_dim, hidden_dim);
          
          // Aggregate expert output with weight on current device
          getp_weighted_aggregate(curr_dev_s->e_agg, curr_dev_s->tb2, expert_weight, hidden_dim);
        }
      }
      
      // Copy results back to device 0 for final aggregation if not device 0
      if (device_id != 0) {
        // Copy device result to host, then add to device 0
        float *temp_result = (float*)malloc(hidden_dim * sizeof(float));
        HIP_CHECK(hipMemcpy(temp_result, curr_dev_s->e_agg, sizeof(float) * hidden_dim, hipMemcpyDeviceToHost));
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipMemcpy(dev_s->tb2, temp_result, sizeof(float) * hidden_dim, hipMemcpyHostToDevice));
        getp_vecadd(dev_s->e_agg, dev_s->tb2, hidden_dim);
        free(temp_result);
      }
    }
    
    // Ensure we're back on device 0
    HIP_CHECK(hipSetDevice(0));
    
    // Residual connection on device 0
    getp_vecadd(dev_x, dev_s->e_agg, hidden_dim);
  }

  // Deallocate precomputing sin, cos
  HIP_CHECK(hipFree(dev_cos_vals));
  HIP_CHECK(hipFree(dev_sin_vals));

  // Copy output result to host
  HIP_CHECK(hipMemcpy(x, dev_x, sizeof(float) * hidden_dim, hipMemcpyDeviceToHost));

  // TODO: put 2 remaining below operations to GPU

  // final rmsnorm
  rmsnorm(x, x, w->rms_out_w, hidden_dim);

  // classifier into logits
  matmul(s->logits, x, w->out, hidden_dim, p->vocab_size);
  
  // Ensure all GPU work is complete before returning
  HIP_CHECK(hipDeviceSynchronize());
  
  return s->logits;
}