// TODO: Modify this file to optimize end-to-end throughput
#include <hip/hip_runtime.h>
#include <hip/driver_types.h>
#include <cstddef>
#include <cstring>
#include <vector>

#include "collectives.cpp"
#include "getp_eval.cpp"
#include "getp_state_ext.cpp"
#include "getp_transformer.cpp"
#include "getp_transformer.hpp"
#include "profiler.hpp"

#ifndef GETP_RUN
#define GETP_RUN

CollectiveGroup g_world;

DeviceTransformer **dev_transformers;
GPUWorker *workers;

#include "getp_forward.cpp"

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...
  int n_devices;
  HIP_CHECK(hipGetDeviceCount(&n_devices));

  dev_transformers = reinterpret_cast<DeviceTransformer **>(
      malloc(sizeof(DeviceTransformer *) * n_devices));
  workers =
      reinterpret_cast<GPUWorker *>(malloc(sizeof(GPUWorker) * n_devices));

  Config *p = &transformer->config;

  for (int i = 0; i < n_devices; ++i) {
    workers[i].device_index = i;

    workers[i].expert_start = 0;
    workers[i].expert_end = p->n_experts;
  }

  for (int i = 0; i < n_devices; ++i) {
    dev_transformers[i] = new DeviceTransformer;
    dev_transformers[i]->device_index = i;
    upload_transformer(transformer, dev_transformers[i], &workers[i]);
  }

  std::vector<int> devices(n_devices);
  for (int i = 0; i < n_devices; ++i) devices[i] = i;
  // Allocate on-device MoE extension state per device

  ext_create(n_devices);
  for (int i = 0; i < n_devices; ++i) {
    ext_alloc_device(i, BATCH_SIZE, p->experts_per_token, p->n_experts,
                     p->hidden_dim, p->intermediate_dim);
  }

  cgCreate(g_world, devices);
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...
  int n_devices;
  HIP_CHECK(hipGetDeviceCount(&n_devices));
  for (int i = 0; i < n_devices; ++i) {
    // cleanup(transformer, dev_transformers[i]);
    free(dev_transformers[i]);
  }
  ext_free_all(n_devices);
  free(dev_transformers);
  cgDestroy(g_world);
}

int is_all_zero(int *a, int size) {
  for (int i = 0; i < size; ++i) {
    if (a[i]) return 0;
  }
  return 1;
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, GPUWorker *worker,
                               const char *inputs_seq[], int *outputs_tokens[],
                               int steps, int B) {
  PROFILE_FUNCTION();

  const char *empty_prompt = "";
  for (int b = 0; b < B; ++b)
    if (inputs_seq[b] == NULL) inputs_seq[b] = empty_prompt;

  std::vector<int> nums_prompt_tokens(B, 0);
  std::vector<int *> prompts_tokens(B);
  for (int b = 0; b < B; ++b) {
    prompts_tokens[b] =
        (int *)malloc((strlen(inputs_seq[b]) + 3) * sizeof(int));
  }
  for (int b = 0; b < B; ++b) {
    encode(tokenizer, inputs_seq[b], -1, -1, prompts_tokens[b],
           &nums_prompt_tokens[b], transformer->config.initial_context_length);
    if (nums_prompt_tokens[b] < 1) {
      fprintf(stderr, "bad prompt\n");
      exit(EXIT_FAILURE);
    }
  }

  std::vector<int> next(B), token(B), epos(B, -1), mask(B, 1);
  int pos = 0;
  for (int b = 0; b < B; ++b) token[b] = prompts_tokens[b][0];

  Config *p = &transformer->config;
  if (!p) {
    fprintf(stderr, "Config missing\n");
    exit(EXIT_FAILURE);
  }

  while (pos + 1 < steps) {
    int *next_gpu = getp_forward(transformer, dev_transformers, worker,
                                 token.data(), pos, B);

    pos++;
    for (int b = 0; b < B; ++b) {
      if (!mask[b]) continue;
      epos[b] = pos;
      if (pos < nums_prompt_tokens[b])
        next[b] = prompts_tokens[b][pos];
      else {
        next[b] = next_gpu[b];
        outputs_tokens[b][pos - nums_prompt_tokens[b]] = next[b];
      }
    }

    for (int b = 0; b < B; ++b) {
      if (!mask[b]) continue;
      if (next[b] == 199999 || next[b] == 200002) mask[b] = 0;
    }
    if (is_all_zero(mask.data(), B)) {
      free(next_gpu);
      break;
    }

    for (int b = 0; b < B; ++b) token[b] = next[b];
    free(next_gpu);
  }

  for (int b = 0; b < B; ++b) {
    if (epos[b] == -1) {
      fprintf(stderr, "bad epos\n");
      exit(EXIT_FAILURE);
    }
    outputs_tokens[b][epos[b] - nums_prompt_tokens[b] + 1] = -1;
  }
  for (int b = 0; b < B; ++b) free(prompts_tokens[b]);
  HIP_CHECK(hipDeviceSynchronize());

  long long acc = 0;
  for (int b = 0; b < B; ++b) acc += epos[b] - nums_prompt_tokens[b] + 1;
  return acc;
}

void single_thread_generate(Transformer *transformer, Tokenizer *tokenizer,
                            Sampler *sampler, Requests *requests,
                            GPUWorker *worker, long long *num_token_out_ptr,
                            int thread_idx) {
  Sampler *local_sampler = reinterpret_cast<Sampler *>(malloc(sizeof(Sampler)));
  build_sampler(local_sampler, sampler->vocab_size, sampler->temperature,
                sampler->topp, sampler->rng_state + thread_idx);

  long long num_token_out = 0;
  const char *inputs_seq[BATCH_SIZE];
  int *outputs_tokens[BATCH_SIZE];

  int idx0 = worker->request_start;
  int requests_per_thread = worker->request_end - worker->request_start;

  for (int idx = 0; idx < requests_per_thread; idx += BATCH_SIZE) {
    int B = std::min(BATCH_SIZE, requests_per_thread - idx);
    for (int b = 0; b < B; ++b) {
      inputs_seq[b] = get_str_req_ptr(requests, idx0 + idx + b);
      outputs_tokens[b] = get_tok_gen_ptr(requests, idx0 + idx + b);
    }
    num_token_out += simple_getp_generate(transformer, tokenizer, local_sampler,
                                          worker, inputs_seq, outputs_tokens,
                                          requests->max_seq_len, B);
  }
  *num_token_out_ptr = num_token_out;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  PROFILE_FUNCTION();

  // Reset timing at the start of inference
  reset_timing_summary();

  int n_devices;
  HIP_CHECK(hipGetDeviceCount(&n_devices));

  int seq_len_eff =
      std::min(transformer->config.seq_len, requests->max_seq_len);
  for (int i = 0; i < n_devices; ++i) {
    resize_kv_cache(dev_transformers[i], seq_len_eff);
  }

  int num_reqs_per_device = requests->num_reqs / n_devices;
  for (int i = 0; i < n_devices; ++i) {
    workers[i].request_start = i * num_reqs_per_device;
    workers[i].request_end = (i + 1) * num_reqs_per_device;
    if (i == n_devices - 1) workers[i].request_end = requests->num_reqs;
    // assert((workers[i].request_end - workers[i].request_start) % BATCH_SIZE
    // == 0);
  }

  std::vector<long long> nums_token_out(n_devices);
  std::vector<std::thread> threads(n_devices);

  for (int i = 0; i < n_devices; ++i) {
    threads[i] =
        std::thread(single_thread_generate, transformer, tokenizer, sampler,
                    requests, &workers[i], &nums_token_out[i], i);
  }

  for (int i = 0; i < n_devices; ++i) {
    threads[i].join();
  }

  // Ensure all GPU work is completed before printing timing
  HIP_CHECK(hipDeviceSynchronize());

  long long num_token_out = 0;
  for (int i = 0; i < n_devices; ++i) {
    num_token_out += nums_token_out[i];
  }

  // Print timing summary at the end of inference
  print_timing_summary();

  return num_token_out;
}

#endif  // GETP_RUN
