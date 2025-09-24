// TODO: Modify this file to optimize end-to-end throughput
#include <cstddef>
#include <cstring>
#include <vector>

#include "collectives.cpp"
#include "getp_eval.cpp"
#include "getp_state_ext.cpp"
#include "getp_transformer.cpp"
#include "getp_transformer.hpp"
#include "getp_barrier.hpp"
#include "profiler.hpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

#include <sched.h>
#include <vector>

#ifndef GETP_RUN
#define GETP_RUN

CollectiveGroup g_world;

DeviceTransformer **dev_transformers;
GPUWorker *workers;
int BATCH_SIZE = 0;

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

  dev_transformers = reinterpret_cast<DeviceTransformer **>(malloc(sizeof(DeviceTransformer *) * n_devices));
  workers = reinterpret_cast<GPUWorker *>(malloc(sizeof(GPUWorker) * n_devices));

  Config *p = &transformer->config;

  if (p->n_experts == 128) {
    // 120b model
    EXPERT_PARALLELISM = n_devices;
    BATCH_SIZE = 512;
  }
  else {
    // 20b model
    EXPERT_PARALLELISM = 1;
    BATCH_SIZE = 896;
  }

  if (n_devices % EXPERT_PARALLELISM) {
    fprintf(stderr, "Number of devices is expected to be divisible by EXPERT_PARALLELISM=%d", EXPERT_PARALLELISM);
    exit(EXIT_FAILURE);
  }

  int experts_per_device = p->n_experts / EXPERT_PARALLELISM;

  for (int i = 0; i < n_devices; i += EXPERT_PARALLELISM) {
    for (int j = 0; j < EXPERT_PARALLELISM; ++j) {
      workers[i + j].device_index = i + j;
      workers[i + j].expert_start = experts_per_device * j;
      workers[i + j].expert_end = experts_per_device * (j + 1);
    }
  }

  std::vector<std::thread> threads(n_devices);

  for (int i = 0; i < n_devices; ++i) {
    dev_transformers[i] = new DeviceTransformer;
    dev_transformers[i]->device_index = i;
    // upload_transformer(transformer, dev_transformers[i], &workers[i]);
    threads[i] = std::thread(upload_transformer,
      transformer, dev_transformers[i], &workers[i]
    );
  }

  for (int i = 0; i < n_devices; ++i) {
    threads[i].join();
  }

  std::vector<int> devices(n_devices);
  for (int i = 0; i < n_devices; ++i) devices[i] = i;
  // Allocate on-device MoE extension state per device

  ext_create(n_devices);
  for (int i = 0; i < n_devices; ++i) {
    ext_alloc_device(i, BATCH_SIZE, EXPERT_PARALLELISM, p);
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

namespace Model_20b {
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
      int *next_gpu = getp_forward_20b(transformer, dev_transformers, worker,
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
    // const char *inputs_seq[BATCH_SIZE];
    // int *outputs_tokens[BATCH_SIZE];
    std::vector<const char*> inputs_seq(BATCH_SIZE);
    std::vector<int*> outputs_tokens(BATCH_SIZE);

    int idx0 = worker->request_start;
    int requests_per_thread = worker->request_end - worker->request_start;

    for (int idx = 0; idx < requests_per_thread; idx += BATCH_SIZE) {
      int B = std::min(BATCH_SIZE, requests_per_thread - idx);
      for (int b = 0; b < B; ++b) {
        inputs_seq[b] = get_str_req_ptr(requests, idx0 + idx + b);
        outputs_tokens[b] = get_tok_gen_ptr(requests, idx0 + idx + b);
      }
      num_token_out += simple_getp_generate(transformer, tokenizer, local_sampler,
                                            worker, inputs_seq.data(), outputs_tokens.data(),
                                            requests->max_seq_len, B);
    }
    *num_token_out_ptr = num_token_out;
  }
}

namespace Model_120b {
  void coop_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                          Sampler * /* sampler */, GPUWorker *workers, 
                          const char *inputs_seq[], int *outputs_tokens[], 
                          long long *num_token_out_ptr, 
                          Barrier &sync_point, int *thread_states,
                          int thread_idx, int steps
  ) {
    PROFILE_FUNCTION();
    // <|start|>: 200006
    // <|end|>: 200007
    // <|return|>: 200002
    // <|message|>: 200008
    // <|channel|>: 200005
    // <|constrain|>: 200003
    // <|endoftext|>: 199999
  
    // Inference here
  
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_idx, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
  
    const char *empty_prompt = "";
    for (int b = 0; b < BATCH_SIZE; ++b) if (inputs_seq[b] == NULL) inputs_seq[b] = empty_prompt;
  
    std::vector<int> nums_prompt_tokens(BATCH_SIZE, 0);
    std::vector<int*> prompts_tokens(BATCH_SIZE);
    for (int b = 0; b < BATCH_SIZE; ++b) {
      prompts_tokens[b] = (int *)malloc((strlen(inputs_seq[b]) + 3) * sizeof(int));
    }
    for (int b = 0; b < BATCH_SIZE; ++b) {
      encode(tokenizer, inputs_seq[b], -1, -1, prompts_tokens[b], &nums_prompt_tokens[b],
             transformer->config.initial_context_length);
      if (nums_prompt_tokens[b] < 1) { fprintf(stderr, "bad prompt\n"); exit(EXIT_FAILURE); }
    }
  
    std::vector<int> next(BATCH_SIZE), token(BATCH_SIZE), epos(BATCH_SIZE, -1), mask(BATCH_SIZE, 1);
    int pos = 0;
    for (int b = 0; b < BATCH_SIZE; ++b) token[b] = prompts_tokens[b][0];
  
    Config *p = &transformer->config;
    if (!p) {
      fprintf(stderr, "Config missing\n");
      exit(EXIT_FAILURE);
    }
  
    while (pos + 1 < steps) {
  
      // forward the transformer to get logits for the next token
      int *next_gpu = getp_forward_120b(
        transformer, dev_transformers, workers, 
        sync_point, thread_idx, 
        token.data(), pos);
  
      pos++;
      for (int b = 0; b < BATCH_SIZE; ++b) {
        if (!mask[b]) continue;
        epos[b] = pos;
        if (pos < nums_prompt_tokens[b])
          next[b] = prompts_tokens[b][pos];
        else {
          next[b] = next_gpu[b];
          outputs_tokens[b][pos - nums_prompt_tokens[b]] = next[b];
        }
      }
  
      for (int b = 0; b < BATCH_SIZE; ++b) {
        if (!mask[b]) continue;
        if (next[b] == 199999 || next[b] == 200002) mask[b] = 0;
      }
      if (thread_states[thread_idx] && is_all_zero(mask.data(), BATCH_SIZE)) {
        thread_states[thread_idx] = 0;
      }
      sync_point.wait();
      if (is_all_zero(thread_states, EXPERT_PARALLELISM)) {
        free(next_gpu);
        break;
      }
  
      // print the token as string, decode it with the Tokenizer object
      // should be removed
      // const char *piece = decode_piece(tokenizer, token, next);
      // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
      // fflush(stdout);
  
      for (int b = 0; b < BATCH_SIZE; ++b) token[b] = next[b];
      free(next_gpu);
    }
  
    for (int b = 0; b < BATCH_SIZE; ++b) {
      if (epos[b] == -1) { fprintf(stderr, "bad epos\n"); exit(EXIT_FAILURE); }
      outputs_tokens[b][epos[b] - nums_prompt_tokens[b] + 1] = -1;
    }
    for (int b = 0; b < BATCH_SIZE; ++b) free(prompts_tokens[b]);
    HIP_CHECK(hipDeviceSynchronize());
  
    long long acc = 0;
    for (int b = 0; b < BATCH_SIZE; ++b) {
      acc += epos[b] - nums_prompt_tokens[b] + 1;
    }
  
    *num_token_out_ptr = acc;
  }
  
  void distribute_requests(Transformer *transformer, 
                           Tokenizer *tokenizer, 
                           Sampler *sampler, 
                           Requests *requests,
                           GPUWorker *workers,
                           long long *num_token_out_ptr,
                           int n_devices
  ) {
    // Sampler *local_sampler = 
    //   reinterpret_cast<Sampler *>(malloc(sizeof(Sampler)));
    // build_sampler(local_sampler, sampler->vocab_size, 
    //   sampler->temperature, sampler->topp, sampler->rng_state + thread_idx);
  
    // long long num_token_out = 0;
    // const char *inputs_seq[EXPERT_PARALLELISM * BATCH_SIZE];
    // int *outputs_tokens[EXPERT_PARALLELISM * BATCH_SIZE];
    // long long nums_token_out[EXPERT_PARALLELISM];
    std::vector<const char *> inputs_seq(EXPERT_PARALLELISM * BATCH_SIZE);
    std::vector<int *> outputs_tokens(EXPERT_PARALLELISM * BATCH_SIZE);
    std::vector<long long> nums_token_out(EXPERT_PARALLELISM * BATCH_SIZE);
  
    long long acc_token_out = 0;
  
    GPUWorker *main_worker = workers;
    int idx0 = main_worker->request_start;
    int requests_per_thread = main_worker->request_end - main_worker->request_start;
    for (int idx = 0; idx < requests_per_thread; idx += EXPERT_PARALLELISM * BATCH_SIZE) {
      for (int b = 0; b < EXPERT_PARALLELISM * BATCH_SIZE; ++b) {
        inputs_seq[b] = get_str_req_ptr(requests, idx0 + idx + b);
        outputs_tokens[b] = get_tok_gen_ptr(requests, idx0 + idx + b);
      }
  
      std::vector<int> thread_states(EXPERT_PARALLELISM, 1);
      std::vector<std::thread> threads(EXPERT_PARALLELISM);
      Barrier sync_point(EXPERT_PARALLELISM);
      for (int i = 0; i < EXPERT_PARALLELISM; ++i) {
        threads[i] = std::thread(coop_getp_generate,
          transformer, tokenizer, (Sampler *)NULL, workers, 
          inputs_seq.data() + (size_t)i * BATCH_SIZE, outputs_tokens.data() + (size_t)i * BATCH_SIZE, 
          &nums_token_out[i],
          std::ref(sync_point), thread_states.data(),
          i, requests->max_seq_len
        );
      }
  
      for (int i = 0; i < EXPERT_PARALLELISM; ++i) {
        threads[i].join();
      }
  
      for (int i = 0; i < EXPERT_PARALLELISM; ++i) {
        acc_token_out += nums_token_out[i];
      }
    }
  
    *num_token_out_ptr = acc_token_out;
  }
}


long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  PROFILE_FUNCTION();

  // Reset timing at the start of inference
  reset_timing_summary();

  int n_devices;
  HIP_CHECK(hipGetDeviceCount(&n_devices));

  // assert(n_devices == EXPERT_PARALLELISM);

  int seq_len_eff = std::min(transformer->config.seq_len, requests->max_seq_len);
  for (int i = 0; i < n_devices; ++i) {
    resize_kv_cache(dev_transformers[i], seq_len_eff);
  }

  int n_parallel_models = n_devices / EXPERT_PARALLELISM;

  int num_reqs_per_device = requests->num_reqs / n_parallel_models;
  for (int i = 0; i < n_devices; i += EXPERT_PARALLELISM) {
    for (int j = 0; j < EXPERT_PARALLELISM; ++j) {
      workers[i + j].request_start = (i / EXPERT_PARALLELISM) * num_reqs_per_device;
      workers[i + j].request_end = (i / EXPERT_PARALLELISM + 1) * num_reqs_per_device;
      if (i == n_parallel_models - 1) workers[i].request_end = requests->num_reqs;
      assert((workers[i].request_end - workers[i].request_start) % (EXPERT_PARALLELISM * BATCH_SIZE) == 0);
    }
  }

  long long num_token_out;

  if (EXPERT_PARALLELISM == 8) {
    // Model 120b
    Model_120b::distribute_requests(transformer, tokenizer, sampler, requests, workers, &num_token_out, n_devices);
  }
  else {
    // Model 20b
    num_token_out = 0;
    std::vector<long long> nums_token_out(n_devices);
    std::vector<std::thread> threads(n_devices);
    for (int i = 0; i < n_devices; ++i) {
      threads[i] =
          std::thread(Model_20b::single_thread_generate, transformer, tokenizer, sampler,
                      requests, &workers[i], &nums_token_out[i], i);
    }

    for (int i = 0; i < n_devices; ++i) {
      threads[i].join();
    }

    for (int i = 0; i < n_devices; ++i) {
      num_token_out += nums_token_out[i];
    }
  }
  
  // Ensure all GPU work is completed before printing timing
  HIP_CHECK(hipDeviceSynchronize());
  
  // Print timing summary at the end of inference
  print_timing_summary();

  return num_token_out;
}

#endif  // GETP_RUN
