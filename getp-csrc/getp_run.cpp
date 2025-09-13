// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"
#include "getp_transformer.cpp"
#include "getp_transformer.hpp"
#include "profiler.hpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include "collectives.cpp"
#include "getp_state_ext.cpp"
#include <vector>

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

  if (n_devices % EXPERT_PARALLELISM) {
    fprintf(stderr, "Number of devices is expected to be divisible by EXPERT_PARALLELISM=%d", EXPERT_PARALLELISM);
    exit(EXIT_FAILURE);
  }

  dev_transformers = reinterpret_cast<DeviceTransformer **>(malloc(sizeof(DeviceTransformer *) * n_devices));
  workers = reinterpret_cast<GPUWorker *>(malloc(sizeof(GPUWorker) * n_devices));

  Config *p = &transformer->config;

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
    ext_alloc_device(i, BATCH_SIZE, p->experts_per_token, p->n_experts, p->hidden_dim);
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
                               Sampler *sampler, GPUWorker *worker, const char *inputs_seq[],
                               int *outputs_tokens[], int steps) {
  PROFILE_FUNCTION();
  // <|start|>: 200006
  // <|end|>: 200007
  // <|return|>: 200002
  // <|message|>: 200008
  // <|channel|>: 200005
  // <|constrain|>: 200003
  // <|endoftext|>: 199999

  // Inference here

  const char *empty_prompt = "";
  for (int b = 0; b < BATCH_SIZE; ++b) {
    if (inputs_seq[b] == NULL) {
      inputs_seq[b] = empty_prompt;
    }
  }

  // encode the (string) prompt into tokens sequence
  int nums_prompt_tokens[BATCH_SIZE];
  memset(nums_prompt_tokens, 0, sizeof(int) * BATCH_SIZE);
  int *prompts_tokens[BATCH_SIZE];
  for (int b = 0; b < BATCH_SIZE; ++b) {
    prompts_tokens[b] = (int *)malloc((strlen(inputs_seq[b]) + 3) *
                                      sizeof(int)); // +3 for '\0', ?BOS, ?EOS 
  }

  for (int b = 0; b < BATCH_SIZE; ++b) {
    encode(tokenizer, inputs_seq[b], -1, -1, prompts_tokens[b], &nums_prompt_tokens[b],
           transformer->config.initial_context_length);  
    if (nums_prompt_tokens[b] < 1) {
      fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
      exit(EXIT_FAILURE);
    }
  }

  // start the main loop
  int next[BATCH_SIZE];  // will store the next token in the sequence
  int token[BATCH_SIZE]; // kick off with the first token in the prompt
  int pos = 0;           // position in the sequence
  int epos[BATCH_SIZE];  // position where the sequence of each batch ends
  int mask[BATCH_SIZE];  // store running state of each batch (0 - done, 1 - otherwise)
  for (int b = 0; b < BATCH_SIZE; ++b) {
    token[b] = prompts_tokens[b][0];
    epos[b] = -1;
    mask[b] = 1;
  }

  // print the very first token
  // should be removed
  // const char *first_piece = decode_piece(tokenizer, 200006, token);
  // safe_printf(first_piece);
  // fflush(stdout);

  Config *p = &transformer->config;
  if (!p) {
    fprintf(stderr, "something is wrong, Config does not exist\n");
    exit(EXIT_FAILURE);
  }

  while (pos + 1 < steps) {

    // forward the transformer to get logits for the next token
    int *next_gpu = getp_forward(transformer, dev_transformers, worker, token, pos);


    // advance the state machine
    pos++;
    for (int b = 0; b < BATCH_SIZE; ++b) {
      if (!mask[b]) continue;
      epos[b] = pos;
      if (pos < nums_prompt_tokens[b]) {
        next[b] = prompts_tokens[b][pos];
      } else {
        next[b] = next_gpu[b];
        outputs_tokens[b][pos - nums_prompt_tokens[b]] = next[b];
      }
    }
    

    // data-dependent terminating condition: the EOS (=199999 or =200002) token
    // delimits sequences
    for (int b = 0; b < BATCH_SIZE; ++b) {
      if (!mask[b]) continue;
      if (next[b] == 199999 || next[b] == 200002) {
        mask[b] = 0;
      }
    }
    if (is_all_zero(mask, BATCH_SIZE)) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    // should be removed
    // const char *piece = decode_piece(tokenizer, token, next);
    // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    // fflush(stdout);

    for (int b = 0; b < BATCH_SIZE; ++b) {
      token[b] = next[b];
    }

    free(next_gpu);
  }

  // should be removed
  // printf("\n");

  // Marker for end of sequence
  for (int b = 0; b < BATCH_SIZE; ++b) {
    if (epos[b] == -1) {
      fprintf(stderr, "something is wrong, epos can not recieve value -1\n");
      exit(EXIT_FAILURE);
    }
    outputs_tokens[b][epos[b] - nums_prompt_tokens[b] + 1] = -1;
  }

  for (int b = 0; b < BATCH_SIZE; ++b) {
    free(prompts_tokens[b]);
  }
  HIP_CHECK(hipDeviceSynchronize());

  int acc = 0;
  for (int b = 0; b < BATCH_SIZE; ++b) {
    acc += epos[b] - nums_prompt_tokens[b] + 1;
  }

  return acc;
}

void single_thread_generate(Transformer *transformer, 
                            Tokenizer *tokenizer, 
                            Sampler *sampler, 
                            Requests *requests,
                            GPUWorker *workers,
                            long long *num_token_out_ptr,
                            int thread_idx
) {
  Sampler *local_sampler = 
    reinterpret_cast<Sampler *>(malloc(sizeof(Sampler)));
  build_sampler(local_sampler, sampler->vocab_size, 
    sampler->temperature, sampler->topp, sampler->rng_state + thread_idx);

  long long num_token_out = 0;
  const char *inputs_seq[BATCH_SIZE];
  int *outputs_tokens[BATCH_SIZE];

  GPUWorker *main_worker = workers;
  int idx0 = main_worker->request_start;
  int requests_per_thread = main_worker->request_end - main_worker->request_start;
  for (int idx = 0; idx < requests_per_thread; idx += BATCH_SIZE) {
    for (int b = 0; b < BATCH_SIZE; ++b) {
      inputs_seq[b] = get_str_req_ptr(requests, idx0 + idx + b);
      outputs_tokens[b] = get_tok_gen_ptr(requests, idx0 + idx + b);
    }
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, local_sampler, workers, inputs_seq,
                             outputs_tokens, requests->max_seq_len);
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
      assert((workers[i].request_end - workers[i].request_start) % BATCH_SIZE == 0);
    }
  }

  std::vector<long long> nums_token_out(n_parallel_models);
  std::vector<std::thread> threads(n_parallel_models);

  for (int i = 0; i < n_devices; i += EXPERT_PARALLELISM) {
    threads[i / EXPERT_PARALLELISM] = std::thread(single_thread_generate, 
      transformer, tokenizer, sampler, requests, &workers[i], &nums_token_out[i / EXPERT_PARALLELISM], i / EXPERT_PARALLELISM);
  }

  for (int i = 0; i < n_parallel_models; ++i) {
    threads[i].join();
  }
  
  // Ensure all GPU work is completed before printing timing
  HIP_CHECK(hipDeviceSynchronize());

  long long num_token_out = 0;
  for (int i = 0; i < n_parallel_models; ++i) {
    num_token_out += nums_token_out[i];
  }
  
  // Print timing summary at the end of inference
  print_timing_summary();
  
  return num_token_out;
}

#endif // GETP_RUN
