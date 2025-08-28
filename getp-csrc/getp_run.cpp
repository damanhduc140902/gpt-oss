// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"
#include "getp_transformer.cpp"
#include "getp_transformer.hpp"
#include "profiler.hpp"
#include <cstring>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include "collectives.cpp"
#include <vector>

#ifndef GETP_RUN
#define GETP_RUN

CollectiveGroup g_world;

DeviceTransformer **dev_transformers;
float *logits_output;

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

  for (int i = 0; i < n_devices; ++i) {
    dev_transformers[i] = new DeviceTransformer;
    dev_transformers[i]->device_index = i;
    upload_transformer(transformer, dev_transformers[i]);
  }

  int vocab_size = (transformer->config).vocab_size;
  logits_output = reinterpret_cast<float *>
    (malloc(sizeof(float) * BATCH_SIZE * vocab_size));

  std::vector<int> devices(n_devices);
  for (int i = 0; i < n_devices; ++i) devices[i] = i;
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

  free(logits_output);

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
                               Sampler *sampler, const char *inputs_seq[],
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

  while (pos < steps) {

    // forward the transformer to get logits for the next token
    float *logits = getp_forward(transformer, dev_transformers, token, pos);
    // float *logits = forward(transformer, token, pos);

    // advance the state machine
    pos++;
    for (int b = 0; b < BATCH_SIZE; ++b) {
      if (!mask[b]) continue;
      epos[b] = pos;
      if (pos < nums_prompt_tokens[b]) {
        // if we are still processing the input prompt, force the next prompt
        // token
        next[b] = prompts_tokens[b][pos];
      } else {
        // otherwise sample the next token from the logits
        next[b] = sample(sampler, logits + b * p->vocab_size);
        // save the output token, it will be printed to file
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

    // free(logits);
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

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  PROFILE_FUNCTION();
  
  // Reset timing at the start of inference
  reset_timing_summary();
  
  long long num_token_out = 0;
  const char *inputs_seq[BATCH_SIZE];
  int *outputs_tokens[BATCH_SIZE];
  for (int idx = 0; idx < requests->num_reqs; idx += BATCH_SIZE) {
    for (int b = 0; b < BATCH_SIZE; ++b) {
      inputs_seq[b] = get_str_req_ptr(requests, idx + b);
      outputs_tokens[b] = get_tok_gen_ptr(requests, idx + b);
    }
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, sampler, inputs_seq,
                             outputs_tokens, requests->max_seq_len);
  }
  
  // Ensure all GPU work is completed before printing timing
  HIP_CHECK(hipDeviceSynchronize());
  
  // Print timing summary at the end of inference
  print_timing_summary();
  
  return num_token_out;
}

#endif // GETP_RUN
