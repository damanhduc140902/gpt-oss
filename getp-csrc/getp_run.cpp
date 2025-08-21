// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"
#include "getp_transformer.cpp"

#ifndef GETP_RUN
#define GETP_RUN

Transformer *dev_transformer = new Transformer;

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...
  upload_transformer(transformer, dev_transformer);

  // Test upload phase
  TransformerWeights *w = &transformer->weights;
  TransformerWeights *dev_w = &dev_transformer->weights;
  Config *cfg = &transformer->config;
  int head_dim = cfg->head_dim;
  int n_layers = cfg->n_layers;
  int n_experts = cfg->n_experts;
  // token_embedding_table points to the beginning of the memory region consisting of the TransformerWeights
  // b_mlp2 points the "almost" ending of the memory region
  ssize_t weights_size = (w->b_mlp2 - w->token_embedding_table + 
    1ll * n_layers * n_experts * cfg->hidden_dim) * sizeof(float);
  float *new_data = reinterpret_cast<float *>(malloc(weights_size));
  HIP_CHECK(hipMemcpy(new_data, dev_w->token_embedding_table, weights_size, hipMemcpyDeviceToHost));

  float *ptr = new_data;
  w->token_embedding_table = ptr;
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
  w->out = ptr; // unembedding
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
  w->rms_attn_w = ptr;
  ptr += 1ll * n_layers * cfg->hidden_dim;
  w->rms_ffn_w = ptr;
  ptr += 1ll * n_layers * cfg->hidden_dim;
  w->rms_out_w = ptr;
  ptr += 1ll * cfg->hidden_dim;
  // hey it's qkvqkv, not qqkkvv
  w->w_qkv = ptr;
  ptr += 1ll * n_layers * cfg->hidden_dim *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  w->b_qkv = ptr;
  ptr += 1ll * n_layers *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  w->w_o = ptr;
  ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;
  w->b_o = ptr;
  ptr += 1ll * n_layers * cfg->hidden_dim;
  w->attn_sinks = ptr;
  ptr += 1ll * n_layers * cfg->n_attn_heads;
  w->w_router = ptr;
  ptr += 1ll * n_layers * cfg->hidden_dim * n_experts;
  w->b_router = ptr;
  ptr += 1ll * n_layers * n_experts;
  // hey it's gate_upgate_up, not gategateupup
  w->w_mlp1 = ptr;
  ptr +=
      1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim;
  w->b_mlp1 = ptr;
  ptr += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim;
  w->w_mlp2 = ptr;
  ptr += 1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim;
  w->b_mlp2 = ptr;
  ptr += 1ll * n_layers * n_experts * cfg->hidden_dim;
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...
  cleanup(transformer, dev_transformer);

  // Test upload phase
  TransformerWeights *w = &transformer->weights;
  free(w->token_embedding_table);
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  // Inference here

  const char *empty_prompt = "";
  if (input_seq == NULL) {
    input_seq = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) *
                                     sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, input_seq, 1, 0, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence
  while (pos < steps) {

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);

    // advance the state machine
    pos++;
    if (pos < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
      // save the output token, it will be printed to file
      output_tokens[pos - num_prompt_tokens] = next;
    }

    // data-dependent terminating condition: the BOS (=1) token delimits
    // sequences
    if (next == 1) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    // should be removed
    const char *piece = decode_piece(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);

    token = next;
  }

  // should be removed
  printf("\n");

  // Marker for end of sequence
  output_tokens[pos - num_prompt_tokens + 1] = -1;

  free(prompt_tokens);

  return pos - num_prompt_tokens + 1;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                             output_tokens, requests->max_seq_len);
  }
  return num_token_out;
}

#endif // GETP_RUN
