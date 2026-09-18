<div align="center">

<img alt="gpt-oss - pure C++ and HIP inference on AMD GPUs" src="docs/assets/banner.svg" width="100%">

<p>
  <img src="https://img.shields.io/badge/C%2B%2B-17-57606a?style=flat-square&labelColor=24292f" alt="C++17">
  <img src="https://img.shields.io/badge/HIP-ROCm%206.2-57606a?style=flat-square&labelColor=24292f" alt="HIP / ROCm 6.2">
  <img src="https://img.shields.io/badge/GPU-8%C3%97%20AMD%20MI250-57606a?style=flat-square&labelColor=24292f" alt="8x AMD MI250">
  <img src="https://img.shields.io/badge/dependencies-none-57606a?style=flat-square&labelColor=24292f" alt="No external libraries">
  <img src="https://img.shields.io/github/last-commit/damanhduc140902/gpt-oss?style=flat-square&label=commit&color=57606a&labelColor=24292f" alt="Last commit">
</p>

[Overview](#overview) · [Quick Start](#quick-start) · [Results](#results) · [How It Works](#how-it-works) · [Blog](#blog) · [Docs](docs/) · [License](LICENSE)

</div>

## Overview

OpenAI's `gpt-oss-20b` and `gpt-oss-120b` are already served by llama.cpp, vLLM and SGLang — all of them
built around CUDA and around large dependency stacks. This project takes the opposite approach. It runs
both models on AMD GPUs in **plain C++ and HIP, with no external libraries at all**: no rocBLAS, no
hipBLAS, no RCCL, no MPI. Every kernel, every collective and the tokenizer are written from scratch in
this repository. The only dependency is the HIP runtime itself.

It began from [llama2.c](https://github.com/karpathy/llama2.c) and grew into a complete inference system.
On a single node of 8 AMD MI250 GPUs it serves **58,309 tokens per second on the 20B model and 19,463 on
the 120B model**, while keeping the generated text faithful to a CPU reference.

Two things are measured, and both have to hold:

- **Throughput** — output tokens per second, as high as the hardware allows.
- **Fidelity** — the optimized system must say what an unoptimized CPU implementation would say. Scored
  with METEOR and BERTScore, which must stay above 0.3 and 0.9.

### The models

Both checkpoints share every width; they differ in depth and in how many experts each layer holds.

| | `gpt-oss-20b` | `gpt-oss-120b` |
| --- | ---: | ---: |
| Transformer layers | 24 | 36 |
| Experts per layer | 32 | 128 |
| Experts per token | 4 | 4 |
| Attention heads | 64 | 64 |
| Key/value heads (GQA) | 8 | 8 |
| Head dimension | 64 | 64 |
| Hidden size | 2880 | 2880 |
| Vocabulary | 201,088 | 201,088 |

---

## Quick Start

You need an AMD GPU node with ROCm and `hipcc` on your `PATH`, Python 3.10+, and disk for the converted
weights — roughly 13 GB for 20B and 65 GB for 120B. The build takes well under a minute.

### 1. Clone and set up

```bash
git clone https://github.com/damanhduc140902/gpt-oss.git
cd gpt-oss

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the weights

Download the `safetensors` and convert them to the flat `.bin` file the C++ loader reads. The conversion
dequantizes the FP4 tensors and flattens everything into one blob.

```bash
python tools/model_export/gpt-oss-20b/export_model_bin.py      # or gpt-oss-120b/
```

| Model | Hugging Face | Export script |
| --- | --- | --- |
| `gpt-oss-20b` | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | [`tools/model_export/gpt-oss-20b/`](tools/model_export/gpt-oss-20b/) |
| `gpt-oss-120b` | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | [`tools/model_export/gpt-oss-120b/`](tools/model_export/gpt-oss-120b/) |
| `gpt-oss-7m` | [tiny-random/gpt-oss](https://huggingface.co/tiny-random/gpt-oss) | [`tools/model_export/gpt-oss-7m/`](tools/model_export/gpt-oss-7m/) |

The 7M model is randomly initialised and says nothing sensible, but it is two layers wide and loads in a
second — use it to prove the build works before committing to a 65 GB download.

### 3. Build

```bash
./run.sh build          # -O3, the one the benchmarks use
./run.sh build omp      # -O3 with OpenMP and -march=native
./run.sh build debug    # -O0 with symbols
```

`run.sh` wraps the `Makefile`; `make runfast`, `make runomp` and the rest still work. Building also
regenerates `tokenizer.bin` from the `o200k_harmony` encoding.

### 4. Run one prompt

Start here. If this prints an answer, everything is wired up:

```bash
./run.sh run gpt-oss-20b.bin -m generate -i "1+1="
```

Something longer, sampled rather than greedy:

```bash
./run.sh run gpt-oss-20b.bin -m generate -i "Once upon a time" -n 256 -t 0.8 -p 0.95
```

Interactive chat, optionally with a system prompt:

```bash
./run.sh run gpt-oss-20b.bin -m chat
./run.sh run gpt-oss-20b.bin -m chat -y "You are a concise assistant."
```

### 5. Run a batch

`getp` is the batch serving mode, and the one every throughput number on this page comes from.

**It accepts exactly one request count, and it is not negotiable.** `distribute_requests` asserts that the
file holds `n_devices × BATCH_SIZE` requests — 1536 per GPU for the 20B model, 768 per GPU for the 120B —
so on eight GPUs that is 12,288 and 6,144 respectively. Those are the two numbers in the
[results table](#results). Anything else aborts at
[`src/getp/run.cpp:268`](src/getp/run.cpp#L268), and **none of the input files shipped in this repository
is the right length**; they declare 32, 256, 448, 3584 and 4096. Build one that fits:

```bash
./run.sh mkinput 20b 8 input_20b.txt        # 12288 requests
./run.sh run gpt-oss-20b.bin -m getp -i input_20b.txt -o out.txt
```

The output file holds token ids, not text. Turn them back into words:

```bash
./run.sh decode -i out.txt                  # every completion
./run.sh decode 0 -i out.txt                # just the first
```

Every visible GPU is used automatically, up to 8. To use fewer, regenerate the input for that count —
the required size follows the device count:

```bash
./run.sh mkinput 20b 2 input_2gpu.txt       # 3072 requests
HIP_VISIBLE_DEVICES=0,1 ./run.sh run gpt-oss-20b.bin -m getp -i input_2gpu.txt -o out.txt
```

[`tools/make_getp_input.py`](tools/make_getp_input.py) repeats a pool of prompts to reach the required
length; pass `-s your_prompts.txt` to use your own. Repetition is fine for a throughput measurement but
makes the output redundant for quality scoring. The file format is a line giving the request count,
then one prompt per line.

> **120B on fewer than 8 GPUs does not work.** The 120B path sets `EXPERT_PARALLELISM = n_devices`, and the
> dispatch at [`src/getp/run.cpp:524`](src/getp/run.cpp#L524) only routes to the 120B implementation when
> that equals 8. On any other count it silently falls through to the 20B path and fails an assert there.

### 6. Check the tokenizer

The C++ tokenizer is a reimplementation, so it is worth confirming it agrees with `tiktoken`:

```bash
./run.sh tok "Hello world"          # -> 13225 2375

python3 tests/test_tokenizer.py \
  --bin ./test_tokenizer \
  --tok ./tokenizer.bin \
  --prompts tests/data/input.txt \
  --verbose
```

### 7. Score the output

```bash
./run.sh eval 20b                   # or 120b
```

This decodes `tests/submission/` against `tests/references/` and reports METEOR and BERTScore F1, failing
if either falls under the threshold in [`tests/threshold.json`](tests/threshold.json). A GPU is strongly
recommended; see [`tests/README.md`](tests/README.md).

### All options

```bash
./run.sh --help
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `-m` | mode: `generate`, `chat` or `getp` | `generate` |
| `-i` | prompt, or input file in `getp` mode | — |
| `-o` | output file, `getp` mode only | — |
| `-n` | steps to run; `0` means the full context | `1024` |
| `-t` | temperature; `0.0` is greedy and reproducible | `0.0` |
| `-p` | top-p (nucleus) sampling | `0.9` |
| `-s` | random seed | `time(NULL)` |
| `-y` | system prompt, `chat` mode only | — |
| `-z` | custom tokenizer path | `tokenizer.bin` |

---

## Results

Measured on one node of 8× AMD MI250 in batch (`getp`) mode.

| Model | Requests | Warm-up (s) | Inference (s) | Throughput (TPS) | METEOR | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-oss-20b` | 12288 | 26 | 209 | **58309** | 0.531 | 0.978 |
| `gpt-oss-120b` | 6144 | 263 | 312 | **19463** | 0.561 | 0.981 |

Where those numbers came from, one optimisation at a time on the 20B model:

| | Throughput | Change |
| --- | ---: | ---: |
| Starting point | 33979 | — |
| GEMM tile tuning and per-tile attention softmax | 42540 | +25.2% |
| Attention rewritten on `mfma_f32_16x16x16bf16_1k` | 50464 | +18.6% |
| LDS tiles stored k-contiguous in all five GEMMs | 56434 | +11.8% |
| Per-step allocations, events and a dead 35 MB memset removed | 57019 | +1.0% |
| Argmax made independent of block arrival order | 57343 | +0.6% |
| mlp2 block 64 -> 96 with a matching register budget | **58309** | +1.7% |

Warm-up is dominated by reading the checkpoint off disk, so it depends on whether the file is still in
the page cache; it is not part of what the optimisation work changed.

Throughput is aggregate across all eight GPUs, not single-stream latency. The quality gates are METEOR
0.3 and BERTScore 0.9; both models clear them several times over, so the speed was not bought with
degraded output.

Every figure in that table comes from one run each, and the completions those scores were computed
from are committed in [`tests/submission/`](tests/submission/) — so the numbers can be checked without
a GPU, by scoring the files in the repository. `./run.sh eval 20b` reproduces the last two columns.

Repeating a run moves throughput by well under a percent, but it moves the completions a great deal.
The engine is not bit-deterministic — the same prompt at a different batch index takes a different path
through the expert grouping — and greedy decoding turns any difference into a different trajectory. Two
runs of the *same binary* at 8 GPUs and 1024 steps agree on only about 68% of tokens, while the quality
scores stay put. That is worth knowing before using token agreement to check a change: at this scale it
cannot tell a real bug from a rounding difference, and METEOR and BERTScore are the gates that can.

---

## How It Works

| | What it does | Detail |
| --- | --- | --- |
| **Model** | The architecture as this code implements it, the `.bin` weight format and its FP4 dequantization, the `o200k_harmony` tokenizer and the sampling path. | [MODEL.md](docs/MODEL.md) |
| **Kernels** | A blocktiled GEMM issued entirely on the matrix cores with double buffering, a FlashAttention-style attention kernel that exploits grouped-query attention, and one fused kernel that serves every expert in a single launch. | [KERNELS.md](docs/KERNELS.md) |
| **Parallelism** | Expert × Data parallelism across the GPUs, with all-gather and reduce-scatter built from peer-to-peer copies alone and ordered so PCIe runs full-duplex. | [PARALLELISM.md](docs/PARALLELISM.md) |
| **Serving** | The `getp` batch runtime: how requests are batched and scheduled, what the warm-up buys, and where the throughput actually comes from. | [SERVING.md](docs/SERVING.md) |

### Code structure

```plain
gpt-oss/
├── include/            # shared headers
├── src/
│   ├── run.cpp         # entry point, CLI, sampling, chat and generate loops
│   ├── tokenizer.cpp   # o200k_harmony tokenizer
│   ├── decode.cpp      # token ids -> text utility
│   ├── getp/           # batch serving: runtime, scheduling, collectives
│   └── hip/            # HIP kernels: GEMM, attention, MoE, norms
├── tests/              # METEOR / BERTScore evaluation, prompts, references
├── tools/              # weight and tokenizer conversion, getp input generator
├── docs/               # engineering write-up
└── run.sh              # build / run / mkinput / decode / tok / eval wrapper
```

---

## Blog

A write-up of how this was built — what was tried, what was discarded, and where the throughput actually
came from — is on its way. Until it lands, [`docs/`](docs/) carries the same material in shorter form.

---

## Acknowledgments

This project was part of the GPU Engineer Training Program, a collaboration between
[Moreh](https://www.linkedin.com/company/moreh-vietnam/) and
[THUNDER Research Group](http://snuvm.snu.ac.kr/) (Seoul National University). It started from
[llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy.
