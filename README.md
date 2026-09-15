<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img alt="gpt-oss on AMD GPUs" src="docs/assets/banner-light.svg" width="80%">
</picture>

<p>
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C.svg" alt="C++17">
  <img src="https://img.shields.io/badge/HIP-ROCm-ED1C24.svg" alt="HIP / ROCm">
  <img src="https://img.shields.io/badge/GPU-8%C3%97%20AMD%20MI250-F26722.svg" alt="8x AMD MI250">
  <img src="https://img.shields.io/badge/throughput-33.9k%20TPS-success.svg" alt="33.9k TPS">
  <img src="https://img.shields.io/github/last-commit/damanhduc140902/gpt-oss?label=commit" alt="Last commit">
</p>

[Overview](#overview) | [Quick Start](#quick-start) | [Results](#results) | [How It Works](#how-it-works) | [Blog](#blog) | [Docs](docs/)

<img alt="gpt-oss model overview" src="docs/assets/model-overview.png" width="100%">

</div>

## Overview

OpenAI's `gpt-oss-20b` and `gpt-oss-120b` are served today by llama.cpp, vLLM and SGLang, but those
engines are built around CUDA and around large dependency stacks. This project takes the opposite
approach: it runs both models on AMD GPUs in **plain C++ and HIP, with no external libraries at all** —
no rocBLAS, no hipBLAS, no RCCL, no MPI. Every kernel, every collective and the tokenizer are written
from scratch in this repository.

It began from [llama2.c](https://github.com/karpathy/llama2.c) and grew into a complete inference
system: batching, multi-GPU expert parallelism, peer-to-peer collectives, a FlashAttention-style
attention kernel and matrix-core GEMMs. On a single node of 8 AMD MI250 GPUs it reaches **33.9k tokens
per second on the 20B model and 13.0k on the 120B model**, while keeping the generated text faithful to
a CPU reference.

Two things are measured, and both matter:

- **Throughput** — output tokens per second, as high as the hardware allows.
- **Fidelity** — the optimized system must say what an unoptimized CPU implementation would say.
  This is scored with METEOR and BERTScore, which have to stay above 0.3 and 0.9.

If you want the engineering detail rather than the instructions, go to [How It Works](#how-it-works)
or straight to [`docs/`](docs/).

---

## Quick Start

You need an AMD GPU node with ROCm and `hipcc` on your `PATH`, Python 3.10+, and enough disk for the
converted weights (about 13 GB for 20B, 65 GB for 120B). The build itself takes under a minute.

### 1. Clone

```bash
git clone https://github.com/damanhduc140902/gpt-oss.git
cd gpt-oss
pip install -r requirements.txt
```

### 2. Get the weights

Download the `safetensors` from the
[OpenAI gpt-oss collection](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4),
then convert them to the flat `.bin` format the C++ runtime reads:

```bash
python tools/model_export/gpt-oss-20b/export_model_bin.py     # or gpt-oss-120b/
```

There is also `tools/model_export/gpt-oss-7m/` — a tiny randomly-initialised model that lets you
exercise the whole pipeline without downloading anything.

### 3. Build

```bash
./run.sh build          # optimized, this is the one you want
./run.sh build omp      # optimized + OpenMP + -march=native
./run.sh build debug    # -O0 with symbols
```

`run.sh` is a thin wrapper over the `Makefile`; `make runfast`, `make runomp` and friends still work.
Building also regenerates `tokenizer.bin` from the `o200k_harmony` encoding.

### 4. Run

```bash
# interactive chat
./run.sh run model.bin -m chat -y "You are a concise assistant."

# one prompt, one completion
./run.sh run model.bin -m generate -i "Write a haiku about parallelism." -t 0.8 -p 0.95

# batch: read prompts from a file, write completions to another
./run.sh run model.bin -m getp -i tests/data/input.txt -o tests/data/output.txt
```

`getp` is the batch serving mode and the one the throughput numbers come from. All GPUs visible to the
process are used automatically, up to 8; to use fewer, set `HIP_VISIBLE_DEVICES=0,1`.

| Flag | Meaning | Default |
| --- | --- | --- |
| `-m` | mode: `generate`, `chat` or `getp` | `generate` |
| `-i` | prompt, or input file in `getp` mode | — |
| `-o` | output file in `getp` mode | — |
| `-n` | number of steps; `0` means the full context | `1024` |
| `-t` | temperature | `0.0` |
| `-p` | top-p | `0.9` |
| `-s` | random seed | `time(NULL)` |
| `-y` | system prompt, `chat` mode only | — |
| `-z` | custom tokenizer path | `tokenizer.bin` |

```bash
./run.sh --help         # colorized summary of all of the above
```

### 5. Score it

```bash
./run.sh eval 20b       # or: cd tests && python eval.py -m 20b
```

This decodes the token ids in `tests/submission/` against `tests/references/` and reports METEOR and
BERTScore F1, failing if either falls under the threshold in `tests/threshold.json`. A GPU is strongly
recommended — see [`tests/README.md`](tests/README.md) for the details.

---

## Results

Measured on one node of 8× AMD MI250, batch (`getp`) mode.

| Model | Requests | Warm-up (s) | Throughput (TPS) | METEOR | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gpt-oss-20b` | 12288 | 180 | **33891** | 0.53 | 0.97 |
| `gpt-oss-120b` | 6144 | 390 | **13010** | 0.56 | 0.98 |

METEOR has to clear 0.3 and BERTScore 0.9; both models pass with room to spare, so the throughput was
not bought with degraded output.

---

## How It Works

| | What it does | Detail |
| --- | --- | --- |
| **Model** | How gpt-oss is put together: a transformer with a mixture-of-experts block, 20B and 120B differing only in depth and expert count. | [MODEL.md](docs/MODEL.md) |
| **Kernels** | A blocktiled matrix-multiply on the matrix cores with double buffering, a FlashAttention-style attention kernel that exploits grouped-query attention, and one fused kernel that serves every expert at once. | [KERNELS.md](docs/KERNELS.md) |
| **Parallelism** | Expert × Data parallelism across the GPUs, with all-gather and reduce-scatter built directly on peer-to-peer copies and ordered to keep PCIe running full-duplex. | [PARALLELISM.md](docs/PARALLELISM.md) |

### Code structure

```plain
gpt-oss/
├── include/            # shared headers
├── src/
│   ├── run.cpp         # entry point, CLI and sampling
│   ├── tokenizer.cpp   # o200k_harmony tokenizer
│   ├── decode.cpp      # token ids -> text utility
│   ├── getp/           # batch serving: scheduling, runtime, collectives
│   └── hip/            # HIP kernels (attention, GEMM, MoE, norms)
├── tests/              # METEOR / BERTScore evaluation, prompts, references
├── tools/              # weight and tokenizer conversion, HF reference runner
├── docs/               # engineering write-up
└── run.sh              # build / run / eval wrapper
```

---

## Blog

A write-up of how this was built — what was tried, what was discarded and where the throughput
actually came from — is on its way. Until it lands, [`docs/`](docs/) carries the same material in
shorter form.

---

## Acknowledgments

This project was part of the GPU Engineer Training Program, a collaboration between
[Moreh](https://www.linkedin.com/company/moreh-vietnam/) and
[THUNDER Research Group](http://snuvm.snu.ac.kr/) (Seoul National University). It started from
[llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy.
