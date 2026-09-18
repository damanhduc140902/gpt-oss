# Engineering documentation

How the inference system is built. The [main README](../README.md) covers what it does and how to run
it; these pages cover why it is shaped the way it is.

| Page                             | Contents                                                                                 |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| [MODEL.md](MODEL.md)             | The gpt-oss architecture, and what the 20B and 120B variants share.                      |
| [KERNELS.md](KERNELS.md)         | Matrix multiply, multi-head attention and the mixture-of-experts kernel.                 |
| [PARALLELISM.md](PARALLELISM.md) | Expert × Data parallelism, and the collectives built on peer-to-peer copies.             |
| [SERVING.md](SERVING.md)         | The `getp` batch runtime: request partitioning, KV cache sizing, warm-up and throughput. |

## Ground rules

The whole system is written against two constraints, which explain most of the design decisions in
these pages:

- Only the standard C/C++ library may be used. Third-party libraries are not allowed.
- `hip` and `omp` are the two exceptions.

So there is no rocBLAS, no hipBLAS, no RCCL and no MPI here. Every GEMM, every attention kernel and
every collective is in this repository.

Targets, in order of priority:

1. **Throughput** — the highest achievable number of output tokens per second.
2. **Preserved meaning** — output tokens must carry the same meaning as those from a plain CPU-only
   implementation, scored by METEOR ≥ 0.3 and BERTScore ≥ 0.9.
