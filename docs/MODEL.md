# The gpt-oss model

Both `gpt-oss-20b` and `gpt-oss-120b` can be visualized as shown in the figure below. For simplicity,
the residual connections are omitted. Overall, gpt-oss is a transformer model combined with the
mixture-of-experts (MoE) mechanism. The only difference between the two is that `gpt-oss-120b` has more
layers and a greater number of experts per layer than `gpt-oss-20b`.

![gpt-oss overview](assets/model-overview.png)

That similarity is what makes one implementation enough for both. The layer count and expert count are
read from the checkpoint, so the same binary serves either model; what changes between them is how the
experts are spread across GPUs, which is covered in [PARALLELISM.md](PARALLELISM.md).

## Where the time goes

Two layers dominate the computational load, and therefore the optimization effort:

- **Matrix multiplication**, including the MLPs inside the mixture of experts.
- **Multi-head attention**.

Both are written with the assumption that the batch size can be very large — 512 or 1024 prompt tokens
at a time — because the goal is throughput rather than single-request latency. That assumption drives
the tiling and scheduling choices described in [KERNELS.md](KERNELS.md).

## Tokenizer

The model uses OpenAI's `o200k_harmony` encoding, 201,088 tokens with a maximum token length of 128.
It is reimplemented in C++ in [`src/tokenizer.cpp`](../src/tokenizer.cpp); `tokenizer.bin` is generated
from `tiktoken` by [`tools/export_tokenizer_bin.py`](../tools/export_tokenizer_bin.py) as part of the
build. [`tests/test_tokenizer.py`](../tests/test_tokenizer.py) checks the C++ implementation against
`tiktoken` directly.
