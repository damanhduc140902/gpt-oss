# Core kernels

Matrix multiplication (including the MLPs in the mixture of experts) and multi-head attention are the
two most important layers of the model, because of their heavy computational load. Given the
high-throughput objective, these core kernels are implemented with the assumption that the batch size
supported by the inference system can be very large — for example 512 or 1024 prompt tokens at a time.

All of this lives in [`src/hip/forward.hip`](../src/hip/forward.hip).

## Matrix multiply kernel

The kernel performs the $\text{C}=\text{A}\times\text{B}^{\top}$ matrix multiplication, where the sizes
of $\text{C}$, $\text{A}$ and $\text{B}$ are respectively $\text{M}\times\text{N}$,
$\text{M}\times\text{K}$ and $\text{N}\times\text{K}$. The value of $\text{M}$ mostly equals the batch
size the inference system operates at, while $\text{N}$ and $\text{K}$ are both fixed by the model's
architecture and vary between kinds of layer. For instance, in the matrix-multiply layer inside the
mixture-of-experts block, the values of $\text{N}$ and $\text{K}$ differ significantly from each other,
although $\text{M}$ remains the same.

The most important technique used is **blocktiling** with appropriately chosen tiling parameters. The
goal is to increase the arithmetic intensity of the kernel and so raise its theoretical peak
throughput. The tiling parameters depend on the values of $\text{M}$, $\text{N}$ and $\text{K}$. When
the matrices are all very large — more than 1000, say — the tile should also be large, such as
128×128. When $\text{M}$ and $\text{N}$ are small, 16 or 32, the tiling parameters have to change to
other appropriate values.

The second technique is **double buffering**, which hides the latency of loading data from high-bandwidth
memory (HBM) into the local data share (LDS) within a compute unit. Beyond that, the specialized
**matrix core** accelerators are exploited through the built-in intrinsics of the HIP programming model;
all computation instructions in the kernel are issued by the matrix cores.

## Multi-head attention kernel

The attention kernel is based on the **FlashAttention** algorithm, which computes the softmax on the fly.
The remaining challenge is computing the attention scores effectively. Mathematically they are the dot
product between each query head and the corresponding key heads, which can be performed much like a
matrix multiplication — so every technique from the matrix-multiply kernel applies here too.

By exploiting the **grouped-query attention** (GQA) mechanism in gpt-oss, one thread block is assigned to
one key/value head instead of one query head, which makes more data available for reuse. The figure
below visualizes how the attention score computation is laid out. A key/value cache avoids recomputing
the keys and values of previously generated tokens.

![attention score computation](assets/attention.png)

## Mixture-of-experts

The MLP layers execute in the same manner as the matrix multiply kernel. To do that, the input for the
experts is "sorted" so that there is one matrix multiplication per expert.

The problem is that the distribution of inputs across experts is non-uniform. Often some experts receive
a large number of inputs to process while many others receive few or none. If the matrix-multiply
kernels for the MoE MLPs are launched naively, the hardware sits partly idle. To address this, **one
specialized kernel is launched once for all experts** rather than one launch per expert.
