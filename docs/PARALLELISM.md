# Parallelism and communication

Code: [`src/getp/run.cpp`](../src/getp/run.cpp) for the runtime and
[`src/getp/collectives.cpp`](../src/getp/collectives.cpp) for the collectives.

## Expert × Data parallelism

To fully leverage a multi-GPU system, an appropriate parallelism model has to be built into the
inference system. Both `gpt-oss-20b` and `gpt-oss-120b` use a combination of expert and data
parallelism, referred to as **Expert × Data parallelism**, illustrated below.

Each GPU has its own transformer weights and key/value cache, independent of the others. Only the
experts are distributed evenly across devices. The transformer layers can therefore be computed in
parallel with no coordination, whereas the mixture-of-experts layers require synchronization between
devices.

![Expert x Data parallelism](assets/expert-data.png)

For `gpt-oss-120b`, the memory footprint of the weights is substantial, so the expert parallelism
parameter (EP) is set to 8 — the experts are spread across every GPU in the system.

`gpt-oss-20b` can run on a single AMD MI250 with 64 GB, yet EP is still set to 2. The reason is not
capacity but headroom: expert parallelism reduces the memory taken by the model weights, which leaves
more room for a larger batch size, which is what actually raises throughput.

The number of GPUs is detected at runtime with `hipGetDeviceCount`, with a compile-time ceiling of 8
(`MAXIMUM_GPU` in [`include/transformer.hpp`](../include/transformer.hpp)). To run on a subset, set
`HIP_VISIBLE_DEVICES`.

## Communication

Expert parallelism needs two collective operations: **all-gather** and **reduce-scatter**. Both are
implemented with peer-to-peer (P2P) communication, and the send and fetch operations are built solely
from `hipMemcpyPeerAsync`. No RCCL, no MPI.

The abstraction of the two collectives is illustrated below — all-gather on the left, reduce-scatter on
the right.

![collective communication](assets/collective-communication.png)

### Ordering matters

To exploit the full-duplex capability of PCIe, the order in which each device issues its sends and
fetches has to be considered carefully. For all-gather, the ordering below works. The key idea is that
under ideal conditions, each device is sending to and receiving from exactly one other device at any
moment.

```c
int i; // index of the device being managed
for (int delta = 1; delta < N_DEVICES; ++delta) {
  int j = (i + delta) % N_DEVICES;
  /*
    Send data to device having index j
  */
}
```

![all gather](assets/all-gather.png)

Incorrect ordering prevents full utilization of the PCIe full-duplex capability. The loop below is the
obvious way to write it, and it is the wrong one — every device targets device 0 first, then device 1,
and so on, so the transfers serialize instead of pairing up.

```c
int i; // index of the device being managed
for (int j = 0; j < N_DEVICES; ++j) {
  if (i != j) {
    /*
      Send data to device having index j
    */
  }
}
```

![incorrect all gather](assets/incorrect-all-gather.png)
