#include "collectives.hpp"
#include <algorithm>
#include <cstdio>

// Simple in-place add: dst[i] += src[i]
static __global__ void add_inplace_f32(float* dst, const float* src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

// Try to enable P2P for all pairs in the group
static void enable_p2p_allpairs(const std::vector<int>& devs) {
  for (size_t i = 0; i < devs.size(); ++i) {
    HIP_CHECK(hipSetDevice(devs[i]));
    for (size_t j = 0; j < devs.size(); ++j) {
      if (i == j) continue;
      int can = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can, devs[i], devs[j]));
      if (can) {
        hipError_t st = hipDeviceEnablePeerAccess(devs[j], 0);
        if (st != hipSuccess && st != hipErrorPeerAccessAlreadyEnabled) {
          fprintf(stderr, "Warning: EnablePeerAccess %d->%d failed: %s\n",
                  devs[i], devs[j], hipGetErrorString(st));
        }
      }
    }
  }
}

// Create group: record device ids, create a comm stream per device, enable P2P
void cgCreate(CollectiveGroup& g, const std::vector<int>& devices) {
  g.ranks = devices;
  g.comm.resize(g.ranks.size());
  for (size_t i = 0; i < g.ranks.size(); ++i) {
    HIP_CHECK(hipSetDevice(g.ranks[i]));
    HIP_CHECK(hipStreamCreate(&g.comm[i]));
  }
  enable_p2p_allpairs(g.ranks);
}

// Destroy group
void cgDestroy(CollectiveGroup& g) {
  for (size_t i = 0; i < g.ranks.size(); ++i) {
    HIP_CHECK(hipSetDevice(g.ranks[i]));
    if (g.comm[i]) HIP_CHECK(hipStreamDestroy(g.comm[i]));
  }
  g.ranks.clear();
  g.comm.clear();
}

// Broadcast from root to all peers (P2P when possible)
void cgBroadcastF32(const CollectiveGroup& g, float** bufs, size_t count,
                    int root_rank, bool sync) {
  const size_t bytes = count * sizeof(float);
  if (g.ranks.size() <= 1) return; // no-op on single GPU

  // Copy root -> others
  for (size_t r = 0; r < g.ranks.size(); ++r) {
    if ((int)r == root_rank) continue;
    HIP_CHECK(hipMemcpyPeerAsync(
        /*dst=*/bufs[r], g.ranks[r],
        /*src=*/bufs[root_rank], g.ranks[root_rank],
        bytes, g.comm[r]));
  }

  if (sync) {
    for (size_t r = 0; r < g.ranks.size(); ++r) {
      HIP_CHECK(hipSetDevice(g.ranks[r]));
      HIP_CHECK(hipStreamSynchronize(g.comm[r]));
    }
  }
}

// Reduce-to-root (sum) then broadcast result to all
void cgAllReduceSumF32(const CollectiveGroup& g, float** bufs, size_t count,
                       int root_rank, bool sync) {
  const size_t bytes = count * sizeof(float);
  if (g.ranks.size() <= 1) return; // no-op

  const int root_dev = g.ranks[root_rank];

  // Root scratch buffer
  HIP_CHECK(hipSetDevice(root_dev));
  float* tmp = nullptr;
  HIP_CHECK(hipMalloc(&tmp, bytes));

  // Accumulate other ranks into root
  for (size_t r = 0; r < g.ranks.size(); ++r) {
    if ((int)r == root_rank) continue;

    // Copy peer buffer to root scratch
    HIP_CHECK(hipMemcpyPeerAsync(
        tmp, root_dev,
        bufs[r], g.ranks[r],
        bytes, g.comm[root_rank]));

    // Launch in-place add on root
    const int BLK = 256;
    const int GRD = (int)((count + BLK - 1) / BLK);
    hipLaunchKernelGGL(add_inplace_f32, dim3(GRD), dim3(BLK), 0, g.comm[root_rank],
                       bufs[root_rank], tmp, count);
  }

  // Make sure reduction is finished before broadcasting
  HIP_CHECK(hipStreamSynchronize(g.comm[root_rank]));
  HIP_CHECK(hipFree(tmp));

  // Broadcast the reduced buffer from root to all
  cgBroadcastF32(g, bufs, count, root_rank, /*sync=*/sync);
}

// Argmax across ranks for a **single** (val, idx) (host staging; tiny payload)
void cgAllReduceArgmaxF32I32(const CollectiveGroup& g,
                             float** vals, int** idxs,
                             int root_rank, bool sync) {
  const size_t P = g.ranks.size();
  if (P <= 1) return; // no-op

  // Pinned host staging
  float* h_vals = nullptr;
  int*   h_idxs = nullptr;
  HIP_CHECK(hipHostMalloc((void**)&h_vals, P * sizeof(float)));
  HIP_CHECK(hipHostMalloc((void**)&h_idxs, P * sizeof(int)));

  // D2H each scalar
  for (size_t r = 0; r < P; ++r) {
    HIP_CHECK(hipMemcpyDtoH(&h_vals[r], vals[r], sizeof(float)));
    HIP_CHECK(hipMemcpyDtoH(&h_idxs[r], idxs[r], sizeof(int)));
  }

  // Reduce on host
  int best_rank = 0;
  float best_val = h_vals[0];
  int best_idx = h_idxs[0];
  for (size_t r = 1; r < P; ++r) {
    if ((h_vals[r] > best_val) ||
        (h_vals[r] == best_val && h_idxs[r] < best_idx)) {
      best_val = h_vals[r];
      best_idx = h_idxs[r];
      best_rank = (int)r;
    }
  }

  // Broadcast result to all devices (copy the same scalar to each device)
  for (size_t r = 0; r < P; ++r) {
    HIP_CHECK(hipMemcpyHtoD(vals[r], &best_val, sizeof(float)));
    HIP_CHECK(hipMemcpyHtoD(idxs[r], &best_idx, sizeof(int)));
  }

  HIP_CHECK(hipHostFree(h_vals));
  HIP_CHECK(hipHostFree(h_idxs));

  if (sync) {
    for (size_t r = 0; r < P; ++r) {
      HIP_CHECK(hipSetDevice(g.ranks[r]));
      HIP_CHECK(hipStreamSynchronize(g.comm[r]));
    }
  }
}

// Generic list of P2P copies (can be used to implement alltoallv/dispatch)
void cgAllToAllvSlicesF32(const CollectiveGroup& g,
                          const P2PSliceF32* slices, int num_slices,
                          bool sync) {
  for (int i = 0; i < num_slices; ++i) {
    const int src_rank = slices[i].src_rank;
    const int dst_rank = slices[i].dst_rank;
    const size_t bytes = slices[i].count * sizeof(float);
    if (src_rank == dst_rank || bytes == 0) continue;
    HIP_CHECK(hipMemcpyPeerAsync(
        /*dst=*/slices[i].dst, g.ranks[dst_rank],
        /*src=*/slices[i].src, g.ranks[src_rank],
        bytes, g.comm[dst_rank]));
  }

  if (sync) {
    for (size_t r = 0; r < g.ranks.size(); ++r) {
      HIP_CHECK(hipSetDevice(g.ranks[r]));
      HIP_CHECK(hipStreamSynchronize(g.comm[r]));
    }
  }
}
