#pragma once
#include <hip/hip_runtime.h>
#include <vector>
#include <cstddef>

#ifndef HIP_CHECK
#define HIP_CHECK(expr)                                                          \
  do {                                                                           \
    hipError_t _st = (expr);                                                     \
    if (_st != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)_st,                  \
              hipGetErrorString(_st), __FILE__, __LINE__);                       \
      abort();                                                                   \
    }                                                                            \
  } while (0)
#endif

struct CollectiveGroup {
  std::vector<int> ranks;          // device IDs participating in the group
  std::vector<hipStream_t> comm;   // one comm stream per device (created in cgCreate)
};

void cgCreate(CollectiveGroup& g, const std::vector<int>& devices);
void cgDestroy(CollectiveGroup& g);
void cgBroadcastF32(const CollectiveGroup& g, float** bufs, size_t count,
                    int root_rank = 0, bool sync = true);

// Broadcast from root to all peers (P2P when possible)
template <typename T>
void cgBroadcast(const CollectiveGroup& g, T** bufs, size_t count,
                 int root_rank, bool sync = true) {
  const size_t bytes = count * sizeof(T);
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

void cgAllReduceSumF32(const CollectiveGroup& g, float** bufs, size_t count,
                       int root_rank = 0, bool sync = true);
void cgReduceSumF32(const CollectiveGroup& g, float** bufs, size_t count,
                    int root_rank = 0, bool sync = true);

void cgAllReduceArgmaxF32I32(const CollectiveGroup& g,
                             float** vals, int** idxs,
                             int root_rank = 0, bool sync = true);

struct P2PSliceF32 {
  int src_rank;
  int dst_rank;
  const float* src; // device pointer on g.ranks[src_rank]
  float* dst;       // device pointer on g.ranks[dst_rank]
  size_t count;     // number of float elements
};
void cgAllToAllvSlicesF32(const CollectiveGroup& g,
                          const P2PSliceF32* slices, int num_slices,
                          bool sync = true);
