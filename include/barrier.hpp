#pragma once
#include <atomic>
#include <thread>

struct Barrier {
  const int n_threads;
  std::atomic<int> count;
  std::atomic<int> phase;

  Barrier(int n) : n_threads(n), count(0), phase(0) {}
  ~Barrier() {}

  void wait() {
    int p = phase.load(std::memory_order_relaxed);
    if (count.fetch_add(1, std::memory_order_acq_rel) == n_threads - 1) {
      count.store(0, std::memory_order_release);
      phase.fetch_add(1, std::memory_order_release);
    } else {
      while (phase.load(std::memory_order_acquire) == p) {
        std::this_thread::yield();
      }
    }
  }
};
