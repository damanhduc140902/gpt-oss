#pragma once

#include <cstddef>
#include <mutex>
#include <condition_variable>

// Assume: there is no critical point required to be executed by the threads

struct Barrier {
  size_t n_threads; // number of threads;
  size_t count;

  std::mutex mtx;
  std::condition_variable cv;

  Barrier(int n_workers) : n_threads(n_workers) {
    count = 0;
  }
  ~Barrier() {}

  void wait() {
    std::unique_lock<std::mutex> lck(mtx);
    ++count;
    if (count < n_threads) {
      cv.wait(lck);
    }
    else {
      count = 0;
      cv.notify_all();
    }
  }
};