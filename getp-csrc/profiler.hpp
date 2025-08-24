#pragma once

// Forward declarations for profiling functions
void reset_timing_summary();
void print_timing_summary();

// ScopedTimer class declaration (implementation in profiler.cpp)
class ScopedTimer {
public:
  const char* label;
  double t0;
  ScopedTimer(const char* l);
  ~ScopedTimer();
};

// Profiling macro - place at the beginning of functions to be profiled
#define PROFILE_FUNCTION() ScopedTimer _scoped_timer_##__LINE__(__func__)