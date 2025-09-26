#include "profiler.hpp"
#include <chrono>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <string>

static inline double wall_now() {
  using clock = std::chrono::steady_clock; // monotonic
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct TimerAgg {
  std::unordered_map<std::string, double> total;
  std::unordered_map<std::string, long long> count;
};
static TimerAgg g_timer;

static inline void accumulate_time(const char* label, double elapsed) {
  #pragma omp critical(timeagg)
  {
    g_timer.total[std::string(label)] += elapsed;
    g_timer.count[std::string(label)] += 1;
  }
}

// ScopedTimer implementation
ScopedTimer::ScopedTimer(const char* l) : label(l), t0(wall_now()) {}
ScopedTimer::~ScopedTimer() { accumulate_time(label, wall_now() - t0); }

static double g_epoch_start_time = 0.0;

static inline double epoch_elapsed_sec() {
  if (g_epoch_start_time <= 0.0) return 0.0;
  return wall_now() - g_epoch_start_time;
}

void reset_timing_summary() {
  #pragma omp critical(timeagg)
  {
    g_timer.total.clear();
    g_timer.count.clear();
  }
  g_epoch_start_time = wall_now(); // bắt đầu epoch mới
}

static inline void print_repeat(char ch, int n) { while (n-- > 0) putchar(ch); }

void print_timing_summary() {
  struct Row { std::string label; double total; long long calls; double avg; double pct; };
  std::vector<Row> rows;
  rows.reserve(g_timer.total.size());

  double sum_total = 0.0;
  size_t max_label_len = 5; // "LABEL"
  for (const auto& kv : g_timer.total) {
    const std::string& label = kv.first;
    const double total = kv.second;
    const long long cnt = g_timer.count[label];
    const double avg = (cnt > 0) ? total / (double)cnt : 0.0; // seconds
    rows.push_back({label, total, cnt, avg, 0.0});
    sum_total += total;
    if (label.size() > max_label_len) max_label_len = label.size();
  }

  // % of wall time since last reset (not of sum of rows — avoids nesting double-count)
  const double epoch_sec = epoch_elapsed_sec();

  // Sort by total desc
  std::sort(rows.begin(), rows.end(),
            [](const Row& a, const Row& b) { return a.total > b.total; });

  // Fill percentages
  if (epoch_sec > 0.0) {
    for (auto& row : rows) row.pct = (row.total / epoch_sec) * 100.0;
  }

  // Columns
  int WL = (int)std::min<size_t>(std::max<size_t>(max_label_len, 24), 64);
  const int W_TOTAL = 12, W_CALLS = 9, W_AVG = 12, W_PCT = 7;

  printf("\n==== Timing summary ====\n");
  printf("Epoch wall time since last reset: %.6f s\n", epoch_sec);

  putchar('+'); print_repeat('-', WL+2); putchar('+');
  print_repeat('-', W_TOTAL+2); putchar('+');
  print_repeat('-', W_CALLS+2); putchar('+');
  print_repeat('-', W_AVG+2);   putchar('+');
  print_repeat('-', W_PCT+2);   putchar('+'); putchar('\n');

  // Updated header: AVG in ms
  printf("| %-*s | %*s | %*s | %*s | %*s |\n",
          WL, "LABEL",
          W_TOTAL, "TOTAL(s)",
          W_CALLS, "CALLS",
          W_AVG,   "AVG(ms)",
          W_PCT,   "%");

  putchar('+'); print_repeat('-', WL+2); putchar('+');
  print_repeat('-', W_TOTAL+2); putchar('+');
  print_repeat('-', W_CALLS+2); putchar('+');
  print_repeat('-', W_AVG+2);   putchar('+');
  print_repeat('-', W_PCT+2);   putchar('+'); putchar('\n');

  if (rows.empty()) {
    printf("| %-*s | %*s | %*s | %*s | %*s |\n",
            WL, "(no timings)", W_TOTAL, "-", W_CALLS, "-", W_AVG, "-", W_PCT, "-");
  } else {
    for (const auto& row : rows) {
      const double avg_ms = row.avg * 1000.0; // convert seconds -> milliseconds
      if (epoch_sec > 0.0) {
        printf("| %-*s | %*.6f | %*lld | %*.3f | %*.2f |\n",
                WL, row.label.c_str(),
                W_TOTAL, row.total,
                W_CALLS, row.calls,
                W_AVG,   avg_ms,
                W_PCT,   row.pct);
      } else {
        // No epoch baseline yet
        printf("| %-*s | %*.6f | %*lld | %*.3f | %*s |\n",
                WL, row.label.c_str(),
                W_TOTAL, row.total,
                W_CALLS, row.calls,
                W_AVG,   avg_ms,
                W_PCT,   "-");
      }
    }
  }

  putchar('+'); print_repeat('-', WL+2); putchar('+');
  print_repeat('-', W_TOTAL+2); putchar('+');
  print_repeat('-', W_CALLS+2); putchar('+');
  print_repeat('-', W_AVG+2);   putchar('+');
  print_repeat('-', W_PCT+2);   putchar('+'); putchar('\n');

  // FYI line: sum of row totals vs epoch wall time (can exceed 100% if you time nested scopes)
  double sum_pct = (epoch_sec > 0.0) ? (sum_total / epoch_sec) * 100.0 : 0.0;
  printf("Sum of row totals: %.6f s  |  Epoch wall time: %.6f s  |  sum%%/epoch = %.2f%%\n",
          sum_total, epoch_sec, sum_pct);
  fflush(stdout);
}
