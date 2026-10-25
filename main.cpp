#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <omp.h>
#include <xmmintrin.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include "aligned_allocator.h"

using Aligned32FloatVector = std::vector<float, AlignedAllocator<float, 32>>;
using Aligned64FloatVector = std::vector<float, AlignedAllocator<float, 64>>;

namespace bm = benchmark;

// Each benchmark runs at two array sizes:
// - 2048 floats (8 KiB per array): x and y both fit in the 32 KiB L1 data cache.
// - 100 million floats (400 MB per array): far larger than any cache, so data streams from DRAM.
constexpr long int kL1Size = 2048;
constexpr long int kDramSize = 100'000'000;

#define SAXPY_BENCHMARK(fn) \
  BENCHMARK(fn)->Arg(kL1Size)->Arg(kDramSize)->Repetitions(5)->DisplayAggregatesOnly(true)

// SAXPY does 2 flops (multiply + add) and moves 12 bytes (load x, load y, store y) per element.
void set_counters(bm::State& state, size_t n) {
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * 3 * sizeof(float)));
  state.counters["FLOPS"] =
      bm::Counter(2.0 * static_cast<double>(n) * state.iterations(), bm::Counter::kIsRate);
}

template <typename T = std::vector<float>>
T generate_random_values(int low, int high, unsigned long n) {
  if (low > high) throw std::runtime_error("Error: low > high");

  std::random_device rd;
  std::mt19937 e1(rd());

  std::uniform_real_distribution<float> dist(low, high);

  T random_values;
  random_values.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    random_values.emplace_back(dist(e1));
  }
  return random_values;
}

void print_vector(std::vector<float> vec) {
  for (float val : vec) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

template <typename T>
void saxpy_aligned(const float& alpha, const T& x, T& y) {
  if (x.size() != y.size()) throw std::runtime_error("Error: x.size() != y.size()");

  for (size_t i = 0; i < x.size(); ++i) {
    y[i] += alpha * x[i];
  }
}

template <typename T>
void saxpy_aligned_restrict(const float& alpha, const T* __restrict x, const size_t size_x,
                            T* __restrict y, const size_t size_y) {
  if (size_x != size_y) throw std::runtime_error("Error: x.size() != y.size()");

  for (size_t i = 0; i < size_x; ++i) {
    y[i] += alpha * x[i];
  }
}

static void BM_aligned32(bm::State& state) {
  const size_t n = state.range(0);
  const auto x = generate_random_values<Aligned32FloatVector>(0, 10, n);
  auto y = generate_random_values<Aligned32FloatVector>(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_aligned<Aligned32FloatVector>(alpha, x, y);
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

static void BM_aligned64(bm::State& state) {
  const size_t n = state.range(0);
  const auto x = generate_random_values<Aligned64FloatVector>(0, 10, n);
  auto y = generate_random_values<Aligned64FloatVector>(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_aligned<Aligned64FloatVector>(alpha, x, y);
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

static void BM_aligned64_restrict(bm::State& state) {
  const size_t n = state.range(0);
  const auto x = generate_random_values<Aligned64FloatVector>(0, 10, n);
  auto y = generate_random_values<Aligned64FloatVector>(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_aligned_restrict<float>(alpha, x.data(), x.size(), y.data(), y.size());
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

void saxpy_multiple_8(const float& alpha, const std::vector<float>& x, std::vector<float>& y) {
  if (x.size() != y.size()) throw std::runtime_error("Error: x.size() != y.size()");

  size_t i_end = x.size() - (x.size() % 8);
  for (size_t i = 0; i < i_end; ++i) {
    y[i] += alpha * x[i];
  }

  for (size_t i = i_end; i < x.size(); ++i) {
    y[i] += alpha * x[i];
  }
}

void saxpy_naive(const float& alpha, const std::vector<float>& x, std::vector<float>& y) {
  if (x.size() != y.size()) throw std::runtime_error("Error: x.size() != y.size()");

  for (size_t i = 0; i < x.size(); ++i) {
    y[i] += alpha * x[i];
  }
}

static void BM_naive(bm::State& state) {
  const size_t n = state.range(0);
  const auto x = generate_random_values(0, 10, n);
  auto y = generate_random_values(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_naive(alpha, x, y);
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

void saxpy_restrict(const float& alpha, const float* __restrict x, const size_t size_x,
                    float* __restrict y, const size_t size_y) {
  if (size_x != size_y) throw std::runtime_error("Error: x.size() != y.size()");

  for (size_t i = 0; i < size_x; ++i) {
    y[i] += alpha * x[i];
  }
}

static void BM_restrict(bm::State& state) {
  const size_t n = state.range(0);
  const auto x = generate_random_values(0, 10, n);
  auto y = generate_random_values(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_restrict(alpha, x.data(), x.size(), y.data(), y.size());
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

static void BM_multiple_8(bm::State& state) {
  const size_t n = state.range(0);
  const auto x = generate_random_values(0, 10, n);
  auto y = generate_random_values(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_multiple_8(alpha, x, y);
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

void saxpy_avx2(float alpha, const float* __restrict x, float* __restrict y, size_t n) {
  size_t i = 0;

  // Broadcast scalar alpha to all 8 floats in a 256-bit register
  __m256 alpha_vec = _mm256_set1_ps(alpha);

  // Process 8 floats at a time
  for (; i + 7 < n; i += 8) {
    __m256 x_vec = _mm256_loadu_ps(x + i);                     // load 8 floats from x
    __m256 y_vec = _mm256_loadu_ps(y + i);                     // load 8 floats from y
    __m256 result = _mm256_fmadd_ps(alpha_vec, x_vec, y_vec);  // fused multiply-add
    _mm256_storeu_ps(y + i, result);                           // store result back to y
  }

  // Tail loop for leftover elements
  for (; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

static void BM_avx2(bm::State& state) {
  const size_t n = state.range(0);
  const auto x = generate_random_values(0, 10, n);
  auto y = generate_random_values(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_avx2(alpha, x.data(), y.data(), x.size());
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

void saxpy_avx2_prefetch(float alpha, const float* __restrict x, float* __restrict y, size_t n,
                         const size_t prefetch_distance = 64) {
  size_t i = 0;

  // Broadcast scalar alpha to all 8 floats in a 256-bit register
  __m256 alpha_vec = _mm256_set1_ps(alpha);

  // Process 8 floats at a time
  for (; i + 7 < n; i += 8) {
    // Prefetch data ahead of time
    if (i + prefetch_distance < n) {
      _mm_prefetch((const char*)(x + i + prefetch_distance), _MM_HINT_T0);
      _mm_prefetch((const char*)(y + i + prefetch_distance), _MM_HINT_T0);
    }

    __m256 x_vec = _mm256_loadu_ps(x + i);                     // load 8 floats from x
    __m256 y_vec = _mm256_loadu_ps(y + i);                     // load 8 floats from y
    __m256 result = _mm256_fmadd_ps(alpha_vec, x_vec, y_vec);  // fused multiply-add
    _mm256_storeu_ps(y + i, result);                           // store result back to y
  }

  // Tail loop for leftover elements
  for (; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

static void BM_avx2_prefetch64(bm::State& state) {
  const size_t n = state.range(0);
  const auto x = generate_random_values(0, 10, n);
  auto y = generate_random_values(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_avx2_prefetch(alpha, x.data(), y.data(), x.size());
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

static void BM_avx2_prefetch128(bm::State& state) {
  const size_t n = state.range(0);
  const auto x = generate_random_values(0, 10, n);
  auto y = generate_random_values(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_avx2_prefetch(alpha, x.data(), y.data(), x.size(), 128);
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

static void BM_avx2_prefetch256(bm::State& state) {
  const size_t n = state.range(0);
  const auto x = generate_random_values(0, 10, n);
  auto y = generate_random_values(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_avx2_prefetch(alpha, x.data(), y.data(), x.size(), 256);
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

void saxpy_loop_unroll_2(const float& alpha, const std::vector<float>& x, std::vector<float>& y) {
  if (x.size() != y.size()) throw std::runtime_error("Error: x.size() != y.size()");

  for (size_t i = 0; i < 2 * (x.size() / 2); i += 2) {
    y[i] = alpha * x[i] + y[i];
    y[i + 1] = alpha * x[i + 1] + y[i + 1];
  }

  for (size_t j = x.size() - (x.size() % 2); j < x.size(); ++j) {
    y[j] = alpha * x[j] + y[j];
  }
}

static void BM_loop_unroll_2(bm::State& state) {
  const size_t n = state.range(0);
  const std::vector<float> x = generate_random_values(0, 10, n);
  std::vector<float> y = generate_random_values(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_loop_unroll_2(alpha, x, y);
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

void saxpy_loop_unroll_4(const float& alpha, const std::vector<float>& x, std::vector<float>& y) {
  if (x.size() != y.size()) throw std::runtime_error("Error: x.size() != y.size()");

  for (size_t i = 0; i < 4 * (x.size() / 4); i += 4) {
    y[i] = alpha * x[i] + y[i];
    y[i + 1] = alpha * x[i + 1] + y[i + 1];
    y[i + 2] = alpha * x[i + 2] + y[i + 2];
    y[i + 3] = alpha * x[i + 3] + y[i + 3];
  }

  for (size_t j = x.size() - (x.size() % 4); j < x.size(); ++j) {
    y[j] = alpha * x[j] + y[j];
  }
}

static void BM_loop_unroll_4(bm::State& state) {
  const size_t n = state.range(0);
  const std::vector<float> x = generate_random_values(0, 10, n);
  std::vector<float> y = generate_random_values(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_loop_unroll_4(alpha, x, y);
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

void saxpy_loop_unroll_8(const float& alpha, const std::vector<float>& x, std::vector<float>& y) {
  if (x.size() != y.size()) throw std::runtime_error("Error: x.size() != y.size()");

  for (size_t i = 0; i < 8 * (x.size() / 8); i += 8) {
    y[i] = alpha * x[i] + y[i];
    y[i + 1] = alpha * x[i + 1] + y[i + 1];
    y[i + 2] = alpha * x[i + 2] + y[i + 2];
    y[i + 3] = alpha * x[i + 3] + y[i + 3];
    y[i + 4] = alpha * x[i + 4] + y[i + 4];
    y[i + 5] = alpha * x[i + 5] + y[i + 5];
    y[i + 6] = alpha * x[i + 6] + y[i + 6];
    y[i + 7] = alpha * x[i + 7] + y[i + 7];
  }

  for (size_t j = x.size() - (x.size() % 8); j < x.size(); ++j) {
    y[j] = alpha * x[j] + y[j];
  }
}

static void BM_loop_unroll_8(bm::State& state) {
  const size_t n = state.range(0);
  const std::vector<float> x = generate_random_values(0, 10, n);
  std::vector<float> y = generate_random_values(0, 10, n);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_loop_unroll_8(alpha, x, y);
    bm::DoNotOptimize(y);
  }
  set_counters(state, n);
}

SAXPY_BENCHMARK(BM_naive);
SAXPY_BENCHMARK(BM_restrict);
SAXPY_BENCHMARK(BM_avx2);
SAXPY_BENCHMARK(BM_avx2_prefetch64);
SAXPY_BENCHMARK(BM_avx2_prefetch128);
SAXPY_BENCHMARK(BM_avx2_prefetch256);
SAXPY_BENCHMARK(BM_multiple_8);
SAXPY_BENCHMARK(BM_aligned32);
SAXPY_BENCHMARK(BM_aligned64);
SAXPY_BENCHMARK(BM_aligned64_restrict);
SAXPY_BENCHMARK(BM_loop_unroll_2);
SAXPY_BENCHMARK(BM_loop_unroll_4);
SAXPY_BENCHMARK(BM_loop_unroll_8);

BENCHMARK_MAIN();
