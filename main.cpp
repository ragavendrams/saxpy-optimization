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

constexpr long int N = 1e8;

template <typename T = std::vector<float>>
T generate_random_values(int low, int high, unsigned long n) {
  if (low > high) throw std::runtime_error("Error: low > high");

  std::random_device rd;
  std::mt19937 e1(rd());

  std::uniform_real_distribution<float> dist(low, high);

  T random_values(n);

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
  const auto x = generate_random_values<Aligned32FloatVector>(0, 10, N);
  auto y = generate_random_values<Aligned32FloatVector>(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_aligned<Aligned32FloatVector>(alpha, x, y);
    bm::DoNotOptimize(y);
  }
}

static void BM_aligned64(bm::State& state) {
  const auto x = generate_random_values<Aligned64FloatVector>(0, 10, N);
  auto y = generate_random_values<Aligned64FloatVector>(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_aligned<Aligned64FloatVector>(alpha, x, y);
    bm::DoNotOptimize(y);
  }
}

static void BM_aligned64_restrict(bm::State& state) {
  const auto x = generate_random_values<Aligned64FloatVector>(0, 10, N);
  auto y = generate_random_values<Aligned64FloatVector>(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_aligned_restrict<float>(alpha, x.data(), x.size(), y.data(), y.size());
    bm::DoNotOptimize(y);
  }
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
  const auto x = generate_random_values(0, 10, N);
  auto y = generate_random_values(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_naive(alpha, x, y);
    bm::DoNotOptimize(y);
  }
}

void saxpy_restrict(const float& alpha, const float* __restrict x, const size_t size_x,
                    float* __restrict y, const size_t size_y) {
  if (size_x != size_y) throw std::runtime_error("Error: x.size() != y.size()");

  for (size_t i = 0; i < size_x; ++i) {
    y[i] += alpha * x[i];
  }
}

static void BM_restrict(bm::State& state) {
  const auto x = generate_random_values(0, 10, N);
  auto y = generate_random_values(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_restrict(alpha, x.data(), x.size(), y.data(), y.size());
    bm::DoNotOptimize(y);
  }
}

static void BM_multiple_8(bm::State& state) {
  const auto x = generate_random_values(0, 10, N);
  auto y = generate_random_values(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_multiple_8(alpha, x, y);
    bm::DoNotOptimize(y);
  }
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
  const auto x = generate_random_values(0, 10, N);
  auto y = generate_random_values(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_avx2(alpha, x.data(), y.data(), N);
    bm::DoNotOptimize(y);
  }
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
  const auto x = generate_random_values(0, 10, N);
  auto y = generate_random_values(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_avx2_prefetch(alpha, x.data(), y.data(), N);
    bm::DoNotOptimize(y);
  }
}

static void BM_avx2_prefetch128(bm::State& state) {
  const auto x = generate_random_values(0, 10, N);
  auto y = generate_random_values(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_avx2_prefetch(alpha, x.data(), y.data(), N, 128);
    bm::DoNotOptimize(y);
  }
}

static void BM_avx2_prefetch256(bm::State& state) {
  const auto x = generate_random_values(0, 10, N);
  auto y = generate_random_values(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_avx2_prefetch(alpha, x.data(), y.data(), N, 256);
    bm::DoNotOptimize(y);
  }
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
  const std::vector<float> x = generate_random_values(0, 10, N);
  std::vector<float> y = generate_random_values(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_loop_unroll_2(alpha, x, y);
    bm::DoNotOptimize(y);
  }
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
  const std::vector<float> x = generate_random_values(0, 10, N);
  std::vector<float> y = generate_random_values(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_loop_unroll_4(alpha, x, y);
    bm::DoNotOptimize(y);
  }
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
  const std::vector<float> x = generate_random_values(0, 10, N);
  std::vector<float> y = generate_random_values(0, 10, N);
  // Make alpha random to prevent compiler from optimizing for a particular
  // alpha
  const float alpha = generate_random_values(0, 10, 1)[0];

  for (auto _ : state) {
    saxpy_loop_unroll_8(alpha, x, y);
    bm::DoNotOptimize(y);
  }
}

BENCHMARK(BM_naive)->Repetitions(4);
BENCHMARK(BM_restrict)->Repetitions(4);
BENCHMARK(BM_avx2)->Repetitions(4);
BENCHMARK(BM_avx2_prefetch64)->Repetitions(4);
BENCHMARK(BM_avx2_prefetch128)->Repetitions(4);
BENCHMARK(BM_avx2_prefetch256)->Repetitions(4);
BENCHMARK(BM_multiple_8)->Repetitions(4);
BENCHMARK(BM_aligned32)->Repetitions(4);
BENCHMARK(BM_aligned64)->Repetitions(4);
BENCHMARK(BM_aligned64_restrict)->Repetitions(4);
BENCHMARK(BM_loop_unroll_2)->Repetitions(2);
BENCHMARK(BM_loop_unroll_4)->Repetitions(2);
BENCHMARK(BM_loop_unroll_8)->Repetitions(2);

BENCHMARK_MAIN();
