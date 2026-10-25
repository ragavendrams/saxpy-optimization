# saxpy-optimization

Single-core SAXPY (`y = a*x + y` on float arrays) written several ways and timed with Google Benchmark.

## Build and run

```bash
cmake -S . -B build          # defaults to a Release (-O3) build
cmake --build build
./build/saxpy-optimization
```

Needs Google Benchmark (`libbenchmark-dev` on Ubuntu), OpenMP, and a CPU with AVX2 and FMA.

## Results

Setup: Intel Core i5-8265U (32 KiB L1 data cache per core, 6 MiB L3), Ubuntu on WSL2, GCC 15.2, `-O3 -march=native`, Google Benchmark 1.9.1. Each number is the median of 5 repetitions.

- **L1**: 2048 floats per array (8 KiB), so both arrays stay in the L1 cache. Shown as the range of medians over two runs, because laptop clock speeds vary between runs.
- **DRAM**: 100 million floats per array (400 MB), so every pass streams from main memory.

| Variant | What it tries | L1 (GFLOP/s) | DRAM (GFLOP/s) | DRAM (GiB/s) |
|---|---|---|---|---|
| `naive` | Plain loop over `std::vector` | 18.6–19.2 | 1.84 | 10.3 |
| `restrict` | Raw pointers marked `__restrict` | 18.1–23.4 | 1.88 | 10.5 |
| `multiple_8` | Main loop to a multiple of 8, then a tail loop | 17.7–25.6 | 2.24 | 12.5 |
| `aligned32` | 32-byte-aligned allocation | 21.0–26.6 | 2.14 | 11.9 |
| `aligned64` | 64-byte-aligned allocation | 25.7–27.2 | 2.22 | 12.4 |
| `aligned64_restrict` | 64-byte-aligned + `__restrict` | 26.0–27.6 | 2.17 | 12.1 |
| `loop_unroll_2` | Manual unroll by 2 | 18.9–26.0 | 2.10 | 11.7 |
| `loop_unroll_4` | Manual unroll by 4 | 18.0–26.4 | 2.26 | 12.6 |
| `loop_unroll_8` | Manual unroll by 8 | 3.2–3.3 | 1.65 | 9.2 |
| `avx2` | AVX2 + FMA intrinsics, 8 floats per instruction | 18.3–21.5 | 2.28 | 12.7 |
| `avx2_prefetch64` | AVX2 + software prefetch 64 floats ahead | 14.2–16.8 | 1.94 | 10.8 |
| `avx2_prefetch128` | AVX2 + software prefetch 128 floats ahead | 12.0–17.3 | 2.25 | 12.6 |
| `avx2_prefetch256` | AVX2 + software prefetch 256 floats ahead | 12.0–17.1 | 2.10 | 11.7 |

GFLOP/s counts 2 flops per element (multiply and add). GiB/s counts 12 bytes per element (load `x`, load `y`, store `y`).

## What the numbers show

- **From DRAM, every variant runs at about 2 GFLOP/s.** SAXPY moves 12 bytes for every 2 flops, so memory bandwidth (about 10–13 GiB/s here) sets the speed, not the instructions.
- **In L1, the compiler already vectorizes the plain loops at `-O3`,** so hand-written AVX2 is no faster. The spread between about 18 and 27 GFLOP/s is mostly run-to-run noise on this laptop.
- **Software prefetch only adds instructions:** slower in L1, no gain from DRAM, because the hardware prefetcher already handles a linear stream.
- **Unroll by 8 ran at scalar speed in L1.** The binary contains a vectorized version of that loop, but this run appears to use the scalar fallback. Not investigated further.

The Ubuntu `libbenchmark` package prints "Library was built as DEBUG". That refers to the benchmark library itself; the measured loops are built with `-O3`.
