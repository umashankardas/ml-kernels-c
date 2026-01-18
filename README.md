# ml-kernels-c 🧱

A collection of low-level AI/ML computational kernels written in C, focusing on hardware-aware design and performance optimization.

## 🎯 Goal
To build highly optimized primitives (MatMul, Conv2D, Activations, Quantization) from scratch, documenting the performance delta between naive implementations and hardware-accelerated versions.

## 📊 Performance Roadmap
### Phase 1: Matrix Multiplication
The current baseline is a naive 3-loop $O(N^3)$ implementation. 

| Version | Matrix Size (N) | Time (s) | GFLOPS |
| :--- | :--- | :--- | :--- |
| Naive | 32 | 0.000027 | 2.46 |
| Naive | 512 | 0.404303 | 0.66 |
| Naive | 1024 | 8.502944 | 0.25 |

**Observation:** We see a **10x performance degradation** in GFLOPS as the matrix size increases. This is a classic "Cache Cliff" where the B-matrix strides exceed the L1/L2 cache capacity, forcing the CPU to wait for high-latency DRAM.



## 🛠️ Project Structure
- `src/matmul/`: Matrix multiplication implementations.
- `src/utils/`: High-resolution timers and memory helpers.
- `benchmarks/`: Harnesses for performance measurement.
- `results/`: CSV data and performance logs.

## 🚀 Next Steps
- [ ] Implement **Blocked (Tiled) MatMul** to exploit L1/L2 cache locality.
- [ ] Implement **SIMD (AVX2/FMA)** kernels.
- [ ] Comparison between Row-Major and Column-Major layouts.

## 🚀 Performance Comparison: Naive vs. Blocked

The following data was collected using `GCC -O3 -march=native` on a $2048 \times 2048$ Matrix Multiplication.

| Implementation | Execution Time | GFLOPS | Speedup |
| :--- | :--- | :--- | :--- |
| **Naive** | 125.16s | 0.14 | 1.0x |
| **Blocked (32x32)** | 9.04s | 1.90 | **13.8x** |

### 💡 Key Insight
The **Naive** version suffers from a "Memory Wall." As the matrix size exceeds the CPU cache (L1/L2/L3), the CPU spends most of its time waiting for data to arrive from the DRAM. 

The **Blocked** version keeps small $32 \times 32$ "tiles" in the L1 cache, allowing each piece of data to be reused 32 times before being swapped out. This transforms a memory-bound problem into a compute-bound one.

## 📊 Performance Results

The following benchmarks were conducted on a Windows machine using `gcc -O3 -march=native`.

| Implementation | Matrix Size ($N$) | Execution Time (s) | Throughput (GFLOPS) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Naive (Baseline)** | 2048 | 125.16 | 0.14 | 1.0x |
| **Blocked (32x32)** | 2048 | 9.04 | 1.90 | **13.8x** |
| **SIMD (AVX2)** | 2048 | *Coming Soon* | *Coming Soon* | -- |

## 🏆 Final Performance Roadmap

| Implementation | $N=2048$ Time | $N=2048$ GFLOPS | Peak GFLOPS | Optimization Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Naive** | 98.29s | 0.17 | 2.50 | None (Scalar) |
| **Blocked** | 5.34s | 3.22 | 3.22 | Cache Locality |
| **SIMD** | 1.51s | 11.38 | 12.18 | Vectorization (AVX2) |
| **OpenMP** | 1.51s | 11.38 | **34.46** | Multi-core Parallelism |

### 🚀 Key Takeaways
- **SIMD** provided the most consistent boost across all sizes.
- **OpenMP** is a beast for medium-sized matrices but is eventually limited by system memory bandwidth at very large sizes.
- **Optimization is a journey:** We improved performance from 0.17 GFLOPS to a peak of 34.46 GFLOPS—a **202x increase** in raw throughput.