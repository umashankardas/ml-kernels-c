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