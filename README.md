# ml-kernels-c 🧱

A collection of low-level AI/ML computational kernels written in C, focusing on hardware-aware design and performance optimization.

## 🎯 Goal
To build highly optimized primitives (MatMul, Conv2D, Activations) from scratch, documenting the performance delta between naive implementations and hardware-accelerated versions.

## 🛠️ Project Structure
- `src/matmul/`: MatMul, Linear Layers, and Fused OpenMP kernels.
- `src/activations/`: SIMD-accelerated ReLU and other functions.
- `src/utils/`: High-resolution timers and memory helpers.
- `bench/`: Harnesses for performance measurement.

## 🏆 Performance Milestone: The 380x Speedup
Benchmarked on a $1024 \times 1024$ workload using `GCC -O3 -march=native -fopenmp`.

| Implementation | Execution Time ($N=1024$) | GFLOPS | Speedup | Optimization Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Naive (Baseline)** | 6.130s | 0.35 | 1.0x | Scalar loops |
| **Blocked (32x32)** | 0.468s | 4.59 | 13.1x | Cache Locality |
| **SIMD (AVX2)** | 0.129s | 16.58 | 47.5x | 8-wide Vectorization |
| **OpenMP** | 0.025s | 85.37 | 244.0x | Multi-core Parallelism |
| **Linear-ReLU Fused** | **0.027s** | **78.66** | **224.7x** | **Bias + ReLU Fusion** |

### 📈 Peak Throughput: **85.37 GFLOPS**

---

## 💡 Engineering Insights

### 1. Cache Blocking & The Memory Wall
The Naive version collapses as $N$ grows because the CPU stalls waiting for RAM. By processing data in $32 \times 32$ tiles, we ensure the "hot" data stays in the L1/L2 cache, allowing for massive data reuse.



### 2. AVX2 Vectorization
Using `__m256` registers allows us to perform 8 floating-point additions/multiplications in a single instruction. This is the difference between a "moped" (Scalar) and a "supercar" (SIMD).

### 3. Deep Learning Operator Fusion
Our `linear_relu_fused` kernel performs $(A \times B) + Bias$ and then applies $ReLU(x)$ before the data ever leaves the CPU registers.
- **Why?** Writing to memory is 100x slower than register math. By "fusing" these operations, we eliminate two entire round-trips to the RAM.



---

## 🚀 Completed Roadmap
- [x] **Naive MatMul** - Baseline established.
- [x] **Blocked MatMul** - Cache locality optimization.
- [x] **SIMD (AVX2/FMA)** - Vectorization.
- [x] **Multi-threading (OpenMP)** - Scaling to all CPU cores.
- [x] **Linear Layer** - Fusing Bias + MatMul.
- [x] **Activation Kernels** - SIMD-accelerated ReLU.
- [x] **Fused Linear-ReLU** - End-to-end optimized layer.

## 🏗️ Next Steps
- [ ] Implement **Conv2D** (Convolutional) kernels.
- [ ] Build a **2-Layer MLP Forward Pass** using the existing library.