# ml-kernels-c 🧱

**Tech Stack:** `C` | `AVX2 / SIMD` | `OpenMP` | `Matrix Calculus` | `HPC`

---

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

# ml-kernels-c 🧱

High-performance AI/ML computational kernels written in C, optimized for modern CPU architectures.

## 🏆 Performance Milestone: Stable Diffusion Simulation
We successfully simulated a 30-step inference pass (UNet workload) for a 512x512 image generation (64x64x320 Latent Space).

| Operation | Naive Implementation | Optimized (GEMM/OpenMP) | Speedup |
| :--- | :--- | :--- | :--- |
| **Conv2D** | 720.05 ms | **323.61 ms** | 2.2x |
| **Transposed Conv** | 9611.62 ms | **359.24 ms** | **26.7x** |
| **Total 30-Step Gen** | ~310 seconds | **23.31 seconds** | **13.3x** |

---

## 💡 Spatial Kernel Engineering

### 1. Fast Convolution (im2col + MatMul)
By transforming the "sliding window" convolution into a matrix problem, we leverage our 85 GFLOPS MatMul engine. This removes the overhead of nested loops and maximizes L1/L2 cache hits.



### 2. Transposed Convolution (Upscaling)
The "Naive" version used an atomic "stamping" logic that caused massive CPU cache thrashing. Our optimized version uses a GEMM-based approach combined with a `col2im` accumulation step, which brought execution time down from **9.6 seconds to 0.3 seconds**.



---

## 🛠️ Project Structure
- `src/conv/`: im2col, Fast/Naive Conv2D, Transposed Conv, and col2im.
- `src/matmul/`: SIMD (AVX2), OpenMP, and Fused Linear-ReLU kernels.
- `src/activations/`: Max-Pooling, Softmax, and ReLU.
- `bench/src/`: Full Stable Diffusion simulation pass (`sim-sd-full.c`).

---

## 🚀 Roadmap
- [x] **Optimized MatMul Suite** (SIMD + Multi-threading).
- [x] **Fast Convolutional Layer** (im2col integration).
- [x] **Fast Upscaling** (Transposed Convolution).
- [x] **End-to-End Simulation** (UNet 30-step pass).
- [ ] **Weight Loading** (Parsing real model weights from .bin/.safetensors).
- [ ] **Winograd 3x3 Kernels** (Advanced spatial optimization).

## 🔨 Build Instructions
```bash
mingw32-make clean
mingw32-make all
./bin/sim-sd-full.exe
