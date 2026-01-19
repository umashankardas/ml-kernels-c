#ifndef MATMUL_H
#define MATMUL_H

void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

void matmul_block(const float* A, const float* B, float* C, int M, int N, int K);

void matmul_simd(const float* A, const float* B, float* C, int M, int N, int K);

void matmul_openmp(const float* A, const float* B, float* C, int M, int N, int K);

void linear_layer_simd(const float* A, const float* B, const float* bias, float* C, int M, int N, int K);

void linear_layer_openmp(const float* A, const float* B, const float* bias, float* C, int M, int N, int K);

void relu_naive(float* data, int size);
void relu_simd(float* data, int size);

void linear_relu_fused_openmp(const float* A, const float* B, const float* bias, float* C, int M, int N, int K);

#endif
