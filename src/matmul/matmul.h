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

#endif
