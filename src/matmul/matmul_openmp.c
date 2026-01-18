#include <immintrin.h>
#include <omp.h>
#include "src/matmul/matmul.h"

#define BLOCK_SIZE 32

void matmul_openmp(const float* A, const float* B, float* C, int M, int N, int K) {
    // 1. Initialize C to zero
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // 2. Parallelize the tile distribution
    // collapse(2) tells OpenMP to parallelize both bi and bj loops together
    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < M; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            // We keep bk inside so cores don't fight over the same C memory
            for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
                
                for (int i = bi; i < bi + BLOCK_SIZE && i < M; i++) {
                    for (int k = bk; k < bk + BLOCK_SIZE && k < K; k++) {
                        __m256 va = _mm256_set1_ps(A[i * K + k]);
                        
                        for (int j = bj; j < bj + BLOCK_SIZE && j < N; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 vc = _mm256_loadu_ps(&C[i * N + j]);
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(&C[i * N + j], vc);
                        }
                    }
                }
            }
        }
    }
}