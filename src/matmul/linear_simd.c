#include <immintrin.h>
#include "src/matmul/matmul.h"

#define BLOCK_SIZE 32

void linear_layer_simd(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    // 1. Initialize C with the Bias vector
    // Each row 'i' gets the same bias vector added to it
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = bias[j];
        }
    }

    // 2. Perform Blocked SIMD MatMul (Accumulating onto the Bias)
    for (int bi = 0; bi < M; bi += BLOCK_SIZE) {
        for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                
                for (int i = bi; i < bi + BLOCK_SIZE && i < M; i++) {
                    for (int k = bk; k < bk + BLOCK_SIZE && k < K; k++) {
                        __m256 va = _mm256_set1_ps(A[i * K + k]);
                        
                        for (int j = bj; j < bj + BLOCK_SIZE && j < N; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 vc = _mm256_loadu_ps(&C[i * N + j]);
                            
                            // C = C + (A * B) -> Since C already has bias, this completes the layer
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(&C[i * N + j], vc);
                        }
                    }
                }
            }
        }
    }
}