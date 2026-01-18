#include <immintrin.h> // Header for AVX intrinsics
#include "src/matmul/matmul.h"

#define BLOCK_SIZE 32

void matmul_simd(const float* A, const float* B, float* C, int M, int N, int K) {
    // Initialize C to zero
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    for (int bi = 0; bi < M; bi += BLOCK_SIZE) {
        for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                
                // Inner loops
                for (int i = bi; i < bi + BLOCK_SIZE && i < M; i++) {
                    for (int k = bk; k < bk + BLOCK_SIZE && k < K; k++) {
                        
                        // Load one element of A and broadcast it to all 8 slots of a YMM register
                        __m256 va = _mm256_set1_ps(A[i * K + k]);
                        
                        for (int j = bj; j < bj + BLOCK_SIZE && j < N; j += 8) {
                            // Load 8 elements of B (must be 32-byte aligned)
                            __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                            
                            // Load 8 elements of C
                            __m256 vc = _mm256_loadu_ps(&C[i * N + j]);
                            
                            // Multiply-Accumulate: C = C + (A * B)
                            // Using Fused Multiply-Add (FMA) if supported, else mul + add
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            
                            // Store results back to C
                            _mm256_storeu_ps(&C[i * N + j], vc);
                        }
                    }
                }
            }
        }
    }
}