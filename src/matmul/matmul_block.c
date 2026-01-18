#include "src/matmul/matmul.h"

#define BLOCK_SIZE 32

void matmul_block(
    const float* A, 
    const float* B, 
    float* C, 
    int M, int N, int K
) {
    // 1. Initialize C to zero
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // 2. The 6-loop nest
    // Outer loops move tile by tile
    for (int bi = 0; bi < M; bi += BLOCK_SIZE) {
        for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
            for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
                
                // Inner loops perform matmul on the tiles
                for (int i = bi; i < bi + BLOCK_SIZE && i < M; i++) {
                    for (int k = bk; k < bk + BLOCK_SIZE && k < K; k++) {
                        
                        // Register Promotion: Keep A[i][k] in a register
                        float temp_a = A[i * K + k];
                        
                        for (int j = bj; j < bj + BLOCK_SIZE && j < N; j++) {
                            C[i * N + j] += temp_a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}