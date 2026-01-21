#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "src/utils/timer.h"
#include "src/activations/activations.h"
#include "src/matmul/matmul.h"

// Windows-friendly alignment wrapper
void* malloc_aligned(size_t size, size_t alignment) {
    return _aligned_malloc(size, alignment);
}

void free_aligned(void* ptr) {
    _aligned_free(ptr);
}

// 2048 is a standard "Medium" model hidden dimension (e.g., Mistral/Llama small)
#define HIDDEN_SIZE 4096

int main() {
    int N = HIDDEN_SIZE;
    
    // Allocate Aligned Memory
    float *input   = (float*)malloc_aligned(N * sizeof(float), 32);
    float *weights = (float*)malloc_aligned(N * N * sizeof(float), 32);
    float *bias    = (float*)malloc_aligned(N * sizeof(float), 32);
    float *output  = (float*)malloc_aligned(N * sizeof(float), 32);

    if (!input || !weights || !bias || !output) return 1;

    // Dummy Init
    for(int i=0; i<N*N; i++) weights[i] = 0.001f;
    for(int i=0; i<N; i++) { input[i] = 1.0f; bias[i] = 0.1f; }

    printf("--- SIMULATING 1 TRANSFORMER BLOCK PASS ---\n");
    
    double start = get_time();

    // 1. Attention Linear Layer (M=1, K=2048, N=2048)
    linear_relu_fused_openmp(input, weights, bias, output, 1, N, N);

    // 2. Feed Forward Expansion (M=1, K=2048, N=2048)
    linear_relu_fused_openmp(output, weights, bias, input, 1, N, N);

    // 3. Final Softmax Logits
    softmax_simd(input, N);

    double end = get_time();

    printf("Simulation Results:\n");
    printf("  Latency: %.4f ms\n", (end - start) * 1000.0);
    printf("  Status: SUCCESS\n");

    free_aligned(input);
    free_aligned(weights);
    free_aligned(bias);
    free_aligned(output);

    return 0;
}