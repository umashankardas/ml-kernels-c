#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "src/utils/timer.h"
#include "src/matmul/matmul.h"
#include "src/activations/activations.h"

// Forward declaration for timer if needed by compiler
double get_time(void);

void* malloc_aligned(size_t size, size_t alignment) {
    return _aligned_malloc(size, alignment);
}

// Reality-Check Parameters for Stable Diffusion 1.5 workload
#define STEPS 4            // Standard denoising steps
#define LAYERS_PER_STEP 70  // UNet Linear + Attention layers
#define MODEL_DIM 2048      // Hidden dimension (d_model)
#define M_SIZE 64           // Number of patches/tokens processed in parallel

int main() {
    int M = M_SIZE;
    int N = MODEL_DIM;
    int K = MODEL_DIM;

    printf("--- STABLE DIFFUSION FULL WORKLOAD SIMULATOR ---\n");
    printf("Config: %d steps, %d layers/step, Dim: %dx%d, Batch: %d\n", 
            STEPS, LAYERS_PER_STEP, N, K, M);

    // 1. Allocate Matrix-sized memory instead of Vector-sized
    // Matrix A: [M x K], Matrix B: [K x N], Matrix C: [M x N]
    float *A    = (float*)malloc_aligned(M * K * sizeof(float), 32);
    float *B    = (float*)malloc_aligned(K * N * sizeof(float), 32);
    float *C    = (float*)malloc_aligned(M * N * sizeof(float), 32);
    float *bias = (float*)malloc_aligned(N * sizeof(float), 32);

    if (!A || !B || !C || !bias) {
        printf("Failed to allocate memory for simulation.\n");
        return 1;
    }

    // 2. Initialize with dummy weights
    for(int i = 0; i < M * K; i++) A[i] = 1.0f;
    for(int i = 0; i < K * N; i++) B[i] = 0.01f;
    for(int i = 0; i < N; i++) bias[i] = 0.1f;

    printf("Starting Stress Test...\n\n");
    double total_start = get_time();

    for (int s = 1; s <= STEPS; s++) {
        double step_start = get_time();
        
        for (int l = 0; l < LAYERS_PER_STEP; l++) {
            // Processing a matrix of M tokens against N weights
            linear_relu_fused_openmp(A, B, bias, C, M, N, K);
            
            // Swap pointers to simulate hidden state flow
            float* temp = A; A = C; C = temp; 
        }

        double step_end = get_time();
        printf("  Step %d: %.2f ms\n", s, (step_end - step_start) * 1000.0);
    }

    double total_end = get_time();
    double total_time = total_end - total_start;

    // 3. Performance Metrics
    // Floating point operations: 2.0 * M * N * K per layer
    double total_ops = (double)2.0 * M * N * K * STEPS * LAYERS_PER_STEP;
    double gflops = (total_ops / 1e9) / total_time;

    printf("\n-----------------------------------------\n");
    printf("TOTAL SIMULATED INFERENCE TIME: %.4f seconds\n", total_time);
    printf("AVERAGE GFLOPS: %.2f\n", gflops);
    printf("-----------------------------------------\n");
    
    _aligned_free(A);
    _aligned_free(B);
    _aligned_free(C);
    _aligned_free(bias);
    
    return 0;
}