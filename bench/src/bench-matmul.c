#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include "src/matmul/matmul.h"
#include "src/utils/timer.h"

void log_to_csv(const char* version, int M, int N, int K, double time_sec) {
    const char* filename = "results/performance.csv";
    FILE* check = fopen(filename, "r");
    int needs_header = (check == NULL);
    if (check) fclose(check);

    FILE* fp = fopen(filename, "a");
    if (fp) {
        if (needs_header) fprintf(fp, "version,M,N,K,seconds,gflops\n");
        // For ReLU, GFLOPS is technically just "Gops" since it's only a comparison, 
        // but we'll use the matrix size M*N*K for consistency in the log.
        double gflops = (2.0 * (double)M * (double)N * (double)K) / (time_sec * 1e9);
        fprintf(fp, "%s,%d,%d,%d,%.6f,%.2f\n", version, M, N, K, time_sec, gflops);
        fclose(fp);
    }
}

int verify(float* ref, float* test, int size) {
    for (int i = 0; i < size; i++) {
        if (fabsf(ref[i] - test[i]) > 1e-2) {
            printf("\n[!] Verification FAILED at index %d (Ref: %f, Test: %f)\n", i, ref[i], test[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    if (argc == 4) {
        M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]);
    }

    float* A = (float*)_aligned_malloc(M * K * sizeof(float), 64);
    float* B = (float*)_aligned_malloc(K * N * sizeof(float), 64);
    float* bias = (float*)_aligned_malloc(N * sizeof(float), 64);
    float* C_ref = (float*)_aligned_malloc(M * N * sizeof(float), 64);
    float* C_test = (float*)_aligned_malloc(M * N * sizeof(float), 64);
    float* C_linear_ref = (float*)_aligned_malloc(M * N * sizeof(float), 64);
    float* C_relu_ref = (float*)_aligned_malloc(M * N * sizeof(float), 64);

    // Initialize with some negative values to properly test ReLU
    for (int i = 0; i < M * K; i++) A[i] = ((float)rand() / RAND_MAX) - 0.5f;
    for (int i = 0; i < K * N; i++) B[i] = ((float)rand() / RAND_MAX) - 0.5f;
    for (int i = 0; i < N; i++) bias[i] = ((float)rand() / RAND_MAX) - 0.5f;

    printf("Benchmarking M=N=K=%d...\n", M);

    // --- 1. Naive Benchmark ---
    matmul_naive(A, B, C_ref, M, N, K); 
    double s1 = now_seconds();
    matmul_naive(A, B, C_ref, M, N, K);
    double e1 = now_seconds();
    double t_naive = e1 - s1;
    log_to_csv("naive_O3", M, N, K, t_naive);

    // --- 2. Blocked Benchmark ---
    matmul_block(A, B, C_test, M, N, K); 
    double s2 = now_seconds();
    matmul_block(A, B, C_test, M, N, K);
    double e2 = now_seconds();
    double t_block = e2 - s2;
    log_to_csv("blocked_O3", M, N, K, t_block);
    verify(C_ref, C_test, M * N);

    // --- 3. SIMD Benchmark ---
    matmul_simd(A, B, C_test, M, N, K); 
    double s3 = now_seconds();
    matmul_simd(A, B, C_test, M, N, K);
    double e3 = now_seconds();
    double t_simd = e3 - s3;
    log_to_csv("simd_O3", M, N, K, t_simd);
    verify(C_ref, C_test, M * N);

    // --- 4. OpenMP Benchmark ---
    matmul_openmp(A, B, C_test, M, N, K); 
    double s4 = now_seconds();
    matmul_openmp(A, B, C_test, M, N, K);
    double e4 = now_seconds();
    double t_openmp = e4 - s4;
    log_to_csv("openmp_O3", M, N, K, t_openmp);
    verify(C_ref, C_test, M * N);

    // --- 5. Linear SIMD Benchmark ---
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_linear_ref[i * N + j] = C_ref[i * N + j] + bias[j];
        }
    }
    linear_layer_simd(A, B, bias, C_test, M, N, K);
    double s5 = now_seconds();
    linear_layer_simd(A, B, bias, C_test, M, N, K);
    double e5 = now_seconds();
    double t_linear = e5 - s5;
    log_to_csv("linear_simd_O3", M, N, K, t_linear);
    verify(C_linear_ref, C_test, M * N);

    // --- 6. Linear OpenMP Benchmark ---
    linear_layer_openmp(A, B, bias, C_test, M, N, K);
    double s6 = now_seconds();
    linear_layer_openmp(A, B, bias, C_test, M, N, K);
    double e6 = now_seconds();
    double t_linear_openmp = e6 - s6;
    log_to_csv("linear_openmp_O3", M, N, K, t_linear_openmp);
    verify(C_linear_ref, C_test, M * N);

    // --- 7. ReLU Benchmarks ---
    // Prepare ReLU reference from the linear result
    for (int i = 0; i < M * N; i++) {
        C_relu_ref[i] = C_linear_ref[i] > 0 ? C_linear_ref[i] : 0;
    }

    // ReLU Naive
    double s7 = now_seconds();
    relu_naive(C_test, M * N);
    double e7 = now_seconds();
    double t_relu_naive = e7 - s7;
    // (Reset C_test to linear result for SIMD test)
    for(int i=0; i<M*N; i++) C_test[i] = C_linear_ref[i];

    // ReLU SIMD
    double s8 = now_seconds();
    relu_simd(C_test, M * N);
    double e8 = now_seconds();
    double t_relu_simd = e8 - s8;
    log_to_csv("relu_simd_O3", M, N, K, t_relu_simd);

    // --- 8. Fused Linear+ReLU Benchmark ---
    linear_relu_fused_openmp(A, B, bias, C_test, M, N, K);
    double s9 = now_seconds();
    linear_relu_fused_openmp(A, B, bias, C_test, M, N, K);
    double e9 = now_seconds();
    double t_linear_relu_fused = e9 - s9;
    log_to_csv("linear_relu_fused_O3", M, N, K, t_linear_relu_fused);

    // --- Results ---
    printf("\nPerformance Results:\n");
    printf("  Naive:             %.4f s\n", t_naive);
    printf("  Blocked:           %.4f s (Speedup: %.2fx)\n", t_block, t_naive / t_block);
    printf("  SIMD:              %.4f s (Speedup: %.2fx)\n", t_simd, t_naive / t_simd);
    printf("  OpenMP:            %.4f s (Speedup: %.2fx)\n", t_openmp, t_naive / t_openmp);
    printf("  Linear SIMD:       %.4f s (Speedup: %.2fx)\n", t_linear, t_naive / t_linear);
    printf("  Linear OpenMP:     %.4f s (Speedup: %.2fx)\n", t_linear_openmp, t_naive / t_linear_openmp);
    printf("  ReLU SIMD:         %.6f s (Speedup vs Naive: %.2fx)\n", t_relu_simd, t_relu_naive / t_relu_simd);
    printf("  Linear+ReLU Fused: %.4f s (Speedup: %.2fx)\n", t_linear_relu_fused, t_naive / t_linear_relu_fused);

    if (verify(C_relu_ref, C_test, M * N)) {
        printf("\nAll verifications passed (including ReLU)!\n");
    }

    _aligned_free(A); _aligned_free(B); _aligned_free(bias);
    _aligned_free(C_ref); _aligned_free(C_test);
    _aligned_free(C_linear_ref); _aligned_free(C_relu_ref);
    
    return 0;
}