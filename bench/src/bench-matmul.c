#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "src/matmul/matmul.h"
#include "src/utils/timer.h"

// Helper to handle CSV logging
void log_to_csv(const char* version, int M, int N, int K, double time_sec) {
    const char* filename = "results/performance.csv";
    
    FILE* check = fopen(filename, "r");
    int needs_header = (check == NULL);
    if (check) fclose(check);

    FILE* fp = fopen(filename, "a");
    if (!fp) {
        perror("Could not open results file");
        return;
    }

    if (needs_header) {
        fprintf(fp, "version,M,N,K,seconds,gflops\n");
    }

    // GFLOPS = (2 * M * N * K) / (time * 1e9)
    double gflops = (2.0 * (double)M * (double)N * (double)K) / (time_sec * 1e9);
    
    fprintf(fp, "%s,%d,%d,%d,%.6f,%.2f\n", version, M, N, K, time_sec, gflops);
    fclose(fp);
}

int main(int argc, char** argv) {
    // Default size
    int M = 512, N = 512, K = 512;

    // Handle command line arguments: ./bench-matmul M N K
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    } else if (argc != 1) {
        printf("Usage: %s [M N K]\n", argv[0]);
        return 1;
    }

    // Allocation with 64-byte alignment
    float* A = (float*)_aligned_malloc(M * K * sizeof(float), 64);
    float* B = (float*)_aligned_malloc(K * N * sizeof(float), 64);
    float* C = (float*)_aligned_malloc(M * N * sizeof(float), 64);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize with data
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;

    // --- Warm-up Run ---
    // This primes the cache and ensures the CPU clock frequency is boosted
    matmul_naive(A, B, C, M, N, K);

    // --- Benchmark Start ---
    double start = now_seconds();
    matmul_naive(A, B, C, M, N, K);
    double end = now_seconds();
    // --- Benchmark End ---

    double elapsed = end - start;
    printf("Naive matmul (%dx%dx%d): %.6f sec (%.2f GFLOPS)\n", 
            M, N, K, elapsed, (2.0 * M * N * K) / (elapsed * 1e9));

    log_to_csv("naive_O3", M, N, K, elapsed);

    _aligned_free(A); 
    _aligned_free(B); 
    _aligned_free(C);
    
    return 0;
}