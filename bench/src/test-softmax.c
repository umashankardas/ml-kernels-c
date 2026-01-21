#include <stdio.h>
#include <stdlib.h>  // Essential for memory functions
#include <math.h>
#include <float.h>
#include <malloc.h>  // Required for Windows aligned allocation
#include "src/activations/activations.h"

// A simple wrapper to make it work on Windows
void* malloc_aligned(size_t size, size_t alignment) {
    return _aligned_malloc(size, alignment);
}

void free_aligned(void* ptr) {
    _aligned_free(ptr);
}
void verify_softmax(float* input, int n, const char* test_name) {
    printf("Testing: %s\n", test_name);
    
    // Run the kernel
    softmax_simd(input, n);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        // 1. Check for NaN (the most common Softmax failure)
        if (isnan(input[i])) {
            printf("  ❌ FAILED: Found NaN at index %d\n", i);
            return;
        }
        sum += input[i];
    }

    // 2. Check if probabilities sum to ~1.0
    if (fabs(sum - 1.0f) > 1e-5) {
        printf("  ❌ FAILED: Sum is %f (expected 1.0)\n", sum);
    } else {
        printf("  ✅ PASSED: Probabilities sum to 1.0\n");
    }

    // 3. Print a sample of the output
    printf("  Sample: [%.4f, %.4f, %.4f ...]\n\n", input[0], input[1], input[2]);
}

int main() {

    printf("--- SOFTMAX TEST SUITE STARTING ---\n");
    fflush(stdout);
    
    int n = 1024; // Must be multiple of 8 for our SIMD kernel
    float* data = (float*)malloc_aligned(32, n * sizeof(float));

    // TEST 1: Uniform numbers (should result in 1/n probabilities)
    for (int i = 0; i < n; i++) data[i] = 1.0f;
    verify_softmax(data, n, "Uniform Inputs");

    // TEST 2: Extreme High Values (The Stability Test)
    // Without max-subtraction, expf(1000) will overflow to Inf
    for (int i = 0; i < n; i++) data[i] = 1000.0f + (i % 10);
    verify_softmax(data, n, "High-Value Stability (1000+)");

    // TEST 3: One Clear Winner
    for (int i = 0; i < n; i++) data[i] = -10.0f;
    data[42] = 50.0f; // Index 42 should have ~100% probability
    verify_softmax(data, n, "Single Winner (Index 42)");
    printf("  Winner Prob: %.6f\n", data[42]);

    free_aligned(data);
    return 0;
}