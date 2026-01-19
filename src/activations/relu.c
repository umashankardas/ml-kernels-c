#include <immintrin.h>
#include "src/matmul/matmul.h" // We can put activation prototypes here or in a new header

void relu_naive(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

void relu_simd(float* data, int size) {
    // Create a register filled with 0.0f
    __m256 vzero = _mm256_setzero_ps();

    for (int i = 0; i < size; i += 8) {
        // Load 8 floats
        __m256 vx = _mm256_loadu_ps(&data[i]);
        
        // Use the MAX intrinsic: it picks the larger of (0, x)
        __m256 vres = _mm256_max_ps(vzero, vx);
        
        // Store back to memory
        _mm256_storeu_ps(&data[i], vres);
    }
}