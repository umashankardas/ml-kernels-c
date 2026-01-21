// 1. The Naive Implementation (Numerically Stable)
// This implementation follows the 3-pass pattern: Find Max → Compute Exp & Sum → Normalize.


#include <math.h>
#include <float.h>
#include <immintrin.h>
#include "src/activations/activations.h"

void softmax_naive(float* x, int n) {
    // Pass 1: Find the maximum value for numerical stability
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // Pass 2: Compute exponentials and their sum
    // We subtract max_val from x[i] to ensure the power is <= 0
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // Pass 3: Normalize the vector so it sums to 1.0
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}


// Helper to find the maximum float inside an 8-lane AVX register
static inline float _mm256_reduce_max_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    __m128 tmp = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2));
    lo = _mm_max_ps(lo, tmp);
    tmp = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 1, 1, 1));
    lo = _mm_max_ps(lo, tmp);
    return _mm_cvtss_f32(lo);
}

// Helper to sum all 8 floats inside an AVX register
static inline float _mm256_reduce_sum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 tmp = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, tmp);
    tmp = _mm_movehl_ps(tmp, lo);
    lo = _mm_add_ss(lo, tmp);
    return _mm_cvtss_f32(lo);
}

void softmax_simd(float* x, int n) {
    // 1. SIMD Max Find
    __m256 max_v = _mm256_set1_ps(-FLT_MAX);
    for (int i = 0; i < n; i += 8) {
        max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(&x[i]));
    }
    float global_max = _mm256_reduce_max_ps(max_v);
    __m256 gmax_v = _mm256_set1_ps(global_max);

    // 2. SIMD Exp & Sum
    __m256 sum_v = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        v = _mm256_sub_ps(v, gmax_v);
        
        // Scalar fallback for expf (math.h expf isn't natively vectorized in standard C)
        float tmp[8];
        _mm256_storeu_ps(tmp, v);
        for(int j=0; j<8; j++) tmp[j] = expf(tmp[j]);
        
        __m256 ev = _mm256_loadu_ps(tmp);
        _mm256_storeu_ps(&x[i], ev);
        sum_v = _mm256_add_ps(sum_v, ev);
    }
    float global_sum = _mm256_reduce_sum_ps(sum_v);
    __m256 inv_sum_v = _mm256_set1_ps(1.0f / global_sum);

    // 3. SIMD Normalize
    for (int i = 0; i < n; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&x[i], _mm256_mul_ps(v, inv_sum_v));
    }
}