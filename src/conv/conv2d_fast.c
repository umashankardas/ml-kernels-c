#include "conv.h"
#include "src/matmul/matmul.h"
#include <stdlib.h>
#include <malloc.h>

void conv2d_fast(float* input, float* kernel, float* output,
                 int in_h, int in_w, int in_c,
                 int k_size, int out_c, int stride, int padding) {
    
    int out_h = (in_h + 2 * padding - k_size) / stride + 1;
    int out_w = (in_w + 2 * padding - k_size) / stride + 1;

    // 1. Calculate workspace size for im2col
    // Each output pixel needs (k_size * k_size * in_c) floats
    size_t col_rows = k_size * k_size * in_c;
    size_t col_cols = out_h * out_w;
    float* data_col = (float*)_aligned_malloc(col_rows * col_cols * sizeof(float), 32);

    // 2. Transform Image to Matrix
    im2col(input, in_c, in_h, in_w, k_size, stride, padding, data_col);

    // 3. GEMM (General Matrix Multiplication)
    // Matrix A (Weights): [out_c x col_rows]
    // Matrix B (Data):    [col_rows x col_cols]
    // Matrix C (Output):  [out_c x col_cols]
    
    // Note: We use your optimized OpenMP MatMul here!
    // Since your linear_openmp is optimized for [M x N] * [N x K], 
    // we map our variables accordingly.
    matmul_openmp(kernel, data_col, output, out_c, col_cols, col_rows);

    _aligned_free(data_col);
}