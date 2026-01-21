#include "conv.h"
#include "src/matmul/matmul.h"
#include <stdlib.h>
#include <malloc.h>

void conv_transposed_fast(float* input, float* kernel, float* output,
                          int in_h, int in_w, int in_c,
                          int k_size, int out_c, int stride, int padding) {
    
    int out_h = (in_h - 1) * stride - 2 * padding + k_size;
    int out_w = (in_w - 1) * stride - 2 * padding + k_size;

    // In a fast transposed conv, we use MatMul to generate the unrolled columns
    // and then use col2im to sum them into the final image.
    int col_rows = out_c * k_size * k_size;
    int col_cols = in_h * in_w;

    float* data_col = (float*)_aligned_malloc(col_rows * col_cols * sizeof(float), 32);

    // Call your optimized MatMul
    matmul_openmp(kernel, input, data_col, col_rows, col_cols, in_c);

    // Fold the columns back into the image grid
    col2im(data_col, out_c, out_h, out_w, k_size, stride, padding, output);

    _aligned_free(data_col);
}