#include <stdio.h>
#include <stdlib.h>
#include "src/utils/timer.h"
#include "src/conv/conv.h"

#define H 128
#define W 128
#define C 64
#define K 3
#define OUT_C 64

int main() {
    float *in = malloc(H * W * C * sizeof(float));
    float *ker = malloc(K * K * C * OUT_C * sizeof(float));
    float *out = malloc(H * W * OUT_C * sizeof(float));

    printf("Benchmarking 128x128x64 Conv (3x3 kernel)...\n");

    // Test Naive
    double start = get_time();
    conv2d_naive(in, ker, out, H, W, C, K, OUT_C, 1, 1);
    printf("Naive Conv: %.2f ms\n", (get_time() - start) * 1000.0);

    // Test Fast (im2col + MatMul)
    start = get_time();
    conv2d_fast(in, ker, out, H, W, C, K, OUT_C, 1, 1);
    printf("Fast Conv (im2col): %.2f ms\n", (get_time() - start) * 1000.0);

    free(in); free(ker); free(out);
    return 0;
}