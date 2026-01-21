#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include "src/utils/timer.h"
#include "src/matmul/matmul.h"
#include "src/conv/conv.h"

#define STEPS 30
#define LATENT_H 64
#define LATENT_W 64
#define CHANNELS 320

void simulate_sampling_step(int step_num, float* latent, float* kernel, float* out, float* up_out) {
    double t0 = get_time();
    
    // 1. Convolution Block (im2col + MatMul)
    conv2d_fast(latent, kernel, out, LATENT_H, LATENT_W, CHANNELS, 3, CHANNELS, 1, 1);
    double t1 = get_time();

    // 2. Attention Block (Pure MatMul)
    matmul_openmp(latent, kernel, out, 4096, CHANNELS, CHANNELS);
    double t2 = get_time();

    // 3. Upscaling (The likely 8-second bottleneck)
    conv_transposed_fast(out, kernel, up_out, LATENT_H, LATENT_W, CHANNELS, 3, CHANNELS, 2, 1);
    double t3 = get_time();

    printf("Step %d | Conv: %.2fms | Attn: %.2fms | Transposed: %.2fms | Total: %.2fms\n", 
            step_num, (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t3-t0)*1000);
}

int main() {
    printf("🚀 SD Simulation (Memory-Reuse Mode)\n\n");

    // Allocate ONCE outside the loop
    float *latent = (float*)malloc(64 * 64 * 320 * sizeof(float));
    float *kernel = (float*)malloc(3 * 3 * 320 * 320 * sizeof(float));
    float *out    = (float*)malloc(64 * 64 * 320 * sizeof(float));
    float *up_out = (float*)malloc(128 * 128 * 320 * sizeof(float));

    double start_gen = get_time();

    for (int i = 1; i <= STEPS; i++) {
        simulate_sampling_step(i, latent, kernel, out, up_out);
    }

    double end_gen = get_time();
    printf("\n✅ Total Generation Time: %.2f seconds\n", (end_gen - start_gen));

    free(latent); free(kernel); free(out); free(up_out);
    return 0;
}