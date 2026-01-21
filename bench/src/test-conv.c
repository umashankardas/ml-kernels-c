#include <stdio.h>
#include "src/conv/conv.h"

int main() {
    // 1. Tiny 2x2 input
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // 2. 2x2 kernel of all 1s
    float kernel[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    // 3. Expected output: (2-1)*2 + 2 = 4x4
    float output[16] = {0};

    printf("Running Transposed Convolution (Upscaling) Test...\n");

    // conv_transposed_naive(input, kernel, output, in_h, in_w, in_c, k_size, out_c, stride, padding)
    conv_transposed_naive(input, kernel, output, 2, 2, 1, 2, 1, 2, 0);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%4.1f ", output[i * 4 + j]);
        }
        printf("\n");
    }

    return 0;
}