#include "conv.h"
#include <math.h>

void conv2d_naive(float* input, float* kernel, float* output,
                  int in_h, int in_w, int in_c,
                  int k_size, int out_c, int stride, int padding) {
    
    // Calculate output dimensions based on standard CNN formula
    int out_h = (in_h + 2 * padding - k_size) / stride + 1;
    int out_w = (in_w + 2 * padding - k_size) / stride + 1;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                
                float sum = 0.0f;
                
                for (int kh = 0; kh < k_size; kh++) {
                    for (int kw = 0; kw < k_size; kw++) {
                        for (int ic = 0; ic < in_c; ic++) {
                            
                            // Map output coordinate back to input coordinate
                            // Apply stride and subtract padding offset
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;

                            // Check if the current kernel position is within image bounds
                            // If it's in the 'padding zone', it contributes 0 (Implicit Zero Padding)
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int input_idx = (ih * in_w + iw) * in_c + ic;
                                // Kernel indexing: [kh, kw, in_c, out_c]
                                int kernel_idx = ((kh * k_size + kw) * in_c + ic) * out_c + oc;
                                
                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                }
                output[(oh * out_w + ow) * out_c + oc] = sum;
            }
        }
    }
}