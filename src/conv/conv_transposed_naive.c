#include "conv.h"

void conv_transposed_naive(float* input, float* kernel, float* output,
                           int in_h, int in_w, int in_c,
                           int k_size, int out_c, int stride, int padding) {
    
    int out_h = (in_h - 1) * stride - 2 * padding + k_size;
    int out_w = (in_w - 1) * stride - 2 * padding + k_size;

    // Initialize output with zeros
    for (int i = 0; i < out_h * out_w * out_c; i++) output[i] = 0;

    for (int ic = 0; ic < in_c; ic++) {
        for (int ih = 0; ih < in_h; ih++) {
            for (int iw = 0; iw < in_w; iw++) {
                
                float input_val = input[(ih * in_w + iw) * in_c + ic];

                for (int oc = 0; oc < out_c; oc++) {
                    for (int kh = 0; kh < k_size; kh++) {
                        for (int kw = 0; kw < k_size; kw++) {
                            
                            int oh = ih * stride + kh - padding;
                            int ow = iw * stride + kw - padding;

                            if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                int out_idx = (oh * out_w + ow) * out_c + oc;
                                int ker_idx = ((kh * k_size + kw) * in_c + ic) * out_c + oc;
                                output[out_idx] += input_val * kernel[ker_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}