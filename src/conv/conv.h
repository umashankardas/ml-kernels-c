#ifndef CONV_H
#define CONV_H

/**
 * Direct 2D Convolution
 * input:  [in_h x in_w x in_c]
 * kernel: [k_h x k_w x in_c x out_c]
 * output: [out_h x out_w x out_c]
 */
void conv2d_naive(float* input, float* kernel, float* output,
                  int in_h, int in_w, int in_c,
                  int k_size, int out_c, int stride, int padding);

void conv2d_im2col(float* input, float* kernel, float* output,
                   int in_h, int in_w, int in_c,
                   int k_size, int out_c, int stride, int padding);

void im2col(float* data_im, int channels, int height, int width,
            int ksize, int stride, int pad, float* data_col);

void conv2d_fast(float* input, float* kernel, float* output,
                 int in_h, int in_w, int in_c,
                 int k_size, int out_c, int stride, int padding);
                 
void conv_transposed_naive(float* input, float* kernel, float* output,
                           int in_h, int in_w, int in_c,
                           int k_size, int out_c, int stride, int padding);

void conv_transposed_fast(float* input, float* kernel, float* output,
                          int in_h, int in_w, int in_c, int k_size, int out_c, int stride, int padding);  

void col2im(float* data_col, int channels, int height, int width,
                int ksize, int stride, int pad, float* data_im);
                          
#endif