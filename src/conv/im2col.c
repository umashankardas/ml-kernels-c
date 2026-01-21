#include <stdio.h>

void im2col(float* data_im, int channels, int height, int width,
            int ksize, int stride, int pad, float* data_col) {
    
    int out_h = (height + 2 * pad - ksize) / stride + 1;
    int out_w = (width + 2 * pad - ksize) / stride + 1;
    int channel_size = height * width;

    // This loop essentially 'unrolls' the image
    for (int c = 0; c < channels; c++) {
        for (int kh = 0; kh < ksize; kh++) {
            for (int kw = 0; kw < ksize; kw++) {
                for (int h = 0; h < out_h; h++) {
                    for (int w = 0; w < out_w; w++) {
                        
                        int im_row = h * stride + kh - pad;
                        int im_col = w * stride + kw - pad;
                        
                        int col_index = ((c * ksize + kh) * ksize + kw) * out_h * out_w + (h * out_w + w);
                        
                        if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                            data_col[col_index] = data_im[c * channel_size + im_row * width + im_col];
                        } else {
                            data_col[col_index] = 0; // Padding
                        }
                    }
                }
            }
        }
    }
}

void col2im(float* data_col, int channels, int height, int width,
            int ksize, int stride, int pad, float* data_im) {
    
    int out_h = (height + 2 * pad - ksize) / stride + 1;
    int out_w = (width + 2 * pad - ksize) / stride + 1;

    // Initialize image with zeros
    for (int i = 0; i < channels * height * width; i++) data_im[i] = 0;

    for (int c = 0; c < channels; c++) {
        for (int kh = 0; kh < ksize; kh++) {
            for (int kw = 0; kw < ksize; kw++) {
                for (int h = 0; h < out_h; h++) {
                    for (int w = 0; w < out_w; w++) {
                        int im_row = h * stride + kh - pad;
                        int im_col = w * stride + kw - pad;
                        
                        if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                            int col_index = ((c * ksize + kh) * ksize + kw) * out_h * out_w + (h * out_w + w);
                            data_im[c * height * width + im_row * width + im_col] += data_col[col_index];
                        }
                    }
                }
            }
        }
    }
}