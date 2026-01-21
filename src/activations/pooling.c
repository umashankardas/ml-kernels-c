#include <float.h>

void max_pool2d(float* input, float* output, int in_h, int in_w, int channels, int stride) {
    int out_h = in_h / stride;
    int out_w = in_w / stride;

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float max_val = -FLT_MAX;
                // Look at the window (stride x stride)
                for (int kh = 0; kh < stride; kh++) {
                    for (int kw = 0; kw < stride; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        float val = input[(ih * in_w + iw) * channels + c];
                        if (val > max_val) max_val = val;
                    }
                }
                output[(oh * out_w + ow) * channels + c] = max_val;
            }
        }
    }
}