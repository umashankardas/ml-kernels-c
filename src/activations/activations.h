#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

// Week 3: Activations
void relu_naive(float* x, int n);
void relu_simd(float* x, int n);

// Week 4: Softmax
void softmax_naive(float* x, int n);
void softmax_simd(float* x, int n);

#endif