#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cuda_runtime.h>

// ReLU activation: output[i] = max(0, input[i])
void relu_forward(const float* d_input, float* d_output, int size);

// ReLU backward: grad_input[i] = grad_output[i] if input[i] > 0, else 0
void relu_backward(const float* d_grad_output, const float* d_input,
                   float* d_grad_input, int size);

// Softmax activation (row-wise): output[i,j] = exp(input[i,j]) / sum_k exp(input[i,k])
void softmax_forward(const float* d_input, float* d_output, int B, int N);

// Sigmoid activation: output[i] = 1 / (1 + exp(-input[i]))
void sigmoid_forward(const float* d_input, float* d_output, int size);

// Sigmoid backward: grad_input[i] = grad_output[i] * sigmoid[i] * (1 - sigmoid[i])
void sigmoid_backward(const float* d_grad_output, const float* d_output,
                      float* d_grad_input, int size);

#endif // ACTIVATIONS_H
