#ifndef ADAM_H
#define ADAM_H

#include <cuda_runtime.h>

// Adam optimizer update for a single parameter
// param: parameter to update
// grad: gradient
// m: first moment estimate
// v: second moment estimate
// lr: learning rate
// beta1: exponential decay rate for first moment
// beta2: exponential decay rate for second moment
// epsilon: small constant for numerical stability
// beta1_t: beta1^t (for bias correction)
// beta2_t: beta2^t (for bias correction)
// weight_decay: L2 regularization coefficient (default: 0.0)
void adam_update(float* d_param, const float* d_grad, float* d_m, float* d_v,
                 float lr, float beta1, float beta2, float epsilon,
                 float beta1_t, float beta2_t, int size, float weight_decay = 0.0f);

#endif // ADAM_H
