#ifndef LOSS_H
#define LOSS_H

#include <cuda_runtime.h>

// MSE loss: L = (1/(2*B)) * sum((pred - target)^2)
float mse_loss(const float* d_pred, const float* d_target, int size, int batch_size);

// MSE gradient: grad[i] = (1/B) * (pred[i] - target[i])
void mse_gradient(const float* d_pred, const float* d_target, float* d_grad,
                  int size, int batch_size);

// Cross-entropy loss (assumes pred is already softmax): L = -(1/B) * sum(target * log(pred))
float cross_entropy_loss(const float* d_pred, const float* d_target, int size, int batch_size);

// Cross-entropy gradient (for softmax + cross-entropy): grad = (1/B) * (pred - target)
void cross_entropy_gradient(const float* d_pred, const float* d_target, float* d_grad,
                             int size, int batch_size);

#endif // LOSS_H
