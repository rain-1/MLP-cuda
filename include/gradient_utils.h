#ifndef GRADIENT_UTILS_H
#define GRADIENT_UTILS_H

#include <cuda_runtime.h>

// Compute L2 norm of a gradient tensor
float gradient_norm(const float* d_grad, int size);

// Clip gradients by global norm
// If ||grad|| > max_norm, scale grad by max_norm / ||grad||
void clip_gradients(float* d_grad, int size, float max_norm);

// Check for NaN or Inf in tensor
bool has_nan_or_inf(const float* d_tensor, int size);

// Compute statistics (mean, std, min, max) of a tensor
struct TensorStats {
    float mean;
    float std;
    float min;
    float max;
    bool has_nan;
    bool has_inf;
};

TensorStats compute_tensor_stats(const float* d_tensor, int size);

#endif // GRADIENT_UTILS_H
