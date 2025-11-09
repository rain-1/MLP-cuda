#ifndef GRADIENT_UTILS_H
#define GRADIENT_UTILS_H

#include <cuda_runtime.h>

// Compute L2 norm of a vector on GPU
// Returns: sqrt(sum(x_i^2))
float compute_l2_norm(const float* d_data, int size);

// Compute global gradient norm across multiple gradient tensors
// This is useful for tracking gradient health during training
struct GradientNorms {
    float total_norm;        // Global L2 norm across all gradients
    float max_grad;          // Maximum gradient value
    float mean_grad_abs;     // Mean absolute gradient value
};

// Clip gradients by global norm
// If global_norm > max_norm, scale all gradients by max_norm / global_norm
void clip_gradients_by_global_norm(
    float** d_grad_arrays,   // Array of gradient pointers
    int* sizes,              // Size of each gradient array
    int num_arrays,          // Number of gradient arrays
    float max_norm           // Maximum allowed norm
);

// Compute global gradient norm from multiple gradient tensors
float compute_global_gradient_norm(
    float** d_grad_arrays,   // Array of gradient pointers
    int* sizes,              // Size of each gradient array
    int num_arrays           // Number of gradient arrays
);

// Compute parameter norm (useful for monitoring parameter scale)
float compute_parameter_norm(const float* d_params, int size);

// Scale a gradient array by a constant factor
void scale_gradients(float* d_grad, int size, float scale);

#endif // GRADIENT_UTILS_H
