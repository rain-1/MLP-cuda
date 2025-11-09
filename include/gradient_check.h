#ifndef GRADIENT_CHECK_H
#define GRADIENT_CHECK_H

#include <cuda_runtime.h>
#include <functional>

// Numerical gradient checking utilities
// These verify that analytical gradients match numerical gradients

struct GradientCheckResult {
    float max_abs_error;
    float max_rel_error;
    float avg_abs_error;
    float avg_rel_error;
    bool passed;
    int num_errors;
    int total_params;
};

// Check gradients for a single parameter tensor
// forward_fn: computes loss given parameter values
// d_param: parameter tensor on device
// d_grad_analytical: analytical gradient on device
// size: number of parameters
// eps: finite difference epsilon
// threshold: relative error threshold for pass/fail
GradientCheckResult check_gradients(
    std::function<float(const float*)> forward_fn,
    float* d_param,
    const float* d_grad_analytical,
    int size,
    float eps = 1e-4f,
    float threshold = 1e-3f
);

// Detailed version that prints mismatches
GradientCheckResult check_gradients_verbose(
    std::function<float(const float*)> forward_fn,
    float* d_param,
    const float* d_grad_analytical,
    int size,
    float eps = 1e-4f,
    float threshold = 1e-3f,
    int max_print = 10
);

// Helper to check if two tensors are approximately equal
bool tensors_close(
    const float* d_a,
    const float* d_b,
    int size,
    float rtol = 1e-3f,
    float atol = 1e-5f
);

// Print detailed comparison of two tensors
void compare_tensors(
    const float* d_a,
    const float* d_b,
    int size,
    const char* name_a = "expected",
    const char* name_b = "actual"
);

#endif // GRADIENT_CHECK_H
