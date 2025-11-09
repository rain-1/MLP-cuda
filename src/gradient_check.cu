#include "gradient_check.h"
#include "matrix_ops.h"
#include <stdio.h>
#include <math.h>
#include <algorithm>

GradientCheckResult check_gradients(
    std::function<float(const float*)> forward_fn,
    float* d_param,
    const float* d_grad_analytical,
    int size,
    float eps,
    float threshold
) {
    GradientCheckResult result = {0};
    result.total_params = size;

    // Copy parameters and gradients to host
    float* h_param = new float[size];
    float* h_grad_analytical = new float[size];
    float* h_grad_numerical = new float[size];

    CUDA_CHECK(cudaMemcpy(h_param, d_param, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_analytical, d_grad_analytical, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute numerical gradients
    float* h_param_perturbed = new float[size];
    float* d_param_temp;
    CUDA_CHECK(cudaMalloc(&d_param_temp, size * sizeof(float)));

    for (int i = 0; i < size; i++) {
        // Copy original parameters
        memcpy(h_param_perturbed, h_param, size * sizeof(float));

        // f(x + eps)
        h_param_perturbed[i] += eps;
        CUDA_CHECK(cudaMemcpy(d_param_temp, h_param_perturbed, size * sizeof(float), cudaMemcpyHostToDevice));
        float loss_plus = forward_fn(d_param_temp);

        // f(x - eps)
        h_param_perturbed[i] = h_param[i] - eps;
        CUDA_CHECK(cudaMemcpy(d_param_temp, h_param_perturbed, size * sizeof(float), cudaMemcpyHostToDevice));
        float loss_minus = forward_fn(d_param_temp);

        // Numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps)
        h_grad_numerical[i] = (loss_plus - loss_minus) / (2.0f * eps);
    }

    // Compare gradients
    float sum_abs_error = 0.0f;
    float sum_rel_error = 0.0f;

    for (int i = 0; i < size; i++) {
        float abs_error = fabsf(h_grad_analytical[i] - h_grad_numerical[i]);
        float denominator = fmaxf(fabsf(h_grad_analytical[i]), fabsf(h_grad_numerical[i]));
        float rel_error = denominator > 1e-7f ? abs_error / denominator : 0.0f;

        sum_abs_error += abs_error;
        sum_rel_error += rel_error;

        result.max_abs_error = fmaxf(result.max_abs_error, abs_error);
        result.max_rel_error = fmaxf(result.max_rel_error, rel_error);

        if (rel_error > threshold && abs_error > 1e-5f) {
            result.num_errors++;
        }
    }

    result.avg_abs_error = sum_abs_error / size;
    result.avg_rel_error = sum_rel_error / size;
    result.passed = (result.max_rel_error < threshold) || (result.max_abs_error < 1e-5f);

    // Cleanup
    delete[] h_param;
    delete[] h_grad_analytical;
    delete[] h_grad_numerical;
    delete[] h_param_perturbed;
    CUDA_CHECK(cudaFree(d_param_temp));

    return result;
}

GradientCheckResult check_gradients_verbose(
    std::function<float(const float*)> forward_fn,
    float* d_param,
    const float* d_grad_analytical,
    int size,
    float eps,
    float threshold,
    int max_print
) {
    GradientCheckResult result = {0};
    result.total_params = size;

    // Copy parameters and gradients to host
    float* h_param = new float[size];
    float* h_grad_analytical = new float[size];
    float* h_grad_numerical = new float[size];

    CUDA_CHECK(cudaMemcpy(h_param, d_param, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_analytical, d_grad_analytical, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute numerical gradients
    float* h_param_perturbed = new float[size];
    float* d_param_temp;
    CUDA_CHECK(cudaMalloc(&d_param_temp, size * sizeof(float)));

    printf("Computing numerical gradients for %d parameters...\n", size);

    for (int i = 0; i < size; i++) {
        // Copy original parameters
        memcpy(h_param_perturbed, h_param, size * sizeof(float));

        // f(x + eps)
        h_param_perturbed[i] += eps;
        CUDA_CHECK(cudaMemcpy(d_param_temp, h_param_perturbed, size * sizeof(float), cudaMemcpyHostToDevice));
        float loss_plus = forward_fn(d_param_temp);

        // f(x - eps)
        h_param_perturbed[i] = h_param[i] - eps;
        CUDA_CHECK(cudaMemcpy(d_param_temp, h_param_perturbed, size * sizeof(float), cudaMemcpyHostToDevice));
        float loss_minus = forward_fn(d_param_temp);

        // Numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps)
        h_grad_numerical[i] = (loss_plus - loss_minus) / (2.0f * eps);
    }

    // Compare gradients
    printf("\n%-8s %-15s %-15s %-15s %-15s\n", "Index", "Analytical", "Numerical", "Abs Error", "Rel Error");
    printf("--------------------------------------------------------------------------------\n");

    float sum_abs_error = 0.0f;
    float sum_rel_error = 0.0f;
    int printed = 0;

    for (int i = 0; i < size; i++) {
        float abs_error = fabsf(h_grad_analytical[i] - h_grad_numerical[i]);
        float denominator = fmaxf(fabsf(h_grad_analytical[i]), fabsf(h_grad_numerical[i]));
        float rel_error = denominator > 1e-7f ? abs_error / denominator : 0.0f;

        sum_abs_error += abs_error;
        sum_rel_error += rel_error;

        result.max_abs_error = fmaxf(result.max_abs_error, abs_error);
        result.max_rel_error = fmaxf(result.max_rel_error, rel_error);

        if (rel_error > threshold && abs_error > 1e-5f) {
            result.num_errors++;
            if (printed < max_print) {
                printf("%-8d %-15.6e %-15.6e %-15.6e %-15.6e [FAIL]\n",
                       i, h_grad_analytical[i], h_grad_numerical[i], abs_error, rel_error);
                printed++;
            }
        }
    }

    result.avg_abs_error = sum_abs_error / size;
    result.avg_rel_error = sum_rel_error / size;
    result.passed = (result.max_rel_error < threshold) || (result.max_abs_error < 1e-5f);

    printf("\n=== Gradient Check Summary ===\n");
    printf("Total parameters: %d\n", size);
    printf("Failed checks: %d\n", result.num_errors);
    printf("Max absolute error: %.6e\n", result.max_abs_error);
    printf("Max relative error: %.6e\n", result.max_rel_error);
    printf("Avg absolute error: %.6e\n", result.avg_abs_error);
    printf("Avg relative error: %.6e\n", result.avg_rel_error);
    printf("Status: %s\n", result.passed ? "PASS" : "FAIL");

    // Cleanup
    delete[] h_param;
    delete[] h_grad_analytical;
    delete[] h_grad_numerical;
    delete[] h_param_perturbed;
    CUDA_CHECK(cudaFree(d_param_temp));

    return result;
}

bool tensors_close(
    const float* d_a,
    const float* d_b,
    int size,
    float rtol,
    float atol
) {
    float* h_a = new float[size];
    float* h_b = new float[size];

    CUDA_CHECK(cudaMemcpy(h_a, d_a, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b, d_b, size * sizeof(float), cudaMemcpyDeviceToHost));

    bool close = true;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(h_a[i] - h_b[i]);
        float threshold = atol + rtol * fabsf(h_b[i]);
        if (diff > threshold) {
            close = false;
            break;
        }
    }

    delete[] h_a;
    delete[] h_b;

    return close;
}

void compare_tensors(
    const float* d_a,
    const float* d_b,
    int size,
    const char* name_a,
    const char* name_b
) {
    float* h_a = new float[size];
    float* h_b = new float[size];

    CUDA_CHECK(cudaMemcpy(h_a, d_a, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b, d_b, size * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n=== Tensor Comparison: %s vs %s ===\n", name_a, name_b);
    printf("%-8s %-15s %-15s %-15s %-15s\n", "Index", name_a, name_b, "Abs Diff", "Rel Diff");
    printf("--------------------------------------------------------------------------------\n");

    int max_print = 20;
    int printed = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;

    for (int i = 0; i < size && printed < max_print; i++) {
        float abs_diff = fabsf(h_a[i] - h_b[i]);
        float denominator = fmaxf(fabsf(h_a[i]), fabsf(h_b[i]));
        float rel_diff = denominator > 1e-7f ? abs_diff / denominator : 0.0f;

        max_abs_diff = fmaxf(max_abs_diff, abs_diff);
        max_rel_diff = fmaxf(max_rel_diff, rel_diff);

        if (abs_diff > 1e-3f || rel_diff > 1e-2f || printed < 10) {
            printf("%-8d %-15.6e %-15.6e %-15.6e %-15.6e\n",
                   i, h_a[i], h_b[i], abs_diff, rel_diff);
            printed++;
        }
    }

    printf("\nMax absolute difference: %.6e\n", max_abs_diff);
    printf("Max relative difference: %.6e\n", max_rel_diff);

    delete[] h_a;
    delete[] h_b;
}
