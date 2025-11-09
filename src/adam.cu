#include "adam.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Adam update kernel
__global__ void adam_update_kernel(float* param, const float* grad, float* m, float* v,
                                    float lr, float beta1, float beta2, float epsilon,
                                    float beta1_corr, float beta2_corr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];

        // Update biased second moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];

        // Compute bias-corrected moment estimates
        float m_hat = m[idx] / beta1_corr;
        float v_hat = v[idx] / beta2_corr;

        // Update parameter
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// Host function
void adam_update(float* d_param, const float* d_grad, float* d_m, float* d_v,
                 float lr, float beta1, float beta2, float epsilon,
                 float beta1_t, float beta2_t, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Compute bias correction terms: 1 - beta^t
    float beta1_corr = 1.0f - beta1_t;
    float beta2_corr = 1.0f - beta2_t;

    adam_update_kernel<<<gridSize, blockSize>>>(d_param, d_grad, d_m, d_v,
                                                 lr, beta1, beta2, epsilon,
                                                 beta1_corr, beta2_corr, size);
    CUDA_CHECK(cudaGetLastError());
}
