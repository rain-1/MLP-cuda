#include "gradient_utils.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdio.h>

// Kernel to compute squared sum for L2 norm
__global__ void compute_squared_sum_kernel(
    const float* d_data,
    float* d_partial_sums,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data and compute square
    float val = 0.0f;
    if (idx < size) {
        float x = d_data[idx];
        val = x * x;
    }
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel to scale gradients
__global__ void scale_gradients_kernel(
    float* d_grad,
    int size,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_grad[idx] *= scale;
    }
}

// Compute L2 norm of a vector on GPU
float compute_l2_norm(const float* d_data, int size) {
    if (size == 0) return 0.0f;

    // Setup kernel launch parameters
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Allocate memory for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(float)));

    // First reduction: compute partial sums
    compute_squared_sum_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        d_data, d_partial_sums, size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums to host and complete reduction on CPU
    float* h_partial_sums = new float[num_blocks];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums,
                         num_blocks * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float total_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_sum += h_partial_sums[i];
    }

    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    return sqrtf(total_sum);
}

// Compute global gradient norm from multiple gradient tensors
float compute_global_gradient_norm(
    float** d_grad_arrays,
    int* sizes,
    int num_arrays
) {
    float total_squared_sum = 0.0f;

    // Compute squared norm for each gradient array
    for (int i = 0; i < num_arrays; i++) {
        if (sizes[i] > 0 && d_grad_arrays[i] != nullptr) {
            float norm = compute_l2_norm(d_grad_arrays[i], sizes[i]);
            total_squared_sum += norm * norm;
        }
    }

    return sqrtf(total_squared_sum);
}

// Scale gradients
void scale_gradients(float* d_grad, int size, float scale) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scale_gradients_kernel<<<num_blocks, block_size>>>(d_grad, size, scale);
    CUDA_CHECK(cudaGetLastError());
}

// Clip gradients by global norm
void clip_gradients_by_global_norm(
    float** d_grad_arrays,
    int* sizes,
    int num_arrays,
    float max_norm
) {
    // Compute global norm
    float global_norm = compute_global_gradient_norm(d_grad_arrays, sizes, num_arrays);

    // If norm exceeds threshold, scale all gradients
    if (global_norm > max_norm) {
        float scale = max_norm / global_norm;

        for (int i = 0; i < num_arrays; i++) {
            if (sizes[i] > 0 && d_grad_arrays[i] != nullptr) {
                scale_gradients(d_grad_arrays[i], sizes[i], scale);
            }
        }
    }
}

// Compute parameter norm
float compute_parameter_norm(const float* d_params, int size) {
    return compute_l2_norm(d_params, size);
}
