#include "loss.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <math.h>

// MSE loss kernel - computes partial sums
__global__ void mse_loss_kernel(const float* pred, const float* target,
                                 float* partial_sums, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Compute local sum
    float diff = 0.0f;
    if (idx < size) {
        diff = pred[idx] - target[idx];
        diff = diff * diff;
    }
    sdata[tid] = diff;
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
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// MSE gradient kernel
__global__ void mse_gradient_kernel(const float* pred, const float* target,
                                     float* grad, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = scale * (pred[idx] - target[idx]);
    }
}

// Cross-entropy loss kernel - computes partial sums
__global__ void cross_entropy_loss_kernel(const float* pred, const float* target,
                                           float* partial_sums, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Compute local sum
    float val = 0.0f;
    if (idx < size) {
        // Avoid log(0) by clamping pred to small positive value
        float p = fmaxf(pred[idx], 1e-7f);
        val = -target[idx] * logf(p);
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
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Cross-entropy gradient kernel (for softmax + cross-entropy)
__global__ void cross_entropy_gradient_kernel(const float* pred, const float* target,
                                               float* grad, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = scale * (pred[idx] - target[idx]);
    }
}

// Host functions

float mse_loss(const float* d_pred, const float* d_target, int size, int batch_size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // Allocate device memory for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // Compute partial sums
    mse_loss_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_pred, d_target,
                                                             d_partial_sums, size);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums to host and reduce
    float* h_partial_sums = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total_loss += h_partial_sums[i];
    }

    // Cleanup
    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    return total_loss / (2.0f * batch_size);
}

void mse_gradient(const float* d_pred, const float* d_target, float* d_grad,
                  int size, int batch_size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    float scale = 1.0f / batch_size;
    mse_gradient_kernel<<<gridSize, blockSize>>>(d_pred, d_target, d_grad, size, scale);
    CUDA_CHECK(cudaGetLastError());
}

float cross_entropy_loss(const float* d_pred, const float* d_target,
                          int size, int batch_size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // Allocate device memory for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // Compute partial sums
    cross_entropy_loss_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_pred, d_target,
                                                                       d_partial_sums, size);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums to host and reduce
    float* h_partial_sums = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total_loss += h_partial_sums[i];
    }

    // Cleanup
    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    return total_loss / batch_size;
}

void cross_entropy_gradient(const float* d_pred, const float* d_target, float* d_grad,
                             int size, int batch_size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    float scale = 1.0f / batch_size;
    cross_entropy_gradient_kernel<<<gridSize, blockSize>>>(d_pred, d_target,
                                                           d_grad, size, scale);
    CUDA_CHECK(cudaGetLastError());
}
