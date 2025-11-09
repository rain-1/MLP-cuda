#include "activations.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <math.h>

// ReLU forward kernel
__global__ void relu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU backward kernel
__global__ void relu_backward_kernel(const float* grad_output, const float* input,
                                      float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// Softmax forward kernel (row-wise)
__global__ void softmax_forward_kernel(const float* input, float* output, int B, int N) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, input[row * N + i]);
    }

    sdata[tid] = max_val;
    __syncthreads();

    // Reduction to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    max_val = sdata[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = expf(input[row * N + i] - max_val);
        output[row * N + i] = val;
        sum += val;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction to find global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    sum = sdata[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        output[row * N + i] /= sum;
    }
}

// Sigmoid forward kernel
__global__ void sigmoid_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// Sigmoid backward kernel
__global__ void sigmoid_backward_kernel(const float* grad_output, const float* output,
                                         float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sig = output[idx];
        grad_input[idx] = grad_output[idx] * sig * (1.0f - sig);
    }
}

// Host functions

void relu_forward(const float* d_input, float* d_output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    relu_forward_kernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
}

void relu_backward(const float* d_grad_output, const float* d_input,
                   float* d_grad_input, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    relu_backward_kernel<<<gridSize, blockSize>>>(d_grad_output, d_input,
                                                   d_grad_input, size);
    CUDA_CHECK(cudaGetLastError());
}

void softmax_forward(const float* d_input, float* d_output, int B, int N) {
    int blockSize = 256;
    int gridSize = B;
    size_t sharedMemSize = blockSize * sizeof(float);

    softmax_forward_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, B, N);
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_forward(const float* d_input, float* d_output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    sigmoid_forward_kernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid_backward(const float* d_grad_output, const float* d_output,
                      float* d_grad_input, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    sigmoid_backward_kernel<<<gridSize, blockSize>>>(d_grad_output, d_output,
                                                     d_grad_input, size);
    CUDA_CHECK(cudaGetLastError());
}
