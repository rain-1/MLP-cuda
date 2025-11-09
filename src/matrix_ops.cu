#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

// Tiled matrix multiplication kernel: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Tiled matrix multiplication with B transposed: C = A * B^T
// A: [M x K], B: [N x K], C: [M x N]
__global__ void matmul_transB_kernel(const float* A, const float* B, float* C,
                                      int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B^T into shared memory (reading from B with swapped indices)
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[col * K + t * TILE_SIZE + threadIdx.y];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Tiled matrix multiplication with A transposed: C = A^T * B
// A: [K x M], B: [K x N], C: [M x N]
__global__ void matmul_transA_kernel(const float* A, const float* B, float* C,
                                      int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A^T into shared memory (reading from A with swapped indices)
        // A is [K x M], we want A^T which is [M x K]
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[(t * TILE_SIZE + threadIdx.x) * M + row];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Add bias to each row of matrix
__global__ void add_bias_kernel(const float* input, const float* bias,
                                 float* output, int B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < B && col < N) {
        output[row * N + col] = input[row * N + col] + bias[col];
    }
}

// Sum across rows using reduction in shared memory
__global__ void sum_rows_kernel(const float* input, float* output, int B, int N) {
    extern __shared__ float sdata[];

    int col = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Each thread sums a subset of rows
    float sum = 0.0f;
    for (int row = tid; row < B; row += stride) {
        sum += input[row * N + col];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[col] = sdata[0];
    }
}

// Element-wise multiplication
__global__ void elementwise_multiply_kernel(const float* A, const float* B,
                                             float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

// Scale matrix by constant
__global__ void scale_matrix_kernel(float* A, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] *= scale;
    }
}

// Host functions

void matmul(const float* d_A, const float* d_B, float* d_C,
            int M, int K, int N) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

void matmul_transB(const float* d_A, const float* d_B, float* d_C,
                   int M, int K, int N) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_transB_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

void matmul_transA(const float* d_A, const float* d_B, float* d_C,
                   int M, int K, int N) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_transA_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
}

void add_bias(const float* d_input, const float* d_bias, float* d_output,
              int B, int N) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (B + 15) / 16);

    add_bias_kernel<<<gridDim, blockDim>>>(d_input, d_bias, d_output, B, N);
    CUDA_CHECK(cudaGetLastError());
}

void sum_rows(const float* d_input, float* d_output, int B, int N) {
    int blockSize = 256;
    int gridSize = N;
    size_t sharedMemSize = blockSize * sizeof(float);

    sum_rows_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, B, N);
    CUDA_CHECK(cudaGetLastError());
}

void elementwise_multiply(const float* d_A, const float* d_B, float* d_C, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    elementwise_multiply_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
    CUDA_CHECK(cudaGetLastError());
}

void scale_matrix(float* d_A, float scale, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    scale_matrix_kernel<<<gridSize, blockSize>>>(d_A, scale, size);
    CUDA_CHECK(cudaGetLastError());
}

// Backward passes

void matmul_transB_backward(
    const float* d_grad_C,
    const float* d_A,
    const float* d_B,
    float* d_grad_A,
    float* d_grad_B,
    int M, int K, int N
) {
    // C = A * B^T where A is [M x K], B is [N x K], C is [M x N]
    // grad_A = grad_C * B  (shape: [M x N] * [N x K] = [M x K])
    // grad_B = grad_C^T * A (shape: [N x M] * [M x K] = [N x K])

    if (d_grad_A != nullptr) {
        // grad_A = grad_C * B
        matmul(d_grad_C, d_B, d_grad_A, M, N, K);
    }

    if (d_grad_B != nullptr) {
        // grad_B = grad_C^T * A
        matmul_transA(d_grad_C, d_A, d_grad_B, N, M, K);
    }
}

void matmul_backward(
    const float* d_grad_C,
    const float* d_A,
    const float* d_B,
    float* d_grad_A,
    float* d_grad_B,
    int M, int K, int N
) {
    // C = A * B where A is [M x K], B is [K x N], C is [M x N]
    // grad_A = grad_C * B^T (shape: [M x N] * [N x K] = [M x K])
    // grad_B = A^T * grad_C (shape: [K x M] * [M x N] = [K x N])

    if (d_grad_A != nullptr) {
        // grad_A = grad_C * B^T
        matmul_transB(d_grad_C, d_B, d_grad_A, M, N, K);
    }

    if (d_grad_B != nullptr) {
        // grad_B = A^T * grad_C
        matmul_transA(d_A, d_grad_C, d_grad_B, K, M, N);
    }
}
