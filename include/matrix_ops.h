#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <cuda_runtime.h>

// Matrix multiplication: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
void matmul(const float* d_A, const float* d_B, float* d_C,
            int M, int K, int N);

// Matrix multiplication with transpose: C = A * B^T
// A: [M x K], B: [N x K], C: [M x N]
void matmul_transB(const float* d_A, const float* d_B, float* d_C,
                   int M, int K, int N);

// Matrix multiplication with transpose: C = A^T * B
// A: [K x M], B: [K x N], C: [M x N]
void matmul_transA(const float* d_A, const float* d_B, float* d_C,
                   int M, int K, int N);

// Add bias to each row: output[i,j] = input[i,j] + bias[j]
// input: [B x N], bias: [N], output: [B x N]
void add_bias(const float* d_input, const float* d_bias, float* d_output,
              int B, int N);

// Sum across rows (for bias gradient): output[j] = sum_i input[i,j]
// input: [B x N], output: [N]
void sum_rows(const float* d_input, float* d_output, int B, int N);

// Element-wise operations
void elementwise_multiply(const float* d_A, const float* d_B, float* d_C, int size);
void scale_matrix(float* d_A, float scale, int size);

// Backward passes for training

// Backward for matmul_transB: C = A * B^T
// Given grad_C [M x N], compute grad_A [M x K] and grad_B [N x K]
// grad_A = grad_C * B
// grad_B = grad_C^T * A
void matmul_transB_backward(
    const float* d_grad_C,  // [M x N]
    const float* d_A,       // [M x K]
    const float* d_B,       // [N x K]
    float* d_grad_A,        // [M x K] - can be nullptr if not needed
    float* d_grad_B,        // [N x K] - can be nullptr if not needed
    int M, int K, int N
);

// Backward for matmul: C = A * B
// Given grad_C [M x N], compute grad_A [M x K] and grad_B [K x N]
// grad_A = grad_C * B^T
// grad_B = A^T * grad_C
void matmul_backward(
    const float* d_grad_C,  // [M x N]
    const float* d_A,       // [M x K]
    const float* d_B,       // [K x N]
    float* d_grad_A,        // [M x K] - can be nullptr if not needed
    float* d_grad_B,        // [K x N] - can be nullptr if not needed
    int M, int K, int N
);

// Utility functions for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif // MATRIX_OPS_H
