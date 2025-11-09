#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOLERANCE 1e-4f

// CPU reference implementation for matrix multiplication
void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU reference for A * B^T
void matmul_transB_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];  // B[j,k] instead of B[k,j]
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU reference for A^T * B
void matmul_transA_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    // A is [K x M], we want A^T which is [M x K]
    // Result C is [M x N]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[k * M + i] * B[k * N + j];  // A^T[i,k] = A[k,i]
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU reference for bias addition
void add_bias_cpu(const float* input, const float* bias, float* output, int B, int N) {
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {
            output[i * N + j] = input[i * N + j] + bias[j];
        }
    }
}

// CPU reference for row sum
void sum_rows_cpu(const float* input, float* output, int B, int N) {
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int i = 0; i < B; i++) {
            sum += input[i * N + j];
        }
        output[j] = sum;
    }
}

// Utility function to compare results
bool compare_results(const float* a, const float* b, int size, const char* name) {
    bool passed = true;
    float max_error = 0.0f;
    for (int i = 0; i < size; i++) {
        float error = fabsf(a[i] - b[i]);
        max_error = fmaxf(max_error, error);
        if (error > TOLERANCE) {
            if (passed) {
                printf("FAILED: %s\n", name);
                printf("  First error at index %d: expected %.6f, got %.6f (error: %.6f)\n",
                       i, a[i], b[i], error);
                passed = false;
            }
        }
    }
    if (passed) {
        printf("PASSED: %s (max error: %.6e)\n", name, max_error);
    }
    return passed;
}

// Initialize matrix with random values
void init_random(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

bool test_matmul() {
    printf("\n=== Testing Matrix Multiplication ===\n");

    int M = 64, K = 48, N = 32;
    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;

    // Allocate host memory
    float *h_A = new float[size_A];
    float *h_B = new float[size_B];
    float *h_C_cpu = new float[size_C];
    float *h_C_gpu = new float[size_C];

    // Initialize inputs
    init_random(h_A, size_A);
    init_random(h_B, size_B);

    // CPU computation
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);

    // GPU computation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));

    matmul(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    bool passed = compare_results(h_C_cpu, h_C_gpu, size_C, "matmul");

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

bool test_matmul_transB() {
    printf("\n=== Testing Matrix Multiplication (B Transposed) ===\n");

    int M = 64, K = 48, N = 32;
    int size_A = M * K;
    int size_B = N * K;  // B is [N x K] for B^T
    int size_C = M * N;

    // Allocate host memory
    float *h_A = new float[size_A];
    float *h_B = new float[size_B];
    float *h_C_cpu = new float[size_C];
    float *h_C_gpu = new float[size_C];

    // Initialize inputs
    init_random(h_A, size_A);
    init_random(h_B, size_B);

    // CPU computation
    matmul_transB_cpu(h_A, h_B, h_C_cpu, M, K, N);

    // GPU computation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));

    matmul_transB(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    bool passed = compare_results(h_C_cpu, h_C_gpu, size_C, "matmul_transB");

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

bool test_matmul_transA() {
    printf("\n=== Testing Matrix Multiplication (A Transposed) ===\n");

    int M = 64, K = 48, N = 32;
    int size_A = K * M;  // A is [K x M] for A^T
    int size_B = K * N;
    int size_C = M * N;

    // Allocate host memory
    float *h_A = new float[size_A];
    float *h_B = new float[size_B];
    float *h_C_cpu = new float[size_C];
    float *h_C_gpu = new float[size_C];

    // Initialize inputs
    init_random(h_A, size_A);
    init_random(h_B, size_B);

    // CPU computation
    matmul_transA_cpu(h_A, h_B, h_C_cpu, M, K, N);

    // GPU computation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));

    matmul_transA(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    bool passed = compare_results(h_C_cpu, h_C_gpu, size_C, "matmul_transA");

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

bool test_add_bias() {
    printf("\n=== Testing Bias Addition ===\n");

    int B = 128, N = 64;
    int size_input = B * N;
    int size_bias = N;

    // Allocate host memory
    float *h_input = new float[size_input];
    float *h_bias = new float[size_bias];
    float *h_output_cpu = new float[size_input];
    float *h_output_gpu = new float[size_input];

    // Initialize inputs
    init_random(h_input, size_input);
    init_random(h_bias, size_bias);

    // CPU computation
    add_bias_cpu(h_input, h_bias, h_output_cpu, B, N);

    // GPU computation
    float *d_input, *d_bias, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size_input * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, size_bias * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size_input * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size_input * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, size_bias * sizeof(float), cudaMemcpyHostToDevice));

    add_bias(d_input, d_bias, d_output, B, N);

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, size_input * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    bool passed = compare_results(h_output_cpu, h_output_gpu, size_input, "add_bias");

    // Cleanup
    delete[] h_input;
    delete[] h_bias;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));

    return passed;
}

bool test_sum_rows() {
    printf("\n=== Testing Row Sum ===\n");

    int B = 128, N = 64;
    int size_input = B * N;
    int size_output = N;

    // Allocate host memory
    float *h_input = new float[size_input];
    float *h_output_cpu = new float[size_output];
    float *h_output_gpu = new float[size_output];

    // Initialize inputs
    init_random(h_input, size_input);

    // CPU computation
    sum_rows_cpu(h_input, h_output_cpu, B, N);

    // GPU computation
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size_input * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size_output * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size_input * sizeof(float), cudaMemcpyHostToDevice));

    sum_rows(d_input, d_output, B, N);

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, size_output * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    bool passed = compare_results(h_output_cpu, h_output_gpu, size_output, "sum_rows");

    // Cleanup
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return passed;
}

int main() {
    printf("======================================\n");
    printf("  Matrix Operations Test Suite\n");
    printf("======================================\n");

    srand(42);  // Fixed seed for reproducibility

    int passed = 0;
    int total = 0;

    total++; if (test_matmul()) passed++;
    total++; if (test_matmul_transB()) passed++;
    total++; if (test_matmul_transA()) passed++;
    total++; if (test_add_bias()) passed++;
    total++; if (test_sum_rows()) passed++;

    printf("\n======================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("======================================\n");

    return (passed == total) ? 0 : 1;
}
