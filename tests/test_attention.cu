#include "multi_head_attention.h"
#include "attention_ops.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOLERANCE 1e-3f

void init_random(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

void init_identity(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = 1.0f;
    }
}

bool test_reshape_operations() {
    printf("\n=== Testing Reshape Operations ===\n");

    int B = 2, N = 4, h = 2, d = 3;
    int size_BNhd = B * N * h * d;
    int size_BhNd = B * h * N * d;

    // Host arrays
    float* h_input = new float[size_BNhd];
    float* h_output = new float[size_BhNd];
    float* h_back = new float[size_BNhd];

    // Initialize with sequential values for easy verification
    for (int i = 0; i < size_BNhd; i++) {
        h_input[i] = (float)i;
    }

    // Device arrays
    float *d_input, *d_output, *d_back;
    CUDA_CHECK(cudaMalloc(&d_input, size_BNhd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size_BhNd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_back, size_BNhd * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size_BNhd * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Forward reshape: [B, N, h*d] -> [B*h, N, d]
    reshape_BNhd_to_BhNd(d_input, d_output, B, N, h, d);

    // Backward reshape: [B*h, N, d] -> [B, N, h*d]
    reshape_BhNd_to_BNhd(d_output, d_back, B, N, h, d);

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_back, d_back, size_BNhd * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Verify round-trip
    bool passed = true;
    for (int i = 0; i < size_BNhd; i++) {
        if (fabsf(h_input[i] - h_back[i]) > TOLERANCE) {
            printf("FAILED: Mismatch at index %d: %.2f vs %.2f\n",
                   i, h_input[i], h_back[i]);
            passed = false;
            break;
        }
    }

    if (passed) {
        printf("PASSED: Reshape operations (round-trip successful)\n");
    }

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    delete[] h_back;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_back));

    return passed;
}

bool test_batched_softmax() {
    printf("\n=== Testing Batched Softmax ===\n");

    int B = 2, N = 3, M = 4;
    int size = B * N * M;

    float* h_input = new float[size];
    float* h_output = new float[size];

    // Create simple input
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(i % M);  // Pattern: 0,1,2,3,0,1,2,3,...
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float),
                         cudaMemcpyHostToDevice));

    batched_softmax(d_input, d_output, B, N, M);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Verify: each row should sum to 1.0
    bool passed = true;
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int m = 0; m < M; m++) {
                int idx = b * N * M + n * M + m;
                sum += h_output[idx];
            }
            if (fabsf(sum - 1.0f) > TOLERANCE) {
                printf("FAILED: Row [%d,%d] sum = %.6f (expected 1.0)\n", b, n, sum);
                passed = false;
            }
        }
    }

    if (passed) {
        printf("PASSED: Batched softmax (all rows sum to 1.0)\n");
    }

    delete[] h_input;
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return passed;
}

bool test_attention_forward() {
    printf("\n=== Testing Multi-Head Attention Forward ===\n");

    int d_model = 64;
    int num_heads = 4;
    int batch_size = 2;
    int seq_len = 8;
    int max_seq_len = 16;
    int max_batch_size = 4;

    MultiHeadAttention mha(d_model, num_heads, max_seq_len, max_batch_size);

    // Create input
    float* h_input = new float[batch_size * seq_len * d_model];
    float* h_output = new float[batch_size * seq_len * d_model];

    init_random(h_input, batch_size * seq_len * d_model);

    // Forward pass
    mha.forward(h_input, h_output, batch_size, seq_len);

    // Check output is not NaN or Inf
    bool passed = true;
    int non_zero_count = 0;
    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        if (isnan(h_output[i]) || isinf(h_output[i])) {
            printf("FAILED: Output contains NaN or Inf at index %d\n", i);
            passed = false;
            break;
        }
        if (fabsf(h_output[i]) > 1e-6f) {
            non_zero_count++;
        }
    }

    if (passed && non_zero_count > 0) {
        printf("PASSED: Forward pass produces valid output (%d/%d non-zero)\n",
               non_zero_count, batch_size * seq_len * d_model);
    } else if (passed) {
        printf("WARNING: All outputs are near zero\n");
    }

    delete[] h_input;
    delete[] h_output;

    return passed;
}

bool test_attention_cross() {
    printf("\n=== Testing Cross-Attention ===\n");

    int d_model = 32;
    int num_heads = 2;
    int batch_size = 2;
    int seq_len_q = 4;
    int seq_len_kv = 6;
    int max_seq_len = 16;
    int max_batch_size = 4;

    MultiHeadAttention mha(d_model, num_heads, max_seq_len, max_batch_size);

    // Create inputs
    float* h_Q = new float[batch_size * seq_len_q * d_model];
    float* h_KV = new float[batch_size * seq_len_kv * d_model];
    float* h_output = new float[batch_size * seq_len_q * d_model];

    init_random(h_Q, batch_size * seq_len_q * d_model);
    init_random(h_KV, batch_size * seq_len_kv * d_model);

    // Cross-attention forward pass
    mha.forward_cross(h_Q, h_KV, h_output, batch_size, seq_len_q, seq_len_kv);

    // Verify output shape and validity
    bool passed = true;
    for (int i = 0; i < batch_size * seq_len_q * d_model; i++) {
        if (isnan(h_output[i]) || isinf(h_output[i])) {
            printf("FAILED: Cross-attention output contains NaN or Inf\n");
            passed = false;
            break;
        }
    }

    if (passed) {
        printf("PASSED: Cross-attention forward pass\n");
    }

    delete[] h_Q;
    delete[] h_KV;
    delete[] h_output;

    return passed;
}

bool test_save_load() {
    printf("\n=== Testing Save/Load Parameters ===\n");

    int d_model = 32;
    int num_heads = 4;
    int max_seq_len = 16;
    int max_batch_size = 4;
    int batch_size = 2;
    int seq_len = 8;

    // Create first attention block
    MultiHeadAttention mha1(d_model, num_heads, max_seq_len, max_batch_size);

    // Create input and get output
    float* h_input = new float[batch_size * seq_len * d_model];
    float* h_output1 = new float[batch_size * seq_len * d_model];
    float* h_output2 = new float[batch_size * seq_len * d_model];

    init_random(h_input, batch_size * seq_len * d_model);

    mha1.forward(h_input, h_output1, batch_size, seq_len);

    // Save parameters
    mha1.save_parameters("/tmp/mha_test_params.bin");

    // Create second attention block and load parameters
    MultiHeadAttention mha2(d_model, num_heads, max_seq_len, max_batch_size);
    mha2.load_parameters("/tmp/mha_test_params.bin");

    // Run forward pass with same input
    mha2.forward(h_input, h_output2, batch_size, seq_len);

    // Compare outputs
    bool passed = true;
    float max_diff = 0.0f;
    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        float diff = fabsf(h_output1[i] - h_output2[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > TOLERANCE) {
            printf("FAILED: Output mismatch at index %d: %.6f vs %.6f\n",
                   i, h_output1[i], h_output2[i]);
            passed = false;
            break;
        }
    }

    if (passed) {
        printf("PASSED: Save/load (max diff: %.6e)\n", max_diff);
    }

    delete[] h_input;
    delete[] h_output1;
    delete[] h_output2;

    return passed;
}

bool test_different_sequence_lengths() {
    printf("\n=== Testing Different Sequence Lengths ===\n");

    int d_model = 64;
    int num_heads = 4;
    int max_seq_len = 32;
    int max_batch_size = 4;
    int batch_size = 2;

    MultiHeadAttention mha(d_model, num_heads, max_seq_len, max_batch_size);

    bool all_passed = true;

    for (int seq_len = 2; seq_len <= 16; seq_len *= 2) {
        float* h_input = new float[batch_size * seq_len * d_model];
        float* h_output = new float[batch_size * seq_len * d_model];

        init_random(h_input, batch_size * seq_len * d_model);

        mha.forward(h_input, h_output, batch_size, seq_len);

        // Check validity
        bool valid = true;
        for (int i = 0; i < batch_size * seq_len * d_model; i++) {
            if (isnan(h_output[i]) || isinf(h_output[i])) {
                valid = false;
                break;
            }
        }

        if (valid) {
            printf("  Sequence length %2d: ✓\n", seq_len);
        } else {
            printf("  Sequence length %2d: FAILED\n", seq_len);
            all_passed = false;
        }

        delete[] h_input;
        delete[] h_output;
    }

    if (all_passed) {
        printf("PASSED: All sequence lengths work correctly\n");
    }

    return all_passed;
}

int main() {
    printf("======================================\n");
    printf("  Multi-Head Attention Test Suite\n");
    printf("======================================\n");

    srand(42);  // Fixed seed for reproducibility

    int passed = 0;
    int total = 0;

    total++; if (test_reshape_operations()) passed++;
    total++; if (test_batched_softmax()) passed++;
    total++; if (test_attention_forward()) passed++;
    total++; if (test_attention_cross()) passed++;
    total++; if (test_save_load()) passed++;
    total++; if (test_different_sequence_lengths()) passed++;

    printf("\n======================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("======================================\n");

    return (passed == total) ? 0 : 1;
}
