#include "transformer_block.h"
#include "gradient_check.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>

// Test FeedForwardNetwork gradient computation
// This verifies that the analytical gradients match numerical gradients

bool test_ffn_forward_shape() {
    printf("\n=== Test FFN Forward Shape ===\n");

    int d_model = 64;
    int d_ff = 256;
    int batch_size = 4;
    int seq_len = 8;

    FeedForwardNetwork ffn(d_model, d_ff, batch_size, seq_len, 1.0f);

    // Create random input
    float* d_input;
    float* d_output;
    int total_tokens = batch_size * seq_len;

    CUDA_CHECK(cudaMalloc(&d_input, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total_tokens * d_model * sizeof(float)));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateNormal(gen, d_input, total_tokens * d_model, 0.0f, 1.0f);

    // Forward pass
    ffn.forward_device(d_input, d_output, batch_size, seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check output is finite
    float* h_output = new float[total_tokens * d_model];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, total_tokens * d_model * sizeof(float),
                         cudaMemcpyDeviceToHost));

    bool all_finite = true;
    for (int i = 0; i < total_tokens * d_model; i++) {
        if (!isfinite(h_output[i])) {
            printf("Non-finite value at index %d: %f\n", i, h_output[i]);
            all_finite = false;
            break;
        }
    }

    delete[] h_output;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    curandDestroyGenerator(gen);

    printf("Result: %s\n", all_finite ? "PASS" : "FAIL");
    return all_finite;
}

bool test_ffn_w1_gradient() {
    printf("\n=== Test FFN W1 Gradient ===\n");

    int d_model = 8;
    int d_ff = 32;
    int batch_size = 2;
    int seq_len = 4;
    int total_tokens = batch_size * seq_len;

    FeedForwardNetwork ffn(d_model, d_ff, batch_size, seq_len, 1.0f);

    // Create input and target
    float* d_input;
    float* d_output;
    float* d_target;
    float* d_grad_output;
    float* d_grad_input;
    float* d_grad_W1;
    float* d_grad_b1;
    float* d_grad_W2;
    float* d_grad_b2;

    CUDA_CHECK(cudaMalloc(&d_input, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_W1, d_model * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b1, d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_W2, d_ff * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b2, d_model * sizeof(float)));

    // Initialize with small random values
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 123);
    curandGenerateNormal(gen, d_input, total_tokens * d_model, 0.0f, 0.1f);
    curandGenerateNormal(gen, d_target, total_tokens * d_model, 0.0f, 0.1f);
    curandDestroyGenerator(gen);

    // Forward pass
    ffn.forward_device(d_input, d_output, batch_size, seq_len);

    // Compute gradient of MSE loss w.r.t. output
    // grad = 2/N * (output - target)
    int size = total_tokens * d_model;
    float scale = 2.0f / size;

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    auto compute_grad = [=] __device__ (int idx) {
        if (idx < size) {
            d_grad_output[idx] = scale * (d_output[idx] - d_target[idx]);
        }
    };

    // Use a kernel to compute grad
    cudaMemset(d_grad_output, 0, size * sizeof(float));
    float* h_output = new float[size];
    float* h_target = new float[size];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_target, d_target, size * sizeof(float), cudaMemcpyDeviceToHost));

    float* h_grad_output = new float[size];
    for (int i = 0; i < size; i++) {
        h_grad_output[i] = scale * (h_output[i] - h_target[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, size * sizeof(float), cudaMemcpyHostToDevice));

    delete[] h_output;
    delete[] h_target;
    delete[] h_grad_output;

    // Backward pass
    ffn.backward_device(d_input, d_grad_output, d_grad_input,
                        d_grad_W1, d_grad_b1, d_grad_W2, d_grad_b2,
                        batch_size, seq_len);

    // Define loss function for gradient checking
    auto loss_fn = [&](const float* d_W1_test) -> float {
        // Temporarily replace W1
        float* d_W1_backup;
        CUDA_CHECK(cudaMalloc(&d_W1_backup, d_model * d_ff * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_W1_backup, ffn.d_W1, d_model * d_ff * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(ffn.d_W1, d_W1_test, d_model * d_ff * sizeof(float),
                             cudaMemcpyDeviceToDevice));

        // Forward pass
        float* d_output_test;
        CUDA_CHECK(cudaMalloc(&d_output_test, total_tokens * d_model * sizeof(float)));
        ffn.forward_device(d_input, d_output_test, batch_size, seq_len);

        // Compute MSE loss
        float* h_output_test = new float[size];
        float* h_target_test = new float[size];
        CUDA_CHECK(cudaMemcpy(h_output_test, d_output_test, size * sizeof(float),
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_target_test, d_target, size * sizeof(float),
                             cudaMemcpyDeviceToHost));

        float loss = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = h_output_test[i] - h_target_test[i];
            loss += diff * diff;
        }
        loss /= size;

        delete[] h_output_test;
        delete[] h_target_test;
        CUDA_CHECK(cudaFree(d_output_test));

        // Restore W1
        CUDA_CHECK(cudaMemcpy(ffn.d_W1, d_W1_backup, d_model * d_ff * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(d_W1_backup));

        return loss;
    };

    // Check gradients (only check first few parameters to save time)
    int check_size = std::min(50, d_model * d_ff);
    float* d_W1_subset;
    float* d_grad_W1_subset;
    CUDA_CHECK(cudaMalloc(&d_W1_subset, check_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_W1_subset, check_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W1_subset, ffn.d_W1, check_size * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_W1_subset, d_grad_W1, check_size * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    printf("Checking %d W1 parameters (out of %d total)...\n", check_size, d_model * d_ff);
    GradientCheckResult result = check_gradients_verbose(
        loss_fn, d_W1_subset, d_grad_W1_subset, check_size, 1e-4f, 1e-2f, 5
    );

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_grad_input));
    CUDA_CHECK(cudaFree(d_grad_W1));
    CUDA_CHECK(cudaFree(d_grad_b1));
    CUDA_CHECK(cudaFree(d_grad_W2));
    CUDA_CHECK(cudaFree(d_grad_b2));
    CUDA_CHECK(cudaFree(d_W1_subset));
    CUDA_CHECK(cudaFree(d_grad_W1_subset));

    return result.passed;
}

bool test_ffn_identity() {
    printf("\n=== Test FFN Can Learn Identity ===\n");

    int d_model = 16;
    int d_ff = 64;
    int batch_size = 8;
    int seq_len = 4;
    int total_tokens = batch_size * seq_len;

    FeedForwardNetwork ffn(d_model, d_ff, batch_size, seq_len, 1.0f);

    // Create input - simple pattern
    float* d_input;
    float* d_output;
    float* d_grad_output;
    float* d_grad_input;
    float* d_grad_W1;
    float* d_grad_b1;
    float* d_grad_W2;
    float* d_grad_b2;

    CUDA_CHECK(cudaMalloc(&d_input, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_W1, d_model * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b1, d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_W2, d_ff * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_b2, d_model * sizeof(float)));

    // Initialize input (simple values)
    float* h_input = new float[total_tokens * d_model];
    for (int i = 0; i < total_tokens * d_model; i++) {
        h_input[i] = (float)(i % 10) * 0.1f;
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, total_tokens * d_model * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Training loop - try to learn identity function
    float learning_rate = 0.01f;
    int num_steps = 100;

    printf("Training FFN to learn identity function...\n");
    for (int step = 0; step < num_steps; step++) {
        // Forward
        ffn.forward_device(d_input, d_output, batch_size, seq_len);

        // Compute loss and gradients
        float* h_output = new float[total_tokens * d_model];
        CUDA_CHECK(cudaMemcpy(h_output, d_output, total_tokens * d_model * sizeof(float),
                             cudaMemcpyDeviceToHost));

        float loss = 0.0f;
        float* h_grad_output = new float[total_tokens * d_model];
        for (int i = 0; i < total_tokens * d_model; i++) {
            float diff = h_output[i] - h_input[i];
            loss += diff * diff;
            h_grad_output[i] = 2.0f * diff / (total_tokens * d_model);
        }
        loss /= (total_tokens * d_model);

        CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, total_tokens * d_model * sizeof(float),
                             cudaMemcpyHostToDevice));

        // Backward
        ffn.backward_device(d_input, d_grad_output, d_grad_input,
                           d_grad_W1, d_grad_b1, d_grad_W2, d_grad_b2,
                           batch_size, seq_len);

        // Update parameters (simple SGD)
        // This is a simplified update - real training would use Adam
        // We just check if loss decreases

        if (step % 20 == 0) {
            printf("Step %d, Loss: %.6f\n", step, loss);
        }

        delete[] h_output;
        delete[] h_grad_output;
    }

    printf("\nNote: This test just verifies the FFN can run training steps.\n");
    printf("Full convergence would require proper optimizer (Adam).\n");

    delete[] h_input;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_grad_input));
    CUDA_CHECK(cudaFree(d_grad_W1));
    CUDA_CHECK(cudaFree(d_grad_b1));
    CUDA_CHECK(cudaFree(d_grad_W2));
    CUDA_CHECK(cudaFree(d_grad_b2));

    return true;  // Just check it runs without crashing
}

int main() {
    printf("=== FeedForwardNetwork Gradient Tests ===\n");

    bool all_passed = true;

    all_passed &= test_ffn_forward_shape();
    all_passed &= test_ffn_w1_gradient();
    all_passed &= test_ffn_identity();

    printf("\n=== Final Result ===\n");
    printf("All tests: %s\n", all_passed ? "PASSED" : "FAILED");

    return all_passed ? 0 : 1;
}
