#include "loss.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Finite difference gradient checking
// Computes numerical gradient: (loss(θ+ε) - loss(θ-ε)) / (2ε)
bool check_lm_loss_gradient() {
    printf("Testing LM cross-entropy gradient with finite differences...\n");

    int batch_size = 2;
    int seq_len = 3;
    int vocab_size = 10;
    float epsilon = 1e-4f;
    float tolerance = 1e-3f;  // Relative error tolerance

    int total_logits = batch_size * seq_len * vocab_size;
    int total_positions = batch_size * seq_len;

    // Allocate host memory
    float* h_logits = new float[total_logits];
    int* h_targets = new int[total_positions];
    float* h_grad_analytical = new float[total_logits];
    float* h_grad_numerical = new float[total_logits];

    // Initialize with random logits
    srand(42);
    for (int i = 0; i < total_logits; i++) {
        h_logits[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random in [-1, 1]
    }

    // Initialize with random targets
    for (int i = 0; i < total_positions; i++) {
        h_targets[i] = rand() % vocab_size;
    }

    // Allocate device memory
    float* d_logits;
    float* d_logits_perturbed;
    int* d_targets;
    float* d_grad;

    CUDA_CHECK(cudaMalloc(&d_logits, total_logits * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_perturbed, total_logits * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, total_positions * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grad, total_logits * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, total_logits * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets, total_positions * sizeof(int), cudaMemcpyHostToDevice));

    // Compute analytical gradient
    lm_cross_entropy_gradient(d_logits, d_targets, d_grad, batch_size, seq_len, vocab_size, nullptr);
    CUDA_CHECK(cudaMemcpy(h_grad_analytical, d_grad, total_logits * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute numerical gradient using finite differences
    printf("Computing numerical gradients (this may take a moment)...\n");

    int num_samples = 50;  // Only check a subset for speed
    int max_errors = 0;
    float max_rel_error = 0.0f;

    for (int sample = 0; sample < num_samples; sample++) {
        // Pick a random logit to perturb
        int idx = rand() % total_logits;

        // Compute loss at θ+ε
        h_logits[idx] += epsilon;
        CUDA_CHECK(cudaMemcpy(d_logits_perturbed, h_logits, total_logits * sizeof(float), cudaMemcpyHostToDevice));
        float loss_plus = lm_cross_entropy_loss(d_logits_perturbed, d_targets, batch_size, seq_len, vocab_size, nullptr);

        // Compute loss at θ-ε
        h_logits[idx] -= 2.0f * epsilon;
        CUDA_CHECK(cudaMemcpy(d_logits_perturbed, h_logits, total_logits * sizeof(float), cudaMemcpyHostToDevice));
        float loss_minus = lm_cross_entropy_loss(d_logits_perturbed, d_targets, batch_size, seq_len, vocab_size, nullptr);

        // Restore original value
        h_logits[idx] += epsilon;

        // Numerical gradient
        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
        h_grad_numerical[idx] = numerical_grad;

        // Compare with analytical gradient
        float analytical_grad = h_grad_analytical[idx];
        float abs_error = fabsf(numerical_grad - analytical_grad);
        float rel_error = abs_error / (fabsf(numerical_grad) + fabsf(analytical_grad) + 1e-8f);

        if (rel_error > max_rel_error) {
            max_rel_error = rel_error;
        }

        if (rel_error > tolerance) {
            if (max_errors < 5) {  // Only print first 5 errors
                printf("  ERROR at index %d: analytical=%.6f, numerical=%.6f, rel_error=%.6f\n",
                       idx, analytical_grad, numerical_grad, rel_error);
            }
            max_errors++;
        }
    }

    // Cleanup
    delete[] h_logits;
    delete[] h_targets;
    delete[] h_grad_analytical;
    delete[] h_grad_numerical;
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_logits_perturbed));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_grad));

    printf("\nGradient check results:\n");
    printf("  Samples checked: %d\n", num_samples);
    printf("  Errors found: %d\n", max_errors);
    printf("  Max relative error: %.6f\n", max_rel_error);
    printf("  Tolerance: %.6f\n", tolerance);

    if (max_errors == 0) {
        printf("  ✓ PASS: All gradients match numerical approximation\n");
        return true;
    } else {
        printf("  ✗ FAIL: %d gradients do not match (%.1f%%)\n",
               max_errors, 100.0f * max_errors / num_samples);
        return false;
    }
}

int main() {
    printf("=== LM Loss Gradient Verification Test ===\n\n");

    bool success = check_lm_loss_gradient();

    printf("\n");
    if (success) {
        printf("=== ALL TESTS PASSED ===\n");
        return 0;
    } else {
        printf("=== TESTS FAILED ===\n");
        return 1;
    }
}
