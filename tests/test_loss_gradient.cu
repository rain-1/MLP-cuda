#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/loss.h"
#include "../src/matrix_ops.h"

// CPU reference implementation for cross-entropy loss
float cpu_cross_entropy_loss(
    const float* logits,
    const int* targets,
    int batch_size,
    int seq_len,
    int vocab_size,
    const float* mask
) {
    int total_positions = batch_size * seq_len;
    float total_loss = 0.0f;
    int num_valid = 0;

    for (int idx = 0; idx < total_positions; idx++) {
        float m = (mask != nullptr) ? mask[idx] : 1.0f;

        if (m > 0.0f) {
            int target = targets[idx];
            const float* logits_ptr = logits + idx * vocab_size;

            // Compute max for numerical stability
            float max_logit = -INFINITY;
            for (int i = 0; i < vocab_size; i++) {
                max_logit = fmaxf(max_logit, logits_ptr[i]);
            }

            // Compute log-sum-exp
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                sum_exp += expf(logits_ptr[i] - max_logit);
            }
            float log_sum_exp = logf(sum_exp) + max_logit;

            // Cross-entropy loss
            total_loss += -(logits_ptr[target] - log_sum_exp);
            num_valid++;
        }
    }

    return (num_valid > 0) ? (total_loss / num_valid) : 0.0f;
}

// CPU reference implementation for cross-entropy gradient
void cpu_cross_entropy_gradient(
    const float* logits,
    const int* targets,
    float* grad,
    int batch_size,
    int seq_len,
    int vocab_size,
    const float* mask
) {
    int total_positions = batch_size * seq_len;

    // Count valid positions
    int num_valid = 0;
    for (int idx = 0; idx < total_positions; idx++) {
        float m = (mask != nullptr) ? mask[idx] : 1.0f;
        if (m > 0.0f) num_valid++;
    }

    float scale = (num_valid > 0) ? (1.0f / num_valid) : 0.0f;

    for (int idx = 0; idx < total_positions; idx++) {
        float m = (mask != nullptr) ? mask[idx] : 1.0f;
        const float* logits_ptr = logits + idx * vocab_size;
        float* grad_ptr = grad + idx * vocab_size;

        if (m > 0.0f) {
            int target = targets[idx];

            // Compute max for numerical stability
            float max_logit = -INFINITY;
            for (int i = 0; i < vocab_size; i++) {
                max_logit = fmaxf(max_logit, logits_ptr[i]);
            }

            // Compute softmax denominator
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                sum_exp += expf(logits_ptr[i] - max_logit);
            }

            // Compute gradient for each vocabulary item
            for (int v = 0; v < vocab_size; v++) {
                float softmax_v = expf(logits_ptr[v] - max_logit) / sum_exp;
                float target_indicator = (v == target) ? 1.0f : 0.0f;
                grad_ptr[v] = scale * m * (softmax_v - target_indicator);
            }
        } else {
            // Masked position - zero gradient
            for (int v = 0; v < vocab_size; v++) {
                grad_ptr[v] = 0.0f;
            }
        }
    }
}

// Finite difference gradient checker
bool check_gradient_finite_diff(
    const float* logits,
    const int* targets,
    const float* grad,
    int batch_size,
    int seq_len,
    int vocab_size,
    const float* mask,
    float epsilon = 1e-4,
    float tolerance = 1e-3
) {
    int total_positions = batch_size * seq_len;
    int total_elements = total_positions * vocab_size;

    // Allocate workspace
    float* logits_plus = new float[total_elements];
    float* logits_minus = new float[total_elements];

    bool all_passed = true;
    int num_checked = 0;
    int num_failed = 0;

    // Check a subset of gradients (checking all would be too slow)
    // Check first position fully, then sample others
    for (int idx = 0; idx < total_positions && num_checked < 500; idx++) {
        int sample_rate = (idx == 0) ? 1 : (vocab_size / 10 + 1);

        for (int v = 0; v < vocab_size; v += sample_rate) {
            // Copy logits
            memcpy(logits_plus, logits, total_elements * sizeof(float));
            memcpy(logits_minus, logits, total_elements * sizeof(float));

            // Perturb
            int elem_idx = idx * vocab_size + v;
            logits_plus[elem_idx] += epsilon;
            logits_minus[elem_idx] -= epsilon;

            // Compute losses
            float loss_plus = cpu_cross_entropy_loss(logits_plus, targets, batch_size, seq_len, vocab_size, mask);
            float loss_minus = cpu_cross_entropy_loss(logits_minus, targets, batch_size, seq_len, vocab_size, mask);

            // Finite difference gradient
            float fd_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
            float analytical_grad = grad[elem_idx];

            // Check difference
            float abs_diff = fabsf(fd_grad - analytical_grad);
            float rel_diff = abs_diff / (fabsf(fd_grad) + fabsf(analytical_grad) + 1e-8);

            if (abs_diff > tolerance && rel_diff > tolerance) {
                if (num_failed < 10) {  // Only print first 10 failures
                    printf("  FAIL: pos=%d, vocab=%d, FD=%.6f, Analytical=%.6f, AbsDiff=%.6f, RelDiff=%.6f\n",
                           idx, v, fd_grad, analytical_grad, abs_diff, rel_diff);
                }
                num_failed++;
                all_passed = false;
            }

            num_checked++;
        }
    }

    delete[] logits_plus;
    delete[] logits_minus;

    printf("  Checked %d gradients, %d failed\n", num_checked, num_failed);
    return all_passed;
}

void test_small_vocab() {
    printf("\nTest 1: Small vocabulary (vocab_size < blockDim)\n");
    printf("=================================================\n");

    int batch_size = 2;
    int seq_len = 3;
    int vocab_size = 50;
    int total_positions = batch_size * seq_len;
    int total_elements = total_positions * vocab_size;

    // Allocate host memory
    float* h_logits = new float[total_elements];
    int* h_targets = new int[total_positions];
    float* h_grad_cpu = new float[total_elements];
    float* h_grad_gpu = new float[total_elements];

    // Initialize with random data
    srand(42);
    for (int i = 0; i < total_elements; i++) {
        h_logits[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < total_positions; i++) {
        h_targets[i] = rand() % vocab_size;
    }

    // Allocate device memory
    float* d_logits;
    int* d_targets;
    float* d_grad;
    CUDA_CHECK(cudaMalloc(&d_logits, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, total_positions * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grad, total_elements * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets, total_positions * sizeof(int), cudaMemcpyHostToDevice));

    // Compute GPU gradient
    lm_cross_entropy_gradient(d_logits, d_targets, d_grad, batch_size, seq_len, vocab_size, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_grad_gpu, d_grad, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute CPU gradient
    cpu_cross_entropy_gradient(h_logits, h_targets, h_grad_cpu, batch_size, seq_len, vocab_size, nullptr);

    // Compare
    float max_diff = 0.0f;
    int num_large_diff = 0;
    for (int i = 0; i < total_elements; i++) {
        float diff = fabsf(h_grad_gpu[i] - h_grad_cpu[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > 1e-5) num_large_diff++;
    }

    printf("  Max difference: %.10f\n", max_diff);
    printf("  Elements with diff > 1e-5: %d / %d\n", num_large_diff, total_elements);

    if (max_diff < 1e-4) {
        printf("  ✓ CPU/GPU gradient match\n");
    } else {
        printf("  ✗ CPU/GPU gradient MISMATCH\n");
    }

    // Finite difference check
    printf("  Running finite difference check...\n");
    bool fd_passed = check_gradient_finite_diff(h_logits, h_targets, h_grad_cpu, batch_size, seq_len, vocab_size, nullptr);
    if (fd_passed) {
        printf("  ✓ Finite difference check passed\n");
    } else {
        printf("  ✗ Finite difference check FAILED\n");
    }

    // Cleanup
    delete[] h_logits;
    delete[] h_targets;
    delete[] h_grad_cpu;
    delete[] h_grad_gpu;
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_grad));
}

void test_large_vocab() {
    printf("\nTest 2: Large vocabulary (vocab_size > blockDim)\n");
    printf("==================================================\n");

    int batch_size = 2;
    int seq_len = 3;
    int vocab_size = 2000;
    int total_positions = batch_size * seq_len;
    int total_elements = total_positions * vocab_size;

    // Allocate host memory
    float* h_logits = new float[total_elements];
    int* h_targets = new int[total_positions];
    float* h_grad_cpu = new float[total_elements];
    float* h_grad_gpu = new float[total_elements];

    // Initialize with random data
    srand(42);
    for (int i = 0; i < total_elements; i++) {
        h_logits[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < total_positions; i++) {
        h_targets[i] = rand() % vocab_size;
    }

    // Allocate device memory
    float* d_logits;
    int* d_targets;
    float* d_grad;
    CUDA_CHECK(cudaMalloc(&d_logits, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, total_positions * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grad, total_elements * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets, total_positions * sizeof(int), cudaMemcpyHostToDevice));

    // Compute GPU gradient
    lm_cross_entropy_gradient(d_logits, d_targets, d_grad, batch_size, seq_len, vocab_size, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_grad_gpu, d_grad, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute CPU gradient
    cpu_cross_entropy_gradient(h_logits, h_targets, h_grad_cpu, batch_size, seq_len, vocab_size, nullptr);

    // Compare
    float max_diff = 0.0f;
    int num_large_diff = 0;
    for (int i = 0; i < total_elements; i++) {
        float diff = fabsf(h_grad_gpu[i] - h_grad_cpu[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > 1e-5) num_large_diff++;
    }

    printf("  Max difference: %.10f\n", max_diff);
    printf("  Elements with diff > 1e-5: %d / %d\n", num_large_diff, total_elements);

    if (max_diff < 1e-4) {
        printf("  ✓ CPU/GPU gradient match\n");
    } else {
        printf("  ✗ CPU/GPU gradient MISMATCH\n");
    }

    // Finite difference check
    printf("  Running finite difference check...\n");
    bool fd_passed = check_gradient_finite_diff(h_logits, h_targets, h_grad_cpu, batch_size, seq_len, vocab_size, nullptr);
    if (fd_passed) {
        printf("  ✓ Finite difference check passed\n");
    } else {
        printf("  ✗ Finite difference check FAILED\n");
    }

    // Cleanup
    delete[] h_logits;
    delete[] h_targets;
    delete[] h_grad_cpu;
    delete[] h_grad_gpu;
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_grad));
}

void test_with_masking() {
    printf("\nTest 3: With masking\n");
    printf("=====================\n");

    int batch_size = 2;
    int seq_len = 4;
    int vocab_size = 100;
    int total_positions = batch_size * seq_len;
    int total_elements = total_positions * vocab_size;

    // Allocate host memory
    float* h_logits = new float[total_elements];
    int* h_targets = new int[total_positions];
    float* h_mask = new float[total_positions];
    float* h_grad_cpu = new float[total_elements];
    float* h_grad_gpu = new float[total_elements];

    // Initialize with random data
    srand(42);
    for (int i = 0; i < total_elements; i++) {
        h_logits[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < total_positions; i++) {
        h_targets[i] = rand() % vocab_size;
        h_mask[i] = (i % 3 == 0) ? 0.0f : 1.0f;  // Mask every 3rd position
    }

    // Allocate device memory
    float* d_logits;
    int* d_targets;
    float* d_mask;
    float* d_grad;
    CUDA_CHECK(cudaMalloc(&d_logits, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_targets, total_positions * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mask, total_positions * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, total_elements * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets, total_positions * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask, total_positions * sizeof(float), cudaMemcpyHostToDevice));

    // Compute GPU gradient
    lm_cross_entropy_gradient(d_logits, d_targets, d_grad, batch_size, seq_len, vocab_size, d_mask);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_grad_gpu, d_grad, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute CPU gradient
    cpu_cross_entropy_gradient(h_logits, h_targets, h_grad_cpu, batch_size, seq_len, vocab_size, h_mask);

    // Compare
    float max_diff = 0.0f;
    int num_large_diff = 0;
    for (int i = 0; i < total_elements; i++) {
        float diff = fabsf(h_grad_gpu[i] - h_grad_cpu[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > 1e-5) num_large_diff++;
    }

    printf("  Max difference: %.10f\n", max_diff);
    printf("  Elements with diff > 1e-5: %d / %d\n", num_large_diff, total_elements);

    if (max_diff < 1e-4) {
        printf("  ✓ CPU/GPU gradient match\n");
    } else {
        printf("  ✗ CPU/GPU gradient MISMATCH\n");
    }

    // Finite difference check
    printf("  Running finite difference check...\n");
    bool fd_passed = check_gradient_finite_diff(h_logits, h_targets, h_grad_cpu, batch_size, seq_len, vocab_size, h_mask);
    if (fd_passed) {
        printf("  ✓ Finite difference check passed\n");
    } else {
        printf("  ✗ Finite difference check FAILED\n");
    }

    // Cleanup
    delete[] h_logits;
    delete[] h_targets;
    delete[] h_mask;
    delete[] h_grad_cpu;
    delete[] h_grad_gpu;
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_grad));
}

int main() {
    printf("Loss Gradient Kernel Validation\n");
    printf("================================\n");

    test_small_vocab();
    test_large_vocab();
    test_with_masking();

    printf("\n================================\n");
    printf("All tests completed!\n");

    return 0;
}
