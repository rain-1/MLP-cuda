#include "transformer.h"
#include "debug_monitor.h"
#include "matrix_ops.h"  // For CUDA_CHECK macro
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Test if transformer can copy a sequence
// Input: [1, 2, 3, 4] -> Output: [1, 2, 3, 4]
bool test_copy_sequence() {
    printf("\n=== Test Copy Sequence ===\n");
    printf("Testing if transformer can learn to copy input to output\n\n");

    // Small transformer
    int vocab_size = 10;  // Small vocabulary
    int d_model = 32;
    int num_layers = 2;
    int num_heads = 4;
    int d_ff = 128;
    int max_seq_len = 16;
    int batch_size = 4;
    int seq_len = 8;

    Transformer model(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len,
                      batch_size);

    // Create simple training data: copy task
    // Input: random sequence, Target: same sequence
    int* h_input = new int[batch_size * seq_len];
    int* h_target = new int[batch_size * seq_len];

    srand(42);
    for (int i = 0; i < batch_size * seq_len; i++) {
        h_input[i] = rand() % vocab_size;
        h_target[i] = h_input[i];  // Copy task
    }

    // Training
    float learning_rate = 0.001f;
    int num_steps = 500;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    printf("Training to learn copy task...\n");
    printf("Vocabulary size: %d, Sequence length: %d, Batch size: %d\n",
           vocab_size, seq_len, batch_size);
    printf("Model: %d layers, %d heads, d_model=%d, d_ff=%d\n\n",
           num_layers, num_heads, d_model, d_ff);

    for (int step = 0; step < num_steps; step++) {
        float loss = model.train_step(h_input, h_target, batch_size, seq_len,
                                      learning_rate, 1.0f);

        if (step == 0) {
            initial_loss = loss;
        }
        final_loss = loss;

        if (step % 100 == 0) {
            printf("Step %4d: Loss = %.4f\n", step, loss);
        }

        // Early stopping if loss is very low
        if (loss < 0.01f) {
            printf("Converged at step %d!\n", step);
            break;
        }
    }

    // Evaluate: generate and check if it matches input
    printf("\n=== Evaluation ===\n");
    printf("Initial loss: %.4f\n", initial_loss);
    printf("Final loss: %.4f\n", final_loss);

    float loss_reduction = (initial_loss - final_loss) / initial_loss;
    printf("Loss reduction: %.1f%%\n", loss_reduction * 100);

    // Test on the training data
    int* d_input;
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * seq_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_logits, batch_size * seq_len * vocab_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, batch_size * seq_len * sizeof(int),
                         cudaMemcpyHostToDevice));

    model.forward_device(d_input, d_logits, batch_size, seq_len);

    // Get predictions
    float* h_logits = new float[batch_size * seq_len * vocab_size];
    CUDA_CHECK(cudaMemcpy(h_logits, d_logits, batch_size * seq_len * vocab_size * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Check accuracy
    int correct = 0;
    int total = batch_size * seq_len;

    printf("\nSample predictions (first batch):\n");
    printf("%-10s %-10s %-10s\n", "Position", "Input", "Predicted");
    printf("------------------------------------\n");

    for (int i = 0; i < seq_len; i++) {
        int input_token = h_input[i];

        // Find argmax
        int pred_token = 0;
        float max_logit = h_logits[i * vocab_size];
        for (int v = 1; v < vocab_size; v++) {
            if (h_logits[i * vocab_size + v] > max_logit) {
                max_logit = h_logits[i * vocab_size + v];
                pred_token = v;
            }
        }

        printf("%-10d %-10d %-10d %s\n", i, input_token, pred_token,
               (input_token == pred_token) ? "✓" : "✗");

        if (input_token == pred_token) correct++;
    }

    // Check all batches
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int idx = b * seq_len + s;
            int input_token = h_input[idx];

            int pred_token = 0;
            float max_logit = h_logits[idx * vocab_size];
            for (int v = 1; v < vocab_size; v++) {
                if (h_logits[idx * vocab_size + v] > max_logit) {
                    max_logit = h_logits[idx * vocab_size + v];
                    pred_token = v;
                }
            }

            if (input_token == pred_token) correct++;
        }
    }

    float accuracy = (float)correct / total;
    printf("\nOverall accuracy: %.1f%% (%d/%d)\n", accuracy * 100, correct, total);

    // Cleanup
    delete[] h_input;
    delete[] h_target;
    delete[] h_logits;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_logits));

    // Success criteria: loss should reduce by at least 50% and accuracy > 50%
    bool success = (loss_reduction > 0.5f) && (accuracy > 0.5f);

    printf("\n%s Copy sequence test\n", success ? "✓ PASSED" : "✗ FAILED");
    return success;
}

// Test if transformer can overfit a single batch
bool test_overfit_single_batch() {
    printf("\n=== Test Overfit Single Batch ===\n");
    printf("Testing if transformer can overfit a small batch\n\n");

    // Small transformer
    int vocab_size = 50;
    int d_model = 64;
    int num_layers = 2;
    int num_heads = 4;
    int d_ff = 256;
    int max_seq_len = 32;
    int batch_size = 4;
    int seq_len = 16;

    Transformer model(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len,
                      batch_size);

    // Create a single batch of random data
    int* h_input = new int[batch_size * seq_len];
    int* h_target = new int[batch_size * seq_len];

    srand(123);
    for (int i = 0; i < batch_size * seq_len; i++) {
        h_input[i] = rand() % vocab_size;
        h_target[i] = rand() % vocab_size;
    }

    // Training with monitoring
    float learning_rate = 0.001f;
    int num_steps = 1000;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    ModelMonitor monitor;

    printf("Training to overfit single batch...\n");
    printf("Batch size: %d, Sequence length: %d\n", batch_size, seq_len);
    printf("Model: %d layers, %d heads, d_model=%d, d_ff=%d\n\n",
           num_layers, num_heads, d_model, d_ff);

    for (int step = 0; step < num_steps; step++) {
        float loss = model.train_step(h_input, h_target, batch_size, seq_len,
                                      learning_rate, 1.0f);

        if (step == 0) {
            initial_loss = loss;
        }
        final_loss = loss;

        if (step % 200 == 0) {
            printf("Step %4d: Loss = %.4f\n", step, loss);
        }

        // Check for NaN/Inf
        if (!isfinite(loss)) {
            printf("✗ Training diverged (loss = %f)\n", loss);
            delete[] h_input;
            delete[] h_target;
            return false;
        }

        // Early stopping
        if (loss < 0.1f) {
            printf("Converged at step %d!\n", step);
            break;
        }
    }

    printf("\n=== Results ===\n");
    printf("Initial loss: %.4f\n", initial_loss);
    printf("Final loss: %.4f\n", final_loss);

    float loss_reduction = (initial_loss - final_loss) / initial_loss;
    printf("Loss reduction: %.1f%%\n", loss_reduction * 100);

    // Cleanup
    delete[] h_input;
    delete[] h_target;

    // Success criteria: loss should reduce by at least 70%
    bool success = (loss_reduction > 0.7f) && isfinite(final_loss);

    printf("\n%s Overfit single batch test\n", success ? "✓ PASSED" : "✗ FAILED");
    return success;
}

// Test model's numerical stability over many steps
bool test_numerical_stability() {
    printf("\n=== Test Numerical Stability ===\n");
    printf("Testing if model remains stable over extended training\n\n");

    int vocab_size = 100;
    int d_model = 64;
    int num_layers = 4;
    int num_heads = 4;
    int d_ff = 256;
    int max_seq_len = 32;
    int batch_size = 8;
    int seq_len = 16;

    Transformer model(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len);

    // Create training data
    int* h_input = new int[batch_size * seq_len];
    int* h_target = new int[batch_size * seq_len];

    srand(456);
    for (int i = 0; i < batch_size * seq_len; i++) {
        h_input[i] = rand() % vocab_size;
        h_target[i] = rand() % vocab_size;
    }

    float learning_rate = 0.0001f;  // Small LR for stability test
    int num_steps = 200;
    bool stable = true;

    printf("Running %d training steps with small learning rate...\n", num_steps);

    for (int step = 0; step < num_steps; step++) {
        float loss = model.train_step(h_input, h_target, batch_size, seq_len,
                                      learning_rate, 1.0f);

        if (!isfinite(loss)) {
            printf("✗ Loss became non-finite at step %d: %f\n", step, loss);
            stable = false;
            break;
        }

        if (loss > 100.0f) {
            printf("✗ Loss exploded at step %d: %f\n", step, loss);
            stable = false;
            break;
        }

        if (step % 50 == 0) {
            printf("Step %4d: Loss = %.4f\n", step, loss);
        }
    }

    delete[] h_input;
    delete[] h_target;

    printf("\n%s Numerical stability test\n", stable ? "✓ PASSED" : "✗ FAILED");
    return stable;
}

int main() {
    printf("=== Toy Problem Tests ===\n");
    printf("These tests verify the model can learn simple patterns\n");

    bool all_passed = true;

    // Run tests
    all_passed &= test_copy_sequence();
    all_passed &= test_overfit_single_batch();
    all_passed &= test_numerical_stability();

    printf("\n=== Final Result ===\n");
    printf("All tests: %s\n", all_passed ? "PASSED" : "FAILED");

    return all_passed ? 0 : 1;
}
