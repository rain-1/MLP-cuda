#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Generate synthetic regression data: y = sin(x1) + cos(x2)
void generate_data(float* X, float* Y, int num_samples, int input_dim, int output_dim) {
    for (int i = 0; i < num_samples; i++) {
        // Random inputs in [-π, π]
        for (int j = 0; j < input_dim; j++) {
            X[i * input_dim + j] = ((float)rand() / RAND_MAX) * 2.0f * M_PI - M_PI;
        }

        // Output: y = sin(x1) + cos(x2) + noise
        float x1 = X[i * input_dim + 0];
        float x2 = X[i * input_dim + 1];
        Y[i] = sinf(x1) + cosf(x2);

        // Add small noise
        Y[i] += ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }
}

int main() {
    printf("======================================\n");
    printf("  MLP CUDA - Regression Example\n");
    printf("======================================\n");

    // Network architecture
    int layer_sizes[4] = {2, 32, 16, 1};  // Input: 2, Hidden: 32, 16, Output: 1
    int batch_size = 128;
    int num_epochs = 500;
    int num_train = 1024;
    int num_test = 256;

    printf("\nNetwork Architecture:\n");
    printf("  Input:    %d neurons\n", layer_sizes[0]);
    printf("  Hidden 1: %d neurons\n", layer_sizes[1]);
    printf("  Hidden 2: %d neurons\n", layer_sizes[2]);
    printf("  Output:   %d neurons\n", layer_sizes[3]);
    printf("\n");

    // Create MLP
    printf("Initializing MLP...\n");
    MLP mlp(layer_sizes, batch_size, 0.001f);  // lr = 0.001

    // Generate training data
    printf("Generating training data (%d samples)...\n", num_train);
    float *train_X = new float[num_train * layer_sizes[0]];
    float *train_Y = new float[num_train * layer_sizes[3]];
    generate_data(train_X, train_Y, num_train, layer_sizes[0], layer_sizes[3]);

    // Generate test data
    printf("Generating test data (%d samples)...\n", num_test);
    float *test_X = new float[num_test * layer_sizes[0]];
    float *test_Y = new float[num_test * layer_sizes[3]];
    generate_data(test_X, test_Y, num_test, layer_sizes[0], layer_sizes[3]);

    printf("\n");

    // Initial evaluation
    float initial_train_loss = mlp.evaluate(train_X, train_Y, num_train);
    float initial_test_loss = mlp.evaluate(test_X, test_Y, num_test);
    printf("Initial - Train Loss: %.6f, Test Loss: %.6f\n",
           initial_train_loss, initial_test_loss);

    printf("\nTraining for %d epochs...\n", num_epochs);
    printf("Epoch | Train Loss | Test Loss\n");
    printf("------|------------|----------\n");

    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Train on all batches
        float epoch_loss = 0.0f;
        int num_batches = (num_train + batch_size - 1) / batch_size;

        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int current_batch_size = (start_idx + batch_size <= num_train) ?
                                      batch_size : (num_train - start_idx);

            float loss = mlp.train_step(train_X + start_idx * layer_sizes[0],
                                       train_Y + start_idx * layer_sizes[3],
                                       current_batch_size);
            epoch_loss += loss * current_batch_size;
        }
        epoch_loss /= num_train;

        // Evaluate on test set every 50 epochs
        if ((epoch + 1) % 50 == 0 || epoch == 0) {
            float test_loss = mlp.evaluate(test_X, test_Y, num_test);
            printf("%5d | %.6f | %.6f\n", epoch + 1, epoch_loss, test_loss);
        }
    }

    // Final evaluation
    printf("\n");
    float final_train_loss = mlp.evaluate(train_X, train_Y, num_train);
    float final_test_loss = mlp.evaluate(test_X, test_Y, num_test);
    printf("Final - Train Loss: %.6f, Test Loss: %.6f\n",
           final_train_loss, final_test_loss);

    // Test on a few examples
    printf("\n======================================\n");
    printf("Sample Predictions:\n");
    printf("======================================\n");
    printf("  x1     |   x2    | Predicted | Target  | Error\n");
    printf("---------|---------|-----------|---------|-------\n");

    float sample_X[10];
    float sample_Y[5];
    float predictions[5];

    for (int i = 0; i < 5; i++) {
        sample_X[i * 2 + 0] = ((float)rand() / RAND_MAX) * 2.0f * M_PI - M_PI;
        sample_X[i * 2 + 1] = ((float)rand() / RAND_MAX) * 2.0f * M_PI - M_PI;

        float x1 = sample_X[i * 2 + 0];
        float x2 = sample_X[i * 2 + 1];
        sample_Y[i] = sinf(x1) + cosf(x2);
    }

    mlp.forward(sample_X, predictions, 5);

    for (int i = 0; i < 5; i++) {
        float x1 = sample_X[i * 2 + 0];
        float x2 = sample_X[i * 2 + 1];
        float error = fabsf(predictions[i] - sample_Y[i]);

        printf("%8.4f | %8.4f | %9.4f | %7.4f | %.4f\n",
               x1, x2, predictions[i], sample_Y[i], error);
    }

    // Save model
    printf("\nSaving model to 'mlp_regression.bin'...\n");
    mlp.save_parameters("mlp_regression.bin");

    // Cleanup
    delete[] train_X;
    delete[] train_Y;
    delete[] test_X;
    delete[] test_Y;

    printf("\nTraining complete!\n");
    printf("======================================\n");

    return 0;
}
