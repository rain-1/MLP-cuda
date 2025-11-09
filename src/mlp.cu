#include "mlp.h"
#include "matrix_ops.h"
#include "activations.h"
#include "loss.h"
#include "adam.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <fstream>

MLP::MLP(int layer_sizes[4], int batch_size,
         float learning_rate, float beta1, float beta2, float epsilon)
    : h1(layer_sizes[0]), h2(layer_sizes[1]), h3(layer_sizes[2]), h4(layer_sizes[3]),
      max_batch_size(batch_size),
      lr(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon),
      step_count(0) {

    allocate_memory();
    initialize_parameters();
}

MLP::~MLP() {
    free_memory();
}

void MLP::allocate_memory() {
    // Layer 1: input h1 -> h2
    CUDA_CHECK(cudaMalloc(&d_W1, h2 * h1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z1, max_batch_size * h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A1, max_batch_size * h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW1, h2 * h1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dZ1, max_batch_size * h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dA1, max_batch_size * h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_W1, h2 * h1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_W1, h2 * h1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_b1, h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_b1, h2 * sizeof(float)));

    // Layer 2: h2 -> h3
    CUDA_CHECK(cudaMalloc(&d_W2, h3 * h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z2, max_batch_size * h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A2, max_batch_size * h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW2, h3 * h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dZ2, max_batch_size * h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dA2, max_batch_size * h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_W2, h3 * h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_W2, h3 * h2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_b2, h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_b2, h3 * sizeof(float)));

    // Layer 3: h3 -> h4 (output)
    CUDA_CHECK(cudaMalloc(&d_W3, h4 * h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, h4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z3, max_batch_size * h4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW3, h4 * h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db3, h4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dZ3, max_batch_size * h4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_W3, h4 * h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_W3, h4 * h3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_b3, h4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_b3, h4 * sizeof(float)));

    // Input/output buffers
    CUDA_CHECK(cudaMalloc(&d_X, max_batch_size * h1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y, max_batch_size * h4 * sizeof(float)));

    // Temporary buffers (for matrix multiplications)
    int max_layer_size = h1;
    if (h2 > max_layer_size) max_layer_size = h2;
    if (h3 > max_layer_size) max_layer_size = h3;
    if (h4 > max_layer_size) max_layer_size = h4;
    int max_temp_size = max_batch_size * max_layer_size;
    CUDA_CHECK(cudaMalloc(&d_temp1, max_temp_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp2, max_temp_size * sizeof(float)));

    // Initialize Adam moments to zero
    CUDA_CHECK(cudaMemset(d_m_W1, 0, h2 * h1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_W1, 0, h2 * h1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_b1, 0, h2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b1, 0, h2 * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_m_W2, 0, h3 * h2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_W2, 0, h3 * h2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_b2, 0, h3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b2, 0, h3 * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_m_W3, 0, h4 * h3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_W3, 0, h4 * h3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_b3, 0, h4 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_b3, 0, h4 * sizeof(float)));
}

void MLP::free_memory() {
    // Layer 1
    cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_Z1); cudaFree(d_A1);
    cudaFree(d_dW1); cudaFree(d_db1);
    cudaFree(d_dZ1); cudaFree(d_dA1);
    cudaFree(d_m_W1); cudaFree(d_v_W1);
    cudaFree(d_m_b1); cudaFree(d_v_b1);

    // Layer 2
    cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_Z2); cudaFree(d_A2);
    cudaFree(d_dW2); cudaFree(d_db2);
    cudaFree(d_dZ2); cudaFree(d_dA2);
    cudaFree(d_m_W2); cudaFree(d_v_W2);
    cudaFree(d_m_b2); cudaFree(d_v_b2);

    // Layer 3
    cudaFree(d_W3); cudaFree(d_b3);
    cudaFree(d_Z3);
    cudaFree(d_dW3); cudaFree(d_db3);
    cudaFree(d_dZ3);
    cudaFree(d_m_W3); cudaFree(d_v_W3);
    cudaFree(d_m_b3); cudaFree(d_v_b3);

    // Buffers
    cudaFree(d_X); cudaFree(d_Y);
    cudaFree(d_temp1); cudaFree(d_temp2);
}

void MLP::initialize_parameters() {
    // Use Xavier initialization: W ~ N(0, 2/(n_in + n_out))
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Layer 1
    float std1 = sqrtf(2.0f / (h1 + h2));
    curandGenerateNormal(gen, d_W1, h2 * h1, 0.0f, std1);
    CUDA_CHECK(cudaMemset(d_b1, 0, h2 * sizeof(float)));

    // Layer 2
    float std2 = sqrtf(2.0f / (h2 + h3));
    curandGenerateNormal(gen, d_W2, h3 * h2, 0.0f, std2);
    CUDA_CHECK(cudaMemset(d_b2, 0, h3 * sizeof(float)));

    // Layer 3
    float std3 = sqrtf(2.0f / (h3 + h4));
    curandGenerateNormal(gen, d_W3, h4 * h3, 0.0f, std3);
    CUDA_CHECK(cudaMemset(d_b3, 0, h4 * sizeof(float)));

    curandDestroyGenerator(gen);
}

void MLP::forward_device(int batch_size) {
    // Layer 1: Z1 = X * W1^T + b1, A1 = ReLU(Z1)
    matmul_transB(d_X, d_W1, d_temp1, batch_size, h1, h2);
    add_bias(d_temp1, d_b1, d_Z1, batch_size, h2);
    relu_forward(d_Z1, d_A1, batch_size * h2);

    // Layer 2: Z2 = A1 * W2^T + b2, A2 = ReLU(Z2)
    matmul_transB(d_A1, d_W2, d_temp1, batch_size, h2, h3);
    add_bias(d_temp1, d_b2, d_Z2, batch_size, h3);
    relu_forward(d_Z2, d_A2, batch_size * h3);

    // Layer 3: Z3 = A2 * W3^T + b3 (no activation)
    matmul_transB(d_A2, d_W3, d_temp1, batch_size, h3, h4);
    add_bias(d_temp1, d_b3, d_Z3, batch_size, h4);
}

void MLP::backward_device(int batch_size) {
    // Output layer gradient: dZ3 = (1/B) * (Z3 - Y)
    mse_gradient(d_Z3, d_Y, d_dZ3, batch_size * h4, batch_size);

    // Layer 3 parameter gradients
    // dW3 = dZ3^T @ A2: [h4 x h3] = [h4 x B] @ [B x h3]
    // dZ3 is stored as [B x h4], A2 is [B x h3]
    matmul_transA(d_dZ3, d_A2, d_dW3, h4, batch_size, h3);

    // db3 = sum over batch of dZ3
    sum_rows(d_dZ3, d_db3, batch_size, h4);

    // Backpropagate to layer 2: dA2 = dZ3 @ W3
    matmul(d_dZ3, d_W3, d_dA2, batch_size, h4, h3);

    // Layer 2 gradient through ReLU: dZ2 = dA2 ⊙ (Z2 > 0)
    relu_backward(d_dA2, d_Z2, d_dZ2, batch_size * h3);

    // Layer 2 parameter gradients
    // dW2 = dZ2^T @ A1: [h3 x h2] = [h3 x B] @ [B x h2]
    matmul_transA(d_dZ2, d_A1, d_dW2, h3, batch_size, h2);

    // db2 = sum over batch of dZ2
    sum_rows(d_dZ2, d_db2, batch_size, h3);

    // Backpropagate to layer 1: dA1 = dZ2 @ W2
    matmul(d_dZ2, d_W2, d_dA1, batch_size, h3, h2);

    // Layer 1 gradient through ReLU: dZ1 = dA1 ⊙ (Z1 > 0)
    relu_backward(d_dA1, d_Z1, d_dZ1, batch_size * h2);

    // Layer 1 parameter gradients
    // dW1 = dZ1^T @ X: [h2 x h1] = [h2 x B] @ [B x h1]
    matmul_transA(d_dZ1, d_X, d_dW1, h2, batch_size, h1);

    // db1 = sum over batch of dZ1
    sum_rows(d_dZ1, d_db1, batch_size, h2);
}

void MLP::update_parameters() {
    step_count++;
    float beta1_t = powf(beta1, step_count);
    float beta2_t = powf(beta2, step_count);

    // Update layer 1 parameters
    adam_update(d_W1, d_dW1, d_m_W1, d_v_W1, lr, beta1, beta2, epsilon,
                beta1_t, beta2_t, 0.0f, h2 * h1);
    adam_update(d_b1, d_db1, d_m_b1, d_v_b1, lr, beta1, beta2, epsilon,
                beta1_t, beta2_t, 0.0f, h2);

    // Update layer 2 parameters
    adam_update(d_W2, d_dW2, d_m_W2, d_v_W2, lr, beta1, beta2, epsilon,
                beta1_t, beta2_t, 0.0f, h3 * h2);
    adam_update(d_b2, d_db2, d_m_b2, d_v_b2, lr, beta1, beta2, epsilon,
                beta1_t, beta2_t, 0.0f, h3);

    // Update layer 3 parameters
    adam_update(d_W3, d_dW3, d_m_W3, d_v_W3, lr, beta1, beta2, epsilon,
                beta1_t, beta2_t, 0.0f, h4 * h3);
    adam_update(d_b3, d_db3, d_m_b3, d_v_b3, lr, beta1, beta2, epsilon,
                beta1_t, beta2_t, 0.0f, h4);
}

void MLP::forward(const float* h_X, float* h_output, int num_samples) {
    // Process in batches if num_samples > max_batch_size
    if (num_samples > max_batch_size) {
        int num_batches = (num_samples + max_batch_size - 1) / max_batch_size;

        for (int i = 0; i < num_batches; i++) {
            int start_idx = i * max_batch_size;
            int current_batch_size = (start_idx + max_batch_size <= num_samples) ?
                                      max_batch_size : (num_samples - start_idx);

            forward(h_X + start_idx * h1,
                   h_output + start_idx * h4,
                   current_batch_size);
        }
        return;
    }

    // Single batch - fits in allocated buffers
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, num_samples * h1 * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Forward pass
    forward_device(num_samples);

    // Copy output to host
    CUDA_CHECK(cudaMemcpy(h_output, d_Z3, num_samples * h4 * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

float MLP::train_step(const float* h_X, const float* h_Y, int batch_size) {
    // Check batch size
    if (batch_size > max_batch_size) {
        fprintf(stderr, "Error: batch_size (%d) exceeds max_batch_size (%d) in train_step\n",
                batch_size, max_batch_size);
        fprintf(stderr, "       Process your training data in smaller batches.\n");
        return -1.0f;
    }

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, batch_size * h1 * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, batch_size * h4 * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Forward pass
    forward_device(batch_size);

    // Compute loss
    float loss = mse_loss(d_Z3, d_Y, batch_size * h4, batch_size);

    // Backward pass
    backward_device(batch_size);

    // Update parameters
    update_parameters();

    return loss;
}

float MLP::evaluate(const float* h_X, const float* h_Y, int num_samples) {
    // Process in batches if num_samples > max_batch_size
    if (num_samples > max_batch_size) {
        float total_loss = 0.0f;
        int num_batches = (num_samples + max_batch_size - 1) / max_batch_size;

        for (int i = 0; i < num_batches; i++) {
            int start_idx = i * max_batch_size;
            int current_batch_size = (start_idx + max_batch_size <= num_samples) ?
                                      max_batch_size : (num_samples - start_idx);

            float batch_loss = evaluate(h_X + start_idx * h1,
                                       h_Y + start_idx * h4,
                                       current_batch_size);

            // Weighted average (last batch might be smaller)
            total_loss += batch_loss * current_batch_size;
        }

        return total_loss / num_samples;
    }

    // Single batch - fits in allocated buffers
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, num_samples * h1 * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, num_samples * h4 * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Forward pass
    forward_device(num_samples);

    // Compute loss
    return mse_loss(d_Z3, d_Y, num_samples * h4, num_samples);
}

void MLP::save_parameters(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }

    // Save network architecture
    file.write(reinterpret_cast<char*>(&h1), sizeof(int));
    file.write(reinterpret_cast<char*>(&h2), sizeof(int));
    file.write(reinterpret_cast<char*>(&h3), sizeof(int));
    file.write(reinterpret_cast<char*>(&h4), sizeof(int));

    // Helper lambda to save a parameter
    auto save_param = [&](float* d_param, int size) {
        float* h_param = new float[size];
        CUDA_CHECK(cudaMemcpy(h_param, d_param, size * sizeof(float),
                             cudaMemcpyDeviceToHost));
        file.write(reinterpret_cast<char*>(h_param), size * sizeof(float));
        delete[] h_param;
    };

    // Save all parameters
    save_param(d_W1, h2 * h1);
    save_param(d_b1, h2);
    save_param(d_W2, h3 * h2);
    save_param(d_b2, h3);
    save_param(d_W3, h4 * h3);
    save_param(d_b3, h4);

    file.close();
}

void MLP::load_parameters(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filename);
        return;
    }

    // Load and verify network architecture
    int loaded_h1, loaded_h2, loaded_h3, loaded_h4;
    file.read(reinterpret_cast<char*>(&loaded_h1), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_h2), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_h3), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_h4), sizeof(int));

    if (loaded_h1 != h1 || loaded_h2 != h2 || loaded_h3 != h3 || loaded_h4 != h4) {
        fprintf(stderr, "Architecture mismatch in loaded file\n");
        file.close();
        return;
    }

    // Helper lambda to load a parameter
    auto load_param = [&](float* d_param, int size) {
        float* h_param = new float[size];
        file.read(reinterpret_cast<char*>(h_param), size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_param, h_param, size * sizeof(float),
                             cudaMemcpyHostToDevice));
        delete[] h_param;
    };

    // Load all parameters
    load_param(d_W1, h2 * h1);
    load_param(d_b1, h2);
    load_param(d_W2, h3 * h2);
    load_param(d_b2, h3);
    load_param(d_W3, h4 * h3);
    load_param(d_b3, h4);

    file.close();
}
