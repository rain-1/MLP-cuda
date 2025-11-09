#include "multi_head_attention.h"
#include "matrix_ops.h"
#include "attention_ops.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <fstream>

MultiHeadAttention::MultiHeadAttention(
    int d_model,
    int num_heads,
    int max_seq_len,
    int max_batch_size,
    float init_scale
) : d_model(d_model),
    num_heads(num_heads),
    max_seq_len(max_seq_len),
    max_batch_size(max_batch_size),
    init_scale(init_scale)
{
    // Check that d_model is divisible by num_heads
    if (d_model % num_heads != 0) {
        fprintf(stderr, "Error: d_model (%d) must be divisible by num_heads (%d)\n",
                d_model, num_heads);
        exit(EXIT_FAILURE);
    }

    d_k = d_v = d_model / num_heads;

    allocate_memory();
    initialize_parameters();
}

MultiHeadAttention::~MultiHeadAttention() {
    free_memory();
}

void MultiHeadAttention::allocate_memory() {
    // Projection weights: [d_model x (h * d_k)]
    CUDA_CHECK(cudaMalloc(&d_W_Q, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_K, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_V, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_O, d_model * d_model * sizeof(float)));

    // Projection biases: [h * d_k] = [d_model]
    CUDA_CHECK(cudaMalloc(&d_b_Q, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_K, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_V, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b_O, d_model * sizeof(float)));

    // Intermediate buffers
    CUDA_CHECK(cudaMalloc(&d_Q_proj, max_batch_size * max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_proj, max_batch_size * max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_proj, max_batch_size * max_seq_len * d_model * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Q_heads, max_batch_size * num_heads * max_seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_heads, max_batch_size * num_heads * max_seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_heads, max_batch_size * num_heads * max_seq_len * d_v * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_scores, max_batch_size * num_heads * max_seq_len * max_seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_weights, max_batch_size * num_heads * max_seq_len * max_seq_len * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_context, max_batch_size * num_heads * max_seq_len * d_v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_concat, max_batch_size * max_seq_len * d_model * sizeof(float)));

    // Input/mask buffers
    CUDA_CHECK(cudaMalloc(&d_X, max_batch_size * max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_KV, max_batch_size * max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask, max_batch_size * num_heads * max_seq_len * max_seq_len * sizeof(float)));
}

void MultiHeadAttention::free_memory() {
    // Parameters
    cudaFree(d_W_Q); cudaFree(d_W_K); cudaFree(d_W_V); cudaFree(d_W_O);
    cudaFree(d_b_Q); cudaFree(d_b_K); cudaFree(d_b_V); cudaFree(d_b_O);

    // Buffers
    cudaFree(d_Q_proj); cudaFree(d_K_proj); cudaFree(d_V_proj);
    cudaFree(d_Q_heads); cudaFree(d_K_heads); cudaFree(d_V_heads);
    cudaFree(d_scores); cudaFree(d_attn_weights);
    cudaFree(d_context); cudaFree(d_output_concat);
    cudaFree(d_X); cudaFree(d_KV); cudaFree(d_mask);
}

void MultiHeadAttention::initialize_parameters() {
    // Xavier initialization: W ~ N(0, 1/√d_model)
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    float std = 1.0f / sqrtf((float)d_model);

    curandGenerateNormal(gen, d_W_Q, d_model * d_model, 0.0f, std);
    curandGenerateNormal(gen, d_W_K, d_model * d_model, 0.0f, std);
    curandGenerateNormal(gen, d_W_V, d_model * d_model, 0.0f, std);

    // W_O initialization with depth scaling (GPT-2/3 style)
    // Scale by init_scale to account for residual path accumulation
    float std_o = std * init_scale;
    curandGenerateNormal(gen, d_W_O, d_model * d_model, 0.0f, std_o);

    // Initialize biases to zero
    CUDA_CHECK(cudaMemset(d_b_Q, 0, d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b_K, 0, d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b_V, 0, d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b_O, 0, d_model * sizeof(float)));

    curandDestroyGenerator(gen);
}

void MultiHeadAttention::forward_device(int batch_size, int seq_len_q, int seq_len_kv) {
    // 1. Project to Q, K, V
    //    Q = X · W_Q^T + b_Q  ->  [B, N, d_model]
    matmul_transB(d_X, d_W_Q, d_Q_proj, batch_size * seq_len_q, d_model, d_model);
    add_bias(d_Q_proj, d_b_Q, d_Q_proj, batch_size * seq_len_q, d_model);

    //    K = KV · W_K^T + b_K  ->  [B, M, d_model]
    matmul_transB(d_KV, d_W_K, d_K_proj, batch_size * seq_len_kv, d_model, d_model);
    add_bias(d_K_proj, d_b_K, d_K_proj, batch_size * seq_len_kv, d_model);

    //    V = KV · W_V^T + b_V  ->  [B, M, d_model]
    matmul_transB(d_KV, d_W_V, d_V_proj, batch_size * seq_len_kv, d_model, d_model);
    add_bias(d_V_proj, d_b_V, d_V_proj, batch_size * seq_len_kv, d_model);

    // 2. Reshape and transpose to separate heads
    //    [B, N, h*d_k] -> [B*h, N, d_k]
    reshape_BNhd_to_BhNd(d_Q_proj, d_Q_heads, batch_size, seq_len_q, num_heads, d_k);
    reshape_BNhd_to_BhNd(d_K_proj, d_K_heads, batch_size, seq_len_kv, num_heads, d_k);
    reshape_BNhd_to_BhNd(d_V_proj, d_V_heads, batch_size, seq_len_kv, num_heads, d_v);

    // 3. Scaled dot-product attention for all heads
    //    Output: [B*h, N, d_v]
    scaled_dot_product_attention(
        d_Q_heads, d_K_heads, d_V_heads,
        d_context,
        d_scores, d_attn_weights,
        batch_size * num_heads,
        seq_len_q, seq_len_kv,
        d_k, d_v,
        nullptr  // No mask for now (will add mask support)
    );

    // 4. Reshape and concatenate heads
    //    [B*h, N, d_v] -> [B, N, h*d_v]
    reshape_BhNd_to_BNhd(d_context, d_output_concat, batch_size, seq_len_q, num_heads, d_v);

    // 5. Output projection
    //    Output = concat · W_O^T + b_O  ->  [B, N, d_model]
    matmul_transB(d_output_concat, d_W_O, d_X, batch_size * seq_len_q, d_model, d_model);
    add_bias(d_X, d_b_O, d_X, batch_size * seq_len_q, d_model);
}

void MultiHeadAttention::forward(
    const float* h_X,
    float* h_output,
    int batch_size,
    int seq_len,
    const float* h_mask
) {
    // Self-attention: Q = K = V = X
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, batch_size * seq_len * d_model * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_KV, h_X, batch_size * seq_len * d_model * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Forward pass
    forward_device(batch_size, seq_len, seq_len);

    // Copy output to host
    CUDA_CHECK(cudaMemcpy(h_output, d_X, batch_size * seq_len * d_model * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

void MultiHeadAttention::forward_cross(
    const float* h_Q,
    const float* h_KV,
    float* h_output,
    int batch_size,
    int seq_len_q,
    int seq_len_kv,
    const float* h_mask
) {
    // Cross-attention: Q and K/V are different
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_X, h_Q, batch_size * seq_len_q * d_model * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_KV, h_KV, batch_size * seq_len_kv * d_model * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Forward pass
    forward_device(batch_size, seq_len_q, seq_len_kv);

    // Copy output to host
    CUDA_CHECK(cudaMemcpy(h_output, d_X, batch_size * seq_len_q * d_model * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

void MultiHeadAttention::forward_device_to_device(
    const float* d_input,
    float* d_output,
    int batch_size,
    int seq_len,
    const float* d_mask_input
) {
    // Self-attention: Q = K = V = input
    // Copy input to internal buffers
    CUDA_CHECK(cudaMemcpy(d_X, d_input, batch_size * seq_len * d_model * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_KV, d_input, batch_size * seq_len * d_model * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // Copy mask if provided
    if (d_mask_input != nullptr && d_mask != nullptr) {
        CUDA_CHECK(cudaMemcpy(d_mask, d_mask_input, seq_len * seq_len * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }

    // Forward pass (result written to d_X)
    forward_device(batch_size, seq_len, seq_len);

    // Copy result to output
    CUDA_CHECK(cudaMemcpy(d_output, d_X, batch_size * seq_len * d_model * sizeof(float),
                         cudaMemcpyDeviceToDevice));
}

void MultiHeadAttention::save_parameters(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }

    // Save architecture
    file.write(reinterpret_cast<char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<char*>(&num_heads), sizeof(int));

    // Helper lambda
    auto save_param = [&](float* d_param, int size) {
        float* h_param = new float[size];
        CUDA_CHECK(cudaMemcpy(h_param, d_param, size * sizeof(float),
                             cudaMemcpyDeviceToHost));
        file.write(reinterpret_cast<char*>(h_param), size * sizeof(float));
        delete[] h_param;
    };

    // Save all parameters
    save_param(d_W_Q, d_model * d_model);
    save_param(d_W_K, d_model * d_model);
    save_param(d_W_V, d_model * d_model);
    save_param(d_W_O, d_model * d_model);
    save_param(d_b_Q, d_model);
    save_param(d_b_K, d_model);
    save_param(d_b_V, d_model);
    save_param(d_b_O, d_model);

    file.close();
}

void MultiHeadAttention::load_parameters(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filename);
        return;
    }

    // Load and verify architecture
    int loaded_d_model, loaded_num_heads;
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(int));

    if (loaded_d_model != d_model || loaded_num_heads != num_heads) {
        fprintf(stderr, "Architecture mismatch in loaded file\n");
        file.close();
        return;
    }

    // Helper lambda
    auto load_param = [&](float* d_param, int size) {
        float* h_param = new float[size];
        file.read(reinterpret_cast<char*>(h_param), size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_param, h_param, size * sizeof(float),
                             cudaMemcpyHostToDevice));
        delete[] h_param;
    };

    // Load all parameters
    load_param(d_W_Q, d_model * d_model);
    load_param(d_W_K, d_model * d_model);
    load_param(d_W_V, d_model * d_model);
    load_param(d_W_O, d_model * d_model);
    load_param(d_b_Q, d_model);
    load_param(d_b_K, d_model);
    load_param(d_b_V, d_model);
    load_param(d_b_O, d_model);

    file.close();
}

void MultiHeadAttention::backward_device_to_device(
    const float* d_input,
    const float* d_grad_output,
    float* d_grad_input,
    float* d_grad_W_Q, float* d_grad_b_Q,
    float* d_grad_W_K, float* d_grad_b_K,
    float* d_grad_W_V, float* d_grad_b_V,
    float* d_grad_W_O, float* d_grad_b_O,
    int batch_size,
    int seq_len
) {
    // Allocate temporary gradient buffers
    float *d_grad_output_concat, *d_grad_context;
    float *d_grad_Q_heads, *d_grad_K_heads, *d_grad_V_heads;
    float *d_grad_Q_proj, *d_grad_K_proj, *d_grad_V_proj;
    float *d_grad_scores_buffer, *d_grad_attn_buffer;
    float *d_grad_input_from_Q, *d_grad_input_from_K, *d_grad_input_from_V;

    CUDA_CHECK(cudaMalloc(&d_grad_output_concat, batch_size * seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_context, batch_size * num_heads * seq_len * d_v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_Q_heads, batch_size * num_heads * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_K_heads, batch_size * num_heads * seq_len * d_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_V_heads, batch_size * num_heads * seq_len * d_v * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_Q_proj, batch_size * seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_K_proj, batch_size * seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_V_proj, batch_size * seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_scores_buffer, batch_size * num_heads * seq_len * seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_attn_buffer, batch_size * num_heads * seq_len * seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input_from_Q, batch_size * seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input_from_K, batch_size * seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input_from_V, batch_size * seq_len * d_model * sizeof(float)));

    // Note: Forward pass saved intermediate values in member buffers
    // d_Q_proj, d_K_proj, d_V_proj, d_Q_heads, d_K_heads, d_V_heads,
    // d_output_concat, d_attn_weights

    // For self-attention, we need input in d_X and d_KV
    CUDA_CHECK(cudaMemcpy(d_X, d_input, batch_size * seq_len * d_model * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_KV, d_input, batch_size * seq_len * d_model * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // ========================================================================
    // Backward through output projection: output = output_concat · W_O^T + b_O
    // ========================================================================

    // Gradient w.r.t. b_O
    sum_rows(d_grad_output, d_grad_b_O, batch_size * seq_len, d_model);

    // Gradients w.r.t. output_concat and W_O
    matmul_transB_backward(
        d_grad_output,        // grad_C [B*seq_len x d_model]
        d_output_concat,      // A [B*seq_len x d_model]
        d_W_O,                // B [d_model x d_model]
        d_grad_output_concat, // grad_A [B*seq_len x d_model]
        d_grad_W_O,           // grad_B [d_model x d_model]
        batch_size * seq_len, // M
        d_model,              // K
        d_model               // N
    );

    // ========================================================================
    // Backward through reshape/concatenate: [B*h, N, d_v] -> [B, N, h*d_v]
    // ========================================================================

    reshape_BNhd_to_BhNd(d_grad_output_concat, d_grad_context, batch_size, seq_len, num_heads, d_v);

    // ========================================================================
    // Backward through scaled dot-product attention
    // ========================================================================

    scaled_dot_product_attention_backward(
        d_Q_heads, d_K_heads, d_V_heads,  // Forward inputs (saved in member buffers)
        d_attn_weights,                    // Saved from forward
        d_grad_context,                    // Gradient w.r.t. context
        d_grad_Q_heads,                    // Gradient w.r.t. Q_heads (output)
        d_grad_K_heads,                    // Gradient w.r.t. K_heads (output)
        d_grad_V_heads,                    // Gradient w.r.t. V_heads (output)
        d_grad_scores_buffer,              // Temporary
        d_grad_attn_buffer,                // Temporary
        batch_size * num_heads,            // Bh
        seq_len, seq_len,                  // N, M
        d_k, d_v,                          // d_k, d_v
        nullptr                            // mask (TODO: add mask support)
    );

    // ========================================================================
    // Backward through reshape: [B, N, h*d] -> [B*h, N, d]
    // ========================================================================

    reshape_BhNd_to_BNhd(d_grad_Q_heads, d_grad_Q_proj, batch_size, seq_len, num_heads, d_k);
    reshape_BhNd_to_BNhd(d_grad_K_heads, d_grad_K_proj, batch_size, seq_len, num_heads, d_k);
    reshape_BhNd_to_BNhd(d_grad_V_heads, d_grad_V_proj, batch_size, seq_len, num_heads, d_v);

    // ========================================================================
    // Backward through Q projection: Q_proj = X · W_Q^T + b_Q
    // ========================================================================

    sum_rows(d_grad_Q_proj, d_grad_b_Q, batch_size * seq_len, d_model);
    matmul_transB_backward(
        d_grad_Q_proj,       // grad_C [B*seq_len x d_model]
        d_X,                 // A [B*seq_len x d_model]
        d_W_Q,               // B [d_model x d_model]
        d_grad_input_from_Q, // grad_A [B*seq_len x d_model]
        d_grad_W_Q,          // grad_B [d_model x d_model]
        batch_size * seq_len, // M
        d_model,             // K
        d_model              // N
    );

    // ========================================================================
    // Backward through K projection: K_proj = KV · W_K^T + b_K
    // ========================================================================

    sum_rows(d_grad_K_proj, d_grad_b_K, batch_size * seq_len, d_model);
    matmul_transB_backward(
        d_grad_K_proj,       // grad_C [B*seq_len x d_model]
        d_KV,                // A [B*seq_len x d_model]
        d_W_K,               // B [d_model x d_model]
        d_grad_input_from_K, // grad_A [B*seq_len x d_model]
        d_grad_W_K,          // grad_B [d_model x d_model]
        batch_size * seq_len, // M
        d_model,             // K
        d_model              // N
    );

    // ========================================================================
    // Backward through V projection: V_proj = KV · W_V^T + b_V
    // ========================================================================

    sum_rows(d_grad_V_proj, d_grad_b_V, batch_size * seq_len, d_model);
    matmul_transB_backward(
        d_grad_V_proj,       // grad_C [B*seq_len x d_model]
        d_KV,                // A [B*seq_len x d_model]
        d_W_V,               // B [d_model x d_model]
        d_grad_input_from_V, // grad_A [B*seq_len x d_model]
        d_grad_W_V,          // grad_B [d_model x d_model]
        batch_size * seq_len, // M
        d_model,             // K
        d_model              // N
    );

    // ========================================================================
    // Accumulate gradients from Q, K, V paths to get total gradient w.r.t. input
    // For self-attention: d_grad_input = grad_Q + grad_K + grad_V
    // ========================================================================

    int total_size = batch_size * seq_len * d_model;

    // First, copy grad_input_from_Q to output
    CUDA_CHECK(cudaMemcpy(d_grad_input, d_grad_input_from_Q, total_size * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // Then add grad_input_from_K
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, total_size * sizeof(float)));
    elementwise_add(d_grad_input, d_grad_input_from_K, d_temp, total_size);
    CUDA_CHECK(cudaMemcpy(d_grad_input, d_temp, total_size * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // Finally add grad_input_from_V
    elementwise_add(d_grad_input, d_grad_input_from_V, d_temp, total_size);
    CUDA_CHECK(cudaMemcpy(d_grad_input, d_temp, total_size * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_grad_output_concat));
    CUDA_CHECK(cudaFree(d_grad_context));
    CUDA_CHECK(cudaFree(d_grad_Q_heads));
    CUDA_CHECK(cudaFree(d_grad_K_heads));
    CUDA_CHECK(cudaFree(d_grad_V_heads));
    CUDA_CHECK(cudaFree(d_grad_Q_proj));
    CUDA_CHECK(cudaFree(d_grad_K_proj));
    CUDA_CHECK(cudaFree(d_grad_V_proj));
    CUDA_CHECK(cudaFree(d_grad_scores_buffer));
    CUDA_CHECK(cudaFree(d_grad_attn_buffer));
    CUDA_CHECK(cudaFree(d_grad_input_from_Q));
    CUDA_CHECK(cudaFree(d_grad_input_from_K));
    CUDA_CHECK(cudaFree(d_grad_input_from_V));
}
