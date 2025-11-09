#include "transformer_block.h"
#include "transformer_layers.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <fstream>

// ============================================================================
// Feed-Forward Network
// ============================================================================

FeedForwardNetwork::FeedForwardNetwork(
    int d_model,
    int d_ff,
    int max_batch_size,
    int max_seq_len
) : d_model(d_model), d_ff(d_ff),
    max_batch_size(max_batch_size), max_seq_len(max_seq_len)
{
    allocate_memory();
    initialize_parameters();
}

FeedForwardNetwork::~FeedForwardNetwork() {
    free_memory();
}

void FeedForwardNetwork::allocate_memory() {
    // Parameters
    CUDA_CHECK(cudaMalloc(&d_W1, d_model * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, d_ff * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, d_model * sizeof(float)));

    // Buffers
    CUDA_CHECK(cudaMalloc(&d_hidden, max_batch_size * max_seq_len * d_ff * sizeof(float)));
}

void FeedForwardNetwork::free_memory() {
    cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_hidden);
}

void FeedForwardNetwork::initialize_parameters() {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Xavier initialization
    float std1 = sqrtf(2.0f / (d_model + d_ff));
    curandGenerateNormal(gen, d_W1, d_model * d_ff, 0.0f, std1);
    CUDA_CHECK(cudaMemset(d_b1, 0, d_ff * sizeof(float)));

    float std2 = sqrtf(2.0f / (d_ff + d_model));
    curandGenerateNormal(gen, d_W2, d_ff * d_model, 0.0f, std2);
    CUDA_CHECK(cudaMemset(d_b2, 0, d_model * sizeof(float)));

    curandDestroyGenerator(gen);
}

void FeedForwardNetwork::forward_device(
    const float* d_input,
    float* d_output,
    int batch_size,
    int seq_len
) {
    int total_tokens = batch_size * seq_len;

    // First layer: hidden = GELU(input · W1^T + b1)
    matmul_transB(d_input, d_W1, d_hidden, total_tokens, d_model, d_ff);
    add_bias(d_hidden, d_b1, d_hidden, total_tokens, d_ff);
    gelu_forward(d_hidden, d_hidden, total_tokens * d_ff);

    // Second layer: output = hidden · W2^T + b2
    matmul_transB(d_hidden, d_W2, d_output, total_tokens, d_ff, d_model);
    add_bias(d_output, d_b2, d_output, total_tokens, d_model);
}

void FeedForwardNetwork::save_parameters(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return;

    file.write(reinterpret_cast<char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<char*>(&d_ff), sizeof(int));

    auto save_param = [&](float* d_param, int size) {
        float* h_param = new float[size];
        CUDA_CHECK(cudaMemcpy(h_param, d_param, size * sizeof(float),
                             cudaMemcpyDeviceToHost));
        file.write(reinterpret_cast<char*>(h_param), size * sizeof(float));
        delete[] h_param;
    };

    save_param(d_W1, d_model * d_ff);
    save_param(d_b1, d_ff);
    save_param(d_W2, d_ff * d_model);
    save_param(d_b2, d_model);

    file.close();
}

void FeedForwardNetwork::load_parameters(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return;

    int loaded_d_model, loaded_d_ff;
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(int));

    if (loaded_d_model != d_model || loaded_d_ff != d_ff) {
        fprintf(stderr, "FFN dimension mismatch\n");
        file.close();
        return;
    }

    auto load_param = [&](float* d_param, int size) {
        float* h_param = new float[size];
        file.read(reinterpret_cast<char*>(h_param), size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_param, h_param, size * sizeof(float),
                             cudaMemcpyHostToDevice));
        delete[] h_param;
    };

    load_param(d_W1, d_model * d_ff);
    load_param(d_b1, d_ff);
    load_param(d_W2, d_ff * d_model);
    load_param(d_b2, d_model);

    file.close();
}

// ============================================================================
// Transformer Block
// ============================================================================

TransformerBlock::TransformerBlock(
    int d_model,
    int num_heads,
    int d_ff,
    int max_batch_size,
    int max_seq_len
) : d_model(d_model), num_heads(num_heads), d_ff(d_ff),
    max_batch_size(max_batch_size), max_seq_len(max_seq_len)
{
    attention = new MultiHeadAttention(d_model, num_heads, max_seq_len, max_batch_size);
    ffn = new FeedForwardNetwork(d_model, d_ff, max_batch_size, max_seq_len);

    allocate_memory();
    initialize_parameters();
}

TransformerBlock::~TransformerBlock() {
    delete attention;
    delete ffn;
    free_memory();
}

void TransformerBlock::allocate_memory() {
    // Layer norm parameters
    CUDA_CHECK(cudaMalloc(&d_ln1_gamma, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln1_beta, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln2_gamma, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln2_beta, d_model * sizeof(float)));

    // Intermediate buffers
    CUDA_CHECK(cudaMalloc(&d_attn_output, max_batch_size * max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_normed, max_batch_size * max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_output, max_batch_size * max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_normed, max_batch_size * max_seq_len * d_model * sizeof(float)));
}

void TransformerBlock::free_memory() {
    cudaFree(d_ln1_gamma); cudaFree(d_ln1_beta);
    cudaFree(d_ln2_gamma); cudaFree(d_ln2_beta);
    cudaFree(d_attn_output); cudaFree(d_attn_normed);
    cudaFree(d_ffn_output); cudaFree(d_ffn_normed);
}

void TransformerBlock::initialize_parameters() {
    // Initialize layer norm parameters (gamma=1, beta=0)
    float* h_ones = new float[d_model];
    float* h_zeros = new float[d_model];

    for (int i = 0; i < d_model; i++) {
        h_ones[i] = 1.0f;
        h_zeros[i] = 0.0f;
    }

    CUDA_CHECK(cudaMemcpy(d_ln1_gamma, h_ones, d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln1_beta, h_zeros, d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln2_gamma, h_ones, d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln2_beta, h_zeros, d_model * sizeof(float), cudaMemcpyHostToDevice));

    delete[] h_ones;
    delete[] h_zeros;
}

void TransformerBlock::forward_device(
    const float* d_input,
    float* d_output,
    const float* d_mask,
    int batch_size,
    int seq_len
) {
    // 1. Multi-head self-attention with residual connection
    // Note: MultiHeadAttention handles the forward pass internally
    // We need to adapt it to work with device pointers directly

    // For now, use a simplified approach:
    // attn_output = Attention(input)
    // Copy input to temporary buffer for attention
    CUDA_CHECK(cudaMemcpy(d_attn_output, d_input,
                         batch_size * seq_len * d_model * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // TODO: Call attention forward pass with mask
    // attention->forward_device(...)

    // 2. Add residual and layer norm: normed = LayerNorm(input + attn_output)
    // For now, simplified: normed = LayerNorm(attn_output)
    layer_norm(d_attn_output, d_attn_normed, d_ln1_gamma, d_ln1_beta,
               batch_size, seq_len, d_model);

    // 3. Feed-forward network
    ffn->forward_device(d_attn_normed, d_ffn_output, batch_size, seq_len);

    // 4. Add residual and layer norm: output = LayerNorm(normed + ffn_output)
    layer_norm(d_ffn_output, d_output, d_ln2_gamma, d_ln2_beta,
               batch_size, seq_len, d_model);
}

void TransformerBlock::save_parameters(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return;

    file.write(reinterpret_cast<char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<char*>(&num_heads), sizeof(int));
    file.write(reinterpret_cast<char*>(&d_ff), sizeof(int));

    auto save_param = [&](float* d_param, int size) {
        float* h_param = new float[size];
        CUDA_CHECK(cudaMemcpy(h_param, d_param, size * sizeof(float),
                             cudaMemcpyDeviceToHost));
        file.write(reinterpret_cast<char*>(h_param), size * sizeof(float));
        delete[] h_param;
    };

    save_param(d_ln1_gamma, d_model);
    save_param(d_ln1_beta, d_model);
    save_param(d_ln2_gamma, d_model);
    save_param(d_ln2_beta, d_model);

    file.close();

    // Save sub-components
    std::string base(filename);
    attention->save_parameters((base + ".attn").c_str());
    ffn->save_parameters((base + ".ffn").c_str());
}

void TransformerBlock::load_parameters(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return;

    int loaded_d_model, loaded_num_heads, loaded_d_ff;
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_num_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(int));

    if (loaded_d_model != d_model || loaded_num_heads != num_heads || loaded_d_ff != d_ff) {
        fprintf(stderr, "TransformerBlock dimension mismatch\n");
        file.close();
        return;
    }

    auto load_param = [&](float* d_param, int size) {
        float* h_param = new float[size];
        file.read(reinterpret_cast<char*>(h_param), size * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_param, h_param, size * sizeof(float),
                             cudaMemcpyHostToDevice));
        delete[] h_param;
    };

    load_param(d_ln1_gamma, d_model);
    load_param(d_ln1_beta, d_model);
    load_param(d_ln2_gamma, d_model);
    load_param(d_ln2_beta, d_model);

    file.close();

    // Load sub-components
    std::string base(filename);
    attention->load_parameters((base + ".attn").c_str());
    ffn->load_parameters((base + ".ffn").c_str());
}
