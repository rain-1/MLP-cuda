#include "transformer_block.h"
#include "transformer_layers.h"
#include "matrix_ops.h"
#include "gradient_utils.h"
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
    int max_seq_len,
    float init_scale
) : d_model(d_model), d_ff(d_ff),
    max_batch_size(max_batch_size), max_seq_len(max_seq_len),
    init_scale(init_scale)
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
    CUDA_CHECK(cudaMalloc(&d_z1, max_batch_size * max_seq_len * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, max_batch_size * max_seq_len * d_ff * sizeof(float)));
}

void FeedForwardNetwork::free_memory() {
    cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_z1); cudaFree(d_hidden);
}

void FeedForwardNetwork::initialize_parameters() {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Xavier initialization
    float std1 = sqrtf(2.0f / (d_model + d_ff));
    curandGenerateNormal(gen, d_W1, d_model * d_ff, 0.0f, std1);
    CUDA_CHECK(cudaMemset(d_b1, 0, d_ff * sizeof(float)));

    // W2 initialization - TEMPORARILY DISABLE DEPTH SCALING TO TEST
    // Scale by init_scale to account for residual path accumulation
    // float std2 = sqrtf(2.0f / (d_ff + d_model)) * init_scale;
    float std2 = sqrtf(2.0f / (d_ff + d_model));  // Standard Xavier, no depth scaling
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

    // First layer: z1 = input · W1^T + b1
    matmul_transB(d_input, d_W1, d_z1, total_tokens, d_model, d_ff);
    add_bias(d_z1, d_b1, d_z1, total_tokens, d_ff);

    // GELU activation: hidden = GELU(z1)
    gelu_forward(d_z1, d_hidden, total_tokens * d_ff);

    // Second layer: output = hidden · W2^T + b2
    matmul_transB(d_hidden, d_W2, d_output, total_tokens, d_ff, d_model);
    add_bias(d_output, d_b2, d_output, total_tokens, d_model);
}

void FeedForwardNetwork::backward_device(
    const float* d_input,
    const float* d_grad_output,
    float* d_grad_input,
    float* d_grad_W1,
    float* d_grad_b1,
    float* d_grad_W2,
    float* d_grad_b2,
    int batch_size,
    int seq_len
) {
    int total_tokens = batch_size * seq_len;

    // Allocate temporary buffers for intermediate gradients
    float* d_grad_hidden;
    float* d_grad_z1;
    CUDA_CHECK(cudaMalloc(&d_grad_hidden, total_tokens * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_z1, total_tokens * d_ff * sizeof(float)));

    // Backward through second layer bias: grad_b2 = sum_rows(grad_output)
    sum_rows(d_grad_output, d_grad_b2, total_tokens, d_model);

    // Backward through second layer matmul: hidden · W2^T
    // Forward was: output = hidden [total_tokens x d_ff] * W2^T [d_ff x d_model]
    matmul_transB_backward(
        d_grad_output,   // grad_C [total_tokens x d_model]
        d_hidden,        // A [total_tokens x d_ff]
        d_W2,            // B [d_model x d_ff]
        d_grad_hidden,   // grad_A [total_tokens x d_ff]
        d_grad_W2,       // grad_B [d_model x d_ff]
        total_tokens,    // M
        d_ff,            // K
        d_model          // N
    );

    // Backward through GELU activation
    gelu_backward(d_grad_hidden, d_z1, d_grad_z1, total_tokens * d_ff);

    // Backward through first layer bias: grad_b1 = sum_rows(grad_z1)
    sum_rows(d_grad_z1, d_grad_b1, total_tokens, d_ff);

    // Backward through first layer matmul: input · W1^T
    // Forward was: z1 = input [total_tokens x d_model] * W1^T [d_model x d_ff]
    matmul_transB_backward(
        d_grad_z1,       // grad_C [total_tokens x d_ff]
        d_input,         // A [total_tokens x d_model]
        d_W1,            // B [d_ff x d_model]
        d_grad_input,    // grad_A [total_tokens x d_model]
        d_grad_W1,       // grad_B [d_ff x d_model]
        total_tokens,    // M
        d_model,         // K
        d_ff             // N
    );

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_grad_hidden));
    CUDA_CHECK(cudaFree(d_grad_z1));
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
    int max_seq_len,
    float residual_scale
) : d_model(d_model), num_heads(num_heads), d_ff(d_ff),
    max_batch_size(max_batch_size), max_seq_len(max_seq_len),
    residual_scale(residual_scale)
{
    // GPT-2/3 style: residual_scale already accounts for depth, use it for init scaling
    attention = new MultiHeadAttention(d_model, num_heads, max_seq_len, max_batch_size, residual_scale);
    ffn = new FeedForwardNetwork(d_model, d_ff, max_batch_size, max_seq_len, residual_scale);

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
    // 1. Layer norm before attention
    layer_norm(d_input, d_attn_normed, d_ln1_gamma, d_ln1_beta,
               batch_size, seq_len, d_model);

    // 2. Multi-head self-attention
    attention->forward_device_to_device(d_attn_normed, d_attn_output,
                                       batch_size, seq_len, d_mask);

    // 3. Scale attention output and add residual connection
    // output = input + residual_scale * attn_output
    // Attention outputs can be large, so we scale them down before residual
    int total_size = batch_size * seq_len * d_model;
    add_residual(d_input, d_attn_output, d_attn_output, total_size, residual_scale);

    // 4. Layer norm before FFN
    layer_norm(d_attn_output, d_ffn_normed, d_ln2_gamma, d_ln2_beta,
               batch_size, seq_len, d_model);

    // 5. Feed-forward network
    ffn->forward_device(d_ffn_normed, d_ffn_output, batch_size, seq_len);

    // 6. Scale FFN output and add residual connection
    // output = attn_output + residual_scale * ffn_output
    add_residual(d_attn_output, d_ffn_output, d_output, total_size, residual_scale);
}

// Residual connection helper kernel with scaling
__global__ void add_residual_kernel(
    const float* input,
    const float* residual,
    float* output,
    int size,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + scale * residual[idx];
    }
}

void add_residual(
    const float* d_input,
    const float* d_residual,
    float* d_output,
    int size,
    float scale
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    add_residual_kernel<<<grid_size, block_size>>>(
        d_input, d_residual, d_output, size, scale
    );
    CUDA_CHECK(cudaGetLastError());
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

void TransformerBlock::backward_device(
    const float* d_input,
    const float* d_grad_output,
    float* d_grad_input,
    float* d_grad_ln1_gamma, float* d_grad_ln1_beta,
    float* d_grad_ln2_gamma, float* d_grad_ln2_beta,
    float* d_grad_attn_W_Q, float* d_grad_attn_b_Q,
    float* d_grad_attn_W_K, float* d_grad_attn_b_K,
    float* d_grad_attn_W_V, float* d_grad_attn_b_V,
    float* d_grad_attn_W_O, float* d_grad_attn_b_O,
    float* d_grad_ffn_W1, float* d_grad_ffn_b1,
    float* d_grad_ffn_W2, float* d_grad_ffn_b2,
    int batch_size,
    int seq_len
) {
    int total_size = batch_size * seq_len * d_model;

    // Allocate temporary gradient buffers
    float *d_grad_attn_output_total, *d_grad_attn_output_from_res2;
    float *d_grad_ffn_normed, *d_grad_attn_output_from_ln2;
    float *d_grad_attn_output_from_attn, *d_grad_attn_normed;
    float *d_grad_input_from_res1, *d_grad_input_from_ln1;
    float *d_temp;

    CUDA_CHECK(cudaMalloc(&d_grad_attn_output_total, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_attn_output_from_res2, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_ffn_normed, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_attn_output_from_ln2, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_attn_output_from_attn, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_attn_normed, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input_from_res1, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input_from_ln1, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, total_size * sizeof(float)));

    // ========================================================================
    // Backward through second residual: output = attn_output + residual_scale * ffn_output
    // grad_attn_output = grad_output
    // grad_ffn_output = residual_scale * grad_output
    // ========================================================================

    CUDA_CHECK(cudaMemcpy(d_grad_attn_output_from_res2, d_grad_output,
                         total_size * sizeof(float), cudaMemcpyDeviceToDevice));
    float *d_grad_ffn_output = d_temp;  // Reuse temp buffer
    CUDA_CHECK(cudaMemcpy(d_grad_ffn_output, d_grad_output,
                         total_size * sizeof(float), cudaMemcpyDeviceToDevice));
    // Scale the gradient for the residual path
    scale_gradients(d_grad_ffn_output, total_size, residual_scale);

    // ========================================================================
    // Backward through FFN
    // ========================================================================

    ffn->backward_device(
        d_ffn_normed,         // Input (saved from forward)
        d_grad_ffn_output,    // Gradient w.r.t. FFN output
        d_grad_ffn_normed,    // Gradient w.r.t. FFN input (output)
        d_grad_ffn_W1, d_grad_ffn_b1,
        d_grad_ffn_W2, d_grad_ffn_b2,
        batch_size, seq_len
    );

    // ========================================================================
    // Backward through second layer norm
    // ========================================================================

    layer_norm_backward(
        d_grad_ffn_normed,           // Gradient w.r.t. normalized output
        d_attn_output,               // Input (saved from forward)
        d_ln2_gamma,                 // Layer norm gamma
        d_grad_attn_output_from_ln2, // Gradient w.r.t. input (output)
        d_grad_ln2_gamma,            // Gradient w.r.t. gamma (output)
        d_grad_ln2_beta,             // Gradient w.r.t. beta (output)
        batch_size, seq_len, d_model,
        1e-5f                        // epsilon
    );

    // ========================================================================
    // Accumulate gradients to attn_output from residual and layer norm
    // ========================================================================

    elementwise_add(d_grad_attn_output_from_res2, d_grad_attn_output_from_ln2,
                    d_grad_attn_output_total, total_size);

    // ========================================================================
    // Backward through first residual: attn_output = input + residual_scale * attn_output
    // grad_input = grad_attn_output_total
    // grad_attn_output = residual_scale * grad_attn_output_total
    // ========================================================================

    CUDA_CHECK(cudaMemcpy(d_grad_input_from_res1, d_grad_attn_output_total,
                         total_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_attn_output_from_attn, d_grad_attn_output_total,
                         total_size * sizeof(float), cudaMemcpyDeviceToDevice));
    // Scale the gradient for the residual path
    scale_gradients(d_grad_attn_output_from_attn, total_size, residual_scale);

    // ========================================================================
    // Backward through attention
    // ========================================================================

    attention->backward_device_to_device(
        d_attn_normed,               // Input (saved from forward)
        d_grad_attn_output_from_attn, // Gradient w.r.t. attention output
        d_grad_attn_normed,          // Gradient w.r.t. attention input (output)
        d_grad_attn_W_Q, d_grad_attn_b_Q,
        d_grad_attn_W_K, d_grad_attn_b_K,
        d_grad_attn_W_V, d_grad_attn_b_V,
        d_grad_attn_W_O, d_grad_attn_b_O,
        batch_size, seq_len
    );

    // ========================================================================
    // Backward through first layer norm
    // ========================================================================

    layer_norm_backward(
        d_grad_attn_normed,     // Gradient w.r.t. normalized output
        d_input,                // Input (from forward)
        d_ln1_gamma,            // Layer norm gamma
        d_grad_input_from_ln1,  // Gradient w.r.t. input (output)
        d_grad_ln1_gamma,       // Gradient w.r.t. gamma (output)
        d_grad_ln1_beta,        // Gradient w.r.t. beta (output)
        batch_size, seq_len, d_model,
        1e-5f                   // epsilon
    );

    // ========================================================================
    // Accumulate final gradient to input from both residual paths
    // ========================================================================

    elementwise_add(d_grad_input_from_res1, d_grad_input_from_ln1,
                    d_grad_input, total_size);

    // Free temporary buffers
    CUDA_CHECK(cudaFree(d_grad_attn_output_total));
    CUDA_CHECK(cudaFree(d_grad_attn_output_from_res2));
    CUDA_CHECK(cudaFree(d_grad_ffn_normed));
    CUDA_CHECK(cudaFree(d_grad_attn_output_from_ln2));
    CUDA_CHECK(cudaFree(d_grad_attn_output_from_attn));
    CUDA_CHECK(cudaFree(d_grad_attn_normed));
    CUDA_CHECK(cudaFree(d_grad_input_from_res1));
    CUDA_CHECK(cudaFree(d_grad_input_from_ln1));
    CUDA_CHECK(cudaFree(d_temp));
}
