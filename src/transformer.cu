#include "transformer.h"
#include "transformer_block.h"
#include "transformer_layers.h"
#include "matrix_ops.h"
#include "loss.h"
#include "adam.h"
#include "gradient_utils.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <fstream>

// ============================================================================
// CUDA Kernels
// ============================================================================

// Accumulate gradients for position embeddings across batch
__global__ void accumulate_position_gradients_kernel(
    const float* d_grad_input,      // [B, N, d_model]
    float* d_grad_pos_embeddings,   // [N, d_model]
    int B, int N, int d_model
) {
    int pos = blockIdx.x;
    int dim = threadIdx.x + blockIdx.y * blockDim.x;

    if (pos < N && dim < d_model) {
        float grad_sum = 0.0f;
        for (int b = 0; b < B; b++) {
            grad_sum += d_grad_input[b * N * d_model + pos * d_model + dim];
        }
        atomicAdd(&d_grad_pos_embeddings[pos * d_model + dim], grad_sum);
    }
}

// Add position embeddings to token embeddings
__global__ void add_position_embeddings_kernel(
    const float* token_emb,
    const float* pos_emb,
    float* output,
    int B, int N, int d_model
) {
    int batch = blockIdx.y;
    int seq = blockIdx.x;
    int dim = threadIdx.x;

    if (batch < B && seq < N && dim < d_model) {
        int idx = batch * N * d_model + seq * d_model + dim;
        output[idx] = token_emb[idx] + pos_emb[seq * d_model + dim];
    }
}

// Softmax for sampling (single row)
__global__ void softmax_row_kernel(
    float* logits,
    int vocab_size,
    float temperature
) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    // Apply temperature and find max
    float max_val = -INFINITY;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        logits[i] /= temperature;
        max_val = fmaxf(max_val, logits[i]);
    }

    shared[tid] = max_val;
    __syncthreads();

    // Reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    shared[tid] = sum;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    sum = shared[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        logits[i] /= sum;
    }
}

// ============================================================================
// Transformer Implementation
// ============================================================================

Transformer::Transformer(
    int vocab_size,
    int d_model,
    int num_layers,
    int num_heads,
    int d_ff,
    int max_seq_len,
    int max_batch_size,
    float residual_scale
) : vocab_size(vocab_size), d_model(d_model), num_layers(num_layers),
    num_heads(num_heads), d_ff(d_ff), max_seq_len(max_seq_len),
    max_batch_size(max_batch_size), training_step(0)
{
    // Auto-compute residual scale if not specified
    if (residual_scale < 0.0f) {
        this->residual_scale = 1.0f / sqrtf((float)num_layers);
    } else {
        this->residual_scale = residual_scale;
    }

    printf("Transformer: Using residual_scale = %.4f (num_layers=%d)\n",
           this->residual_scale, num_layers);

    allocate_memory();

    // Create transformer blocks with residual scaling
    for (int i = 0; i < num_layers; i++) {
        blocks.push_back(new TransformerBlock(
            d_model, num_heads, d_ff, max_batch_size, max_seq_len, this->residual_scale
        ));
    }

    initialize_parameters();

    // Initialize gradient and optimizer state pointers to nullptr
    d_grad_token_embeddings = nullptr;
    d_grad_position_embeddings = nullptr;
    d_grad_output_weights = nullptr;
    d_grad_output_bias = nullptr;
    d_grad_ln_final_gamma = nullptr;
    d_grad_ln_final_beta = nullptr;

    d_m_token_embeddings = nullptr;
    d_v_token_embeddings = nullptr;
    d_m_position_embeddings = nullptr;
    d_v_position_embeddings = nullptr;
    d_m_output_weights = nullptr;
    d_v_output_weights = nullptr;
    d_m_output_bias = nullptr;
    d_v_output_bias = nullptr;
    d_m_ln_final_gamma = nullptr;
    d_v_ln_final_gamma = nullptr;
    d_m_ln_final_beta = nullptr;
    d_v_ln_final_beta = nullptr;
}

Transformer::~Transformer() {
    for (auto block : blocks) {
        delete block;
    }
    free_gradient_buffers();
    free_optimizer_state();
    free_memory();
}

void Transformer::allocate_memory() {
    // Embeddings
    CUDA_CHECK(cudaMalloc(&d_token_embeddings, vocab_size * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_position_embeddings, max_seq_len * d_model * sizeof(float)));

    // Final layer norm
    CUDA_CHECK(cudaMalloc(&d_ln_final_gamma, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_final_beta, d_model * sizeof(float)));

    // Output projection
    CUDA_CHECK(cudaMalloc(&d_output_weights, d_model * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_bias, vocab_size * sizeof(float)));

    // Buffers
    CUDA_CHECK(cudaMalloc(&d_embeddings, max_batch_size * max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_output, max_batch_size * max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normed, max_batch_size * max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_causal_mask, max_seq_len * max_seq_len * sizeof(float)));

    // Generation buffers
    CUDA_CHECK(cudaMalloc(&d_token_ids, max_batch_size * max_seq_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_logits_buffer, max_batch_size * max_seq_len * vocab_size * sizeof(float)));
}

void Transformer::free_memory() {
    cudaFree(d_token_embeddings);
    cudaFree(d_position_embeddings);
    cudaFree(d_ln_final_gamma);
    cudaFree(d_ln_final_beta);
    cudaFree(d_output_weights);
    cudaFree(d_output_bias);
    cudaFree(d_embeddings);
    cudaFree(d_block_output);
    cudaFree(d_normed);
    cudaFree(d_causal_mask);
    cudaFree(d_token_ids);
    cudaFree(d_logits_buffer);
}

void Transformer::initialize_parameters() {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Token embeddings: N(0, 1/sqrt(d_model))
    float emb_std = 1.0f / sqrtf(d_model);
    curandGenerateNormal(gen, d_token_embeddings, vocab_size * d_model, 0.0f, emb_std);

    // Position embeddings: use sinusoidal (computed on host)
    // Scale by sqrt(d_model) to match token embedding magnitude
    float* h_pos_emb = new float[max_seq_len * d_model];
    float pos_scale = sqrtf((float)d_model);  // Scale sinusoids to match token emb std
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < d_model / 2; i++) {
            float angle = (float)pos / powf(10000.0f, 2.0f * i / d_model);
            h_pos_emb[pos * d_model + 2 * i] = sinf(angle) / pos_scale;
            h_pos_emb[pos * d_model + 2 * i + 1] = cosf(angle) / pos_scale;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_position_embeddings, h_pos_emb,
                         max_seq_len * d_model * sizeof(float),
                         cudaMemcpyHostToDevice));
    delete[] h_pos_emb;

    // Final layer norm (gamma=1, beta=0)
    float* h_ones = new float[d_model];
    float* h_zeros = new float[d_model];
    for (int i = 0; i < d_model; i++) {
        h_ones[i] = 1.0f;
        h_zeros[i] = 0.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_ln_final_gamma, h_ones, d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln_final_beta, h_zeros, d_model * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_ones;
    delete[] h_zeros;

    // Output projection: Xavier initialization
    float out_std = sqrtf(2.0f / (d_model + vocab_size));
    curandGenerateNormal(gen, d_output_weights, d_model * vocab_size, 0.0f, out_std);
    CUDA_CHECK(cudaMemset(d_output_bias, 0, vocab_size * sizeof(float)));

    // Create causal mask
    create_causal_mask(d_causal_mask, max_seq_len);

    curandDestroyGenerator(gen);
}

void Transformer::forward_device(
    const int* d_token_ids,
    float* d_logits,
    int batch_size,
    int seq_len
) {
    // 1. Embed tokens
    embedding_forward(d_token_ids, d_token_embeddings, d_embeddings,
                     batch_size, seq_len, vocab_size, d_model);

    // 2. Add positional embeddings
    dim3 block_size(d_model);
    dim3 grid_size(seq_len, batch_size);
    add_position_embeddings_kernel<<<grid_size, block_size>>>(
        d_embeddings, d_position_embeddings, d_embeddings,
        batch_size, seq_len, d_model
    );
    CUDA_CHECK(cudaGetLastError());

    // 3. Pass through transformer blocks
    float* current_input = d_embeddings;
    float* current_output = d_block_output;

    for (int i = 0; i < num_layers; i++) {
        blocks[i]->forward_device(current_input, current_output,
                                  d_causal_mask, batch_size, seq_len);

        // Swap buffers for next layer
        float* temp = current_input;
        current_input = current_output;
        current_output = temp;
    }

    // After all blocks, current_input contains the final output
    // 4. Final layer norm
    layer_norm(current_input, d_normed, d_ln_final_gamma, d_ln_final_beta,
               batch_size, seq_len, d_model);

    // 5. Project to vocabulary: logits = normed · W^T + b
    int total_tokens = batch_size * seq_len;
    matmul_transB(d_normed, d_output_weights, d_logits,
                  total_tokens, d_model, vocab_size);
    add_bias(d_logits, d_output_bias, d_logits, total_tokens, vocab_size);
}

void Transformer::forward(
    const int* h_token_ids,
    float* h_logits,
    int batch_size,
    int seq_len
) {
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_token_ids, h_token_ids,
                         batch_size * seq_len * sizeof(int),
                         cudaMemcpyHostToDevice));

    // Forward pass
    forward_device(d_token_ids, d_logits_buffer, batch_size, seq_len);

    // Copy output to host
    CUDA_CHECK(cudaMemcpy(h_logits, d_logits_buffer,
                         batch_size * seq_len * vocab_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

int Transformer::sample_token(
    const float* logits,
    int vocab_size,
    float temperature,
    int top_k,
    float top_p,
    unsigned int& rng_state
) {
    // Simple implementation using host memory
    // For production, this should be done on GPU

    std::vector<float> probs(vocab_size);

    // Apply temperature
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        max_logit = std::max(max_logit, logits[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }

    // Normalize
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }

    // Top-k filtering
    if (top_k > 0 && top_k < vocab_size) {
        std::vector<std::pair<float, int>> prob_idx;
        for (int i = 0; i < vocab_size; i++) {
            prob_idx.push_back({probs[i], i});
        }
        std::partial_sort(prob_idx.begin(), prob_idx.begin() + top_k, prob_idx.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        // Zero out probabilities outside top-k
        std::vector<float> filtered(vocab_size, 0.0f);
        float filtered_sum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            filtered[prob_idx[i].second] = prob_idx[i].first;
            filtered_sum += prob_idx[i].first;
        }

        // Renormalize
        for (int i = 0; i < vocab_size; i++) {
            probs[i] = filtered[i] / filtered_sum;
        }
    }

    // Top-p (nucleus) filtering
    if (top_p < 1.0f) {
        std::vector<std::pair<float, int>> prob_idx;
        for (int i = 0; i < vocab_size; i++) {
            if (probs[i] > 0) {
                prob_idx.push_back({probs[i], i});
            }
        }
        std::sort(prob_idx.begin(), prob_idx.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });

        float cumsum = 0.0f;
        std::vector<float> filtered(vocab_size, 0.0f);
        float filtered_sum = 0.0f;

        for (const auto& p : prob_idx) {
            cumsum += p.first;
            filtered[p.second] = p.first;
            filtered_sum += p.first;
            if (cumsum >= top_p) break;
        }

        // Renormalize
        for (int i = 0; i < vocab_size; i++) {
            probs[i] = filtered[i] / filtered_sum;
        }
    }

    // Sample from distribution
    // Simple LCG random number generator
    rng_state = rng_state * 1103515245 + 12345;
    float rand_val = ((rng_state / 65536) % 32768) / 32768.0f;

    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (rand_val < cumsum) {
            return i;
        }
    }

    return vocab_size - 1;  // Fallback
}

std::vector<int> Transformer::generate(
    const std::vector<int>& prompt,
    int max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    int seed
) {
    std::vector<int> result = prompt;
    unsigned int rng_state = seed;

    float* h_logits = new float[max_seq_len * vocab_size];

    for (int i = 0; i < max_new_tokens; i++) {
        // Get current sequence length
        int seq_len = result.size();

        // Truncate to max_seq_len if needed
        int start_idx = seq_len > max_seq_len ? seq_len - max_seq_len : 0;
        std::vector<int> context(result.begin() + start_idx, result.end());
        seq_len = context.size();

        // Forward pass
        forward(context.data(), h_logits, 1, seq_len);

        // Get logits for last position
        float* last_logits = h_logits + (seq_len - 1) * vocab_size;

        // Sample next token
        int next_token = sample_token(last_logits, vocab_size, temperature,
                                     top_k, top_p, rng_state);

        result.push_back(next_token);
    }

    delete[] h_logits;
    return result;
}

void Transformer::generate_batch(
    const std::vector<std::vector<int>>& prompts,
    std::vector<std::vector<int>>& outputs,
    int max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    int seed
) {
    // For simplicity, generate one at a time
    // TODO: Implement true batched generation with padding
    outputs.clear();
    for (size_t i = 0; i < prompts.size(); i++) {
        outputs.push_back(generate(prompts[i], max_new_tokens, temperature,
                                  top_k, top_p, seed + i));
    }
}

float Transformer::compute_loss(
    const int* h_token_ids,
    const int* h_targets,
    int batch_size,
    int seq_len
) {
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_token_ids, h_token_ids,
                         batch_size * seq_len * sizeof(int),
                         cudaMemcpyHostToDevice));

    int* d_targets;
    CUDA_CHECK(cudaMalloc(&d_targets, batch_size * seq_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets,
                         batch_size * seq_len * sizeof(int),
                         cudaMemcpyHostToDevice));

    // Forward pass to get logits
    forward_device(d_token_ids, d_logits_buffer, batch_size, seq_len);

    // Compute loss
    float loss = lm_cross_entropy_loss(d_logits_buffer, d_targets,
                                       batch_size, seq_len, vocab_size);

    CUDA_CHECK(cudaFree(d_targets));
    return loss;
}

void Transformer::save_parameters(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    // Save architecture
    file.write(reinterpret_cast<char*>(&vocab_size), sizeof(int));
    file.write(reinterpret_cast<char*>(&d_model), sizeof(int));
    file.write(reinterpret_cast<char*>(&num_layers), sizeof(int));
    file.write(reinterpret_cast<char*>(&num_heads), sizeof(int));
    file.write(reinterpret_cast<char*>(&d_ff), sizeof(int));
    file.write(reinterpret_cast<char*>(&max_seq_len), sizeof(int));

    auto save_param = [&](float* d_param, int size) {
        float* h_param = new float[size];
        CUDA_CHECK(cudaMemcpy(h_param, d_param, size * sizeof(float),
                             cudaMemcpyDeviceToHost));
        file.write(reinterpret_cast<char*>(h_param), size * sizeof(float));
        delete[] h_param;
    };

    // Save embeddings and parameters
    save_param(d_token_embeddings, vocab_size * d_model);
    save_param(d_position_embeddings, max_seq_len * d_model);
    save_param(d_ln_final_gamma, d_model);
    save_param(d_ln_final_beta, d_model);
    save_param(d_output_weights, d_model * vocab_size);
    save_param(d_output_bias, vocab_size);

    file.close();

    // Save transformer blocks
    std::string base(filename);
    for (int i = 0; i < num_layers; i++) {
        std::string block_file = base + ".block" + std::to_string(i);
        blocks[i]->save_parameters(block_file.c_str());
    }
}

void Transformer::load_parameters(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open %s for reading\n", filename);
        return;
    }

    // Load and verify architecture
    int loaded_vocab, loaded_d_model, loaded_layers, loaded_heads, loaded_d_ff, loaded_seq_len;
    file.read(reinterpret_cast<char*>(&loaded_vocab), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_d_model), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_layers), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_heads), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_d_ff), sizeof(int));
    file.read(reinterpret_cast<char*>(&loaded_seq_len), sizeof(int));

    if (loaded_vocab != vocab_size || loaded_d_model != d_model ||
        loaded_layers != num_layers || loaded_heads != num_heads ||
        loaded_d_ff != d_ff || loaded_seq_len != max_seq_len) {
        fprintf(stderr, "Architecture mismatch in loaded model\n");
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

    // Load embeddings and parameters
    load_param(d_token_embeddings, vocab_size * d_model);
    load_param(d_position_embeddings, max_seq_len * d_model);
    load_param(d_ln_final_gamma, d_model);
    load_param(d_ln_final_beta, d_model);
    load_param(d_output_weights, d_model * vocab_size);
    load_param(d_output_bias, vocab_size);

    file.close();

    // Load transformer blocks
    std::string base(filename);
    for (int i = 0; i < num_layers; i++) {
        std::string block_file = base + ".block" + std::to_string(i);
        blocks[i]->load_parameters(block_file.c_str());
    }
}

// ============================================================================
// Training Implementation
// ============================================================================

void Transformer::allocate_gradient_buffers() {
    // Allocate gradient buffers for embeddings and output projection
    CUDA_CHECK(cudaMalloc(&d_grad_token_embeddings, vocab_size * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_position_embeddings, max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output_weights, d_model * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output_bias, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_ln_final_gamma, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_ln_final_beta, d_model * sizeof(float)));

    // Zero out gradient buffers
    CUDA_CHECK(cudaMemset(d_grad_token_embeddings, 0, vocab_size * d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_position_embeddings, 0, max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_output_weights, 0, d_model * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_output_bias, 0, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_ln_final_gamma, 0, d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_ln_final_beta, 0, d_model * sizeof(float)));

    // Allocate gradient buffers for each TransformerBlock
    block_grads.clear();
    for (int i = 0; i < num_layers; i++) {
        BlockGradients grads;

        // Layer norm gradients
        CUDA_CHECK(cudaMalloc(&grads.d_grad_ln1_gamma, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_ln1_beta, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_ln2_gamma, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_ln2_beta, d_model * sizeof(float)));

        // Attention gradients
        CUDA_CHECK(cudaMalloc(&grads.d_grad_attn_W_Q, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_attn_b_Q, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_attn_W_K, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_attn_b_K, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_attn_W_V, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_attn_b_V, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_attn_W_O, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_attn_b_O, d_model * sizeof(float)));

        // FFN gradients
        CUDA_CHECK(cudaMalloc(&grads.d_grad_ffn_W1, d_ff * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_ffn_b1, d_ff * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_ffn_W2, d_model * d_ff * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&grads.d_grad_ffn_b2, d_model * sizeof(float)));

        block_grads.push_back(grads);
    }
}

void Transformer::free_gradient_buffers() {
    if (d_grad_token_embeddings) cudaFree(d_grad_token_embeddings);
    if (d_grad_position_embeddings) cudaFree(d_grad_position_embeddings);
    if (d_grad_output_weights) cudaFree(d_grad_output_weights);
    if (d_grad_output_bias) cudaFree(d_grad_output_bias);
    if (d_grad_ln_final_gamma) cudaFree(d_grad_ln_final_gamma);
    if (d_grad_ln_final_beta) cudaFree(d_grad_ln_final_beta);

    d_grad_token_embeddings = nullptr;
    d_grad_position_embeddings = nullptr;
    d_grad_output_weights = nullptr;
    d_grad_output_bias = nullptr;
    d_grad_ln_final_gamma = nullptr;
    d_grad_ln_final_beta = nullptr;

    // Free block gradient buffers
    for (auto& grads : block_grads) {
        cudaFree(grads.d_grad_ln1_gamma);
        cudaFree(grads.d_grad_ln1_beta);
        cudaFree(grads.d_grad_ln2_gamma);
        cudaFree(grads.d_grad_ln2_beta);
        cudaFree(grads.d_grad_attn_W_Q);
        cudaFree(grads.d_grad_attn_b_Q);
        cudaFree(grads.d_grad_attn_W_K);
        cudaFree(grads.d_grad_attn_b_K);
        cudaFree(grads.d_grad_attn_W_V);
        cudaFree(grads.d_grad_attn_b_V);
        cudaFree(grads.d_grad_attn_W_O);
        cudaFree(grads.d_grad_attn_b_O);
        cudaFree(grads.d_grad_ffn_W1);
        cudaFree(grads.d_grad_ffn_b1);
        cudaFree(grads.d_grad_ffn_W2);
        cudaFree(grads.d_grad_ffn_b2);
    }
    block_grads.clear();
}

void Transformer::allocate_optimizer_state() {
    // Allocate Adam state for embeddings and output projection
    CUDA_CHECK(cudaMalloc(&d_m_token_embeddings, vocab_size * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_token_embeddings, vocab_size * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_position_embeddings, max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_position_embeddings, max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_output_weights, d_model * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_output_weights, d_model * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_output_bias, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_output_bias, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_ln_final_gamma, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_ln_final_gamma, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_ln_final_beta, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_ln_final_beta, d_model * sizeof(float)));

    // Zero out Adam state
    CUDA_CHECK(cudaMemset(d_m_token_embeddings, 0, vocab_size * d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_token_embeddings, 0, vocab_size * d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_position_embeddings, 0, max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_position_embeddings, 0, max_seq_len * d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_output_weights, 0, d_model * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_output_weights, 0, d_model * vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_output_bias, 0, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_output_bias, 0, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_ln_final_gamma, 0, d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_ln_final_gamma, 0, d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_ln_final_beta, 0, d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_ln_final_beta, 0, d_model * sizeof(float)));

    // Allocate Adam state for each TransformerBlock
    block_optim_state.clear();
    for (int i = 0; i < num_layers; i++) {
        BlockOptimState state;

        // Layer norm
        CUDA_CHECK(cudaMalloc(&state.d_m_ln1_gamma, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_ln1_gamma, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_ln1_beta, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_ln1_beta, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_ln2_gamma, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_ln2_gamma, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_ln2_beta, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_ln2_beta, d_model * sizeof(float)));

        // Attention
        CUDA_CHECK(cudaMalloc(&state.d_m_attn_W_Q, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_attn_W_Q, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_attn_b_Q, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_attn_b_Q, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_attn_W_K, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_attn_W_K, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_attn_b_K, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_attn_b_K, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_attn_W_V, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_attn_W_V, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_attn_b_V, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_attn_b_V, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_attn_W_O, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_attn_W_O, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_attn_b_O, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_attn_b_O, d_model * sizeof(float)));

        // FFN
        CUDA_CHECK(cudaMalloc(&state.d_m_ffn_W1, d_ff * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_ffn_W1, d_ff * d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_ffn_b1, d_ff * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_ffn_b1, d_ff * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_ffn_W2, d_model * d_ff * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_ffn_W2, d_model * d_ff * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_m_ffn_b2, d_model * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&state.d_v_ffn_b2, d_model * sizeof(float)));

        // Zero initialize
        CUDA_CHECK(cudaMemset(state.d_m_ln1_gamma, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_ln1_gamma, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_ln1_beta, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_ln1_beta, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_ln2_gamma, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_ln2_gamma, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_ln2_beta, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_ln2_beta, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_attn_W_Q, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_attn_W_Q, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_attn_b_Q, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_attn_b_Q, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_attn_W_K, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_attn_W_K, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_attn_b_K, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_attn_b_K, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_attn_W_V, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_attn_W_V, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_attn_b_V, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_attn_b_V, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_attn_W_O, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_attn_W_O, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_attn_b_O, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_attn_b_O, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_ffn_W1, 0, d_ff * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_ffn_W1, 0, d_ff * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_ffn_b1, 0, d_ff * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_ffn_b1, 0, d_ff * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_ffn_W2, 0, d_model * d_ff * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_ffn_W2, 0, d_model * d_ff * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_m_ffn_b2, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(state.d_v_ffn_b2, 0, d_model * sizeof(float)));

        block_optim_state.push_back(state);
    }
}

void Transformer::free_optimizer_state() {
    if (d_m_token_embeddings) cudaFree(d_m_token_embeddings);
    if (d_v_token_embeddings) cudaFree(d_v_token_embeddings);
    if (d_m_position_embeddings) cudaFree(d_m_position_embeddings);
    if (d_v_position_embeddings) cudaFree(d_v_position_embeddings);
    if (d_m_output_weights) cudaFree(d_m_output_weights);
    if (d_v_output_weights) cudaFree(d_v_output_weights);
    if (d_m_output_bias) cudaFree(d_m_output_bias);
    if (d_v_output_bias) cudaFree(d_v_output_bias);
    if (d_m_ln_final_gamma) cudaFree(d_m_ln_final_gamma);
    if (d_v_ln_final_gamma) cudaFree(d_v_ln_final_gamma);
    if (d_m_ln_final_beta) cudaFree(d_m_ln_final_beta);
    if (d_v_ln_final_beta) cudaFree(d_v_ln_final_beta);

    d_m_token_embeddings = nullptr;
    d_v_token_embeddings = nullptr;
    d_m_position_embeddings = nullptr;
    d_v_position_embeddings = nullptr;
    d_m_output_weights = nullptr;
    d_v_output_weights = nullptr;
    d_m_output_bias = nullptr;
    d_v_output_bias = nullptr;
    d_m_ln_final_gamma = nullptr;
    d_v_ln_final_gamma = nullptr;
    d_m_ln_final_beta = nullptr;
    d_v_ln_final_beta = nullptr;

    // Free block optimizer state
    for (auto& state : block_optim_state) {
        cudaFree(state.d_m_ln1_gamma);
        cudaFree(state.d_v_ln1_gamma);
        cudaFree(state.d_m_ln1_beta);
        cudaFree(state.d_v_ln1_beta);
        cudaFree(state.d_m_ln2_gamma);
        cudaFree(state.d_v_ln2_gamma);
        cudaFree(state.d_m_ln2_beta);
        cudaFree(state.d_v_ln2_beta);
        cudaFree(state.d_m_attn_W_Q);
        cudaFree(state.d_v_attn_W_Q);
        cudaFree(state.d_m_attn_b_Q);
        cudaFree(state.d_v_attn_b_Q);
        cudaFree(state.d_m_attn_W_K);
        cudaFree(state.d_v_attn_W_K);
        cudaFree(state.d_m_attn_b_K);
        cudaFree(state.d_v_attn_b_K);
        cudaFree(state.d_m_attn_W_V);
        cudaFree(state.d_v_attn_W_V);
        cudaFree(state.d_m_attn_b_V);
        cudaFree(state.d_v_attn_b_V);
        cudaFree(state.d_m_attn_W_O);
        cudaFree(state.d_v_attn_W_O);
        cudaFree(state.d_m_attn_b_O);
        cudaFree(state.d_v_attn_b_O);
        cudaFree(state.d_m_ffn_W1);
        cudaFree(state.d_v_ffn_W1);
        cudaFree(state.d_m_ffn_b1);
        cudaFree(state.d_v_ffn_b1);
        cudaFree(state.d_m_ffn_W2);
        cudaFree(state.d_v_ffn_W2);
        cudaFree(state.d_m_ffn_b2);
        cudaFree(state.d_v_ffn_b2);
    }
    block_optim_state.clear();
}

void Transformer::backward(
    const int* d_token_ids,
    const int* d_targets,
    int batch_size,
    int seq_len
) {
    // NOTE: Forward pass must have been called before this!
    // The forward pass fills d_block_output, d_normed, etc.

    int total_tokens = batch_size * seq_len;
    int total_logits = total_tokens * vocab_size;

    // Allocate gradient buffers if not already allocated
    if (d_grad_token_embeddings == nullptr) {
        allocate_gradient_buffers();
    } else {
        // Zero out gradients for this iteration
        CUDA_CHECK(cudaMemset(d_grad_token_embeddings, 0, vocab_size * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_position_embeddings, 0, max_seq_len * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_output_weights, 0, d_model * vocab_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_output_bias, 0, vocab_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_ln_final_gamma, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_ln_final_beta, 0, d_model * sizeof(float)));
    }

    // Temporary gradient buffers
    float *d_grad_logits, *d_grad_normed, *d_grad_block_out;
    float *d_grad_embeddings;

    CUDA_CHECK(cudaMalloc(&d_grad_logits, total_logits * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_normed, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_block_out, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_embeddings, total_tokens * d_model * sizeof(float)));

    // ========================================================================
    // 1. Compute gradient of loss w.r.t. logits
    // ========================================================================

    lm_cross_entropy_gradient(
        d_logits_buffer,  // logits [batch_size, seq_len, vocab_size]
        d_targets,         // targets [batch_size, seq_len]
        d_grad_logits,     // gradient output
        batch_size, seq_len, vocab_size,
        nullptr            // no mask
    );

    // ========================================================================
    // 2. Backward through output projection: logits = normed · W_out^T + b_out
    // ========================================================================

    sum_rows(d_grad_logits, d_grad_output_bias, total_tokens, vocab_size);
    matmul_transB_backward(
        d_grad_logits,        // grad_C [total_tokens x vocab_size]
        d_normed,             // A [total_tokens x d_model]
        d_output_weights,     // B [vocab_size x d_model]
        d_grad_normed,        // grad_A [total_tokens x d_model]
        d_grad_output_weights, // grad_B [vocab_size x d_model]
        total_tokens,         // M
        d_model,              // K
        vocab_size            // N
    );

    // ========================================================================
    // 3. Backward through final layer norm
    // ========================================================================

    layer_norm_backward(
        d_grad_normed,           // Gradient w.r.t. output
        d_block_output,          // Input (last block output)
        d_ln_final_gamma,        // Gamma
        d_grad_block_out,        // Gradient w.r.t. input
        d_grad_ln_final_gamma,   // Gradient w.r.t. gamma
        d_grad_ln_final_beta,    // Gradient w.r.t. beta
        batch_size, seq_len, d_model,
        1e-5f
    );

    // ========================================================================
    // 4. Backward through transformer blocks (in reverse order)
    // ========================================================================

    // We need to reconstruct which buffer contains the output of each layer
    // In forward: d_embeddings -> block0 -> buffer1 -> block1 -> buffer2 -> ...
    // The buffers alternate between d_embeddings and d_block_output

    // Allocate buffers for intermediate gradients during block backward
    float *d_grad_block_input, *d_grad_block_input_temp;
    CUDA_CHECK(cudaMalloc(&d_grad_block_input, total_tokens * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_block_input_temp, total_tokens * d_model * sizeof(float)));

    // Start with gradient w.r.t. the output of the last block
    // After final layer norm backward, d_grad_block_out contains this gradient
    CUDA_CHECK(cudaMemcpy(d_grad_block_input, d_grad_block_out,
                         total_tokens * d_model * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // Zero out block gradient buffers
    for (int i = 0; i < num_layers; i++) {
        auto& grads = block_grads[i];
        CUDA_CHECK(cudaMemset(grads.d_grad_ln1_gamma, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_ln1_beta, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_ln2_gamma, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_ln2_beta, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_attn_W_Q, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_attn_b_Q, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_attn_W_K, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_attn_b_K, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_attn_W_V, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_attn_b_V, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_attn_W_O, 0, d_model * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_attn_b_O, 0, d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_ffn_W1, 0, d_ff * d_model * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_ffn_b1, 0, d_ff * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_ffn_W2, 0, d_model * d_ff * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.d_grad_ffn_b2, 0, d_model * sizeof(float)));
    }

    // Backward through each block in reverse order
    // We need to track the input to each block from the forward pass
    // For simplicity, we'll re-run a minimal forward to get the inputs
    // (In production, these would be cached during forward)

    // Allocate buffer to store block inputs from forward pass
    std::vector<float*> block_inputs(num_layers + 1);
    block_inputs[0] = d_embeddings;  // Input to first block is embeddings

    // Allocate buffers for intermediate block outputs
    for (int i = 1; i <= num_layers; i++) {
        CUDA_CHECK(cudaMalloc(&block_inputs[i], total_tokens * d_model * sizeof(float)));
    }

    // Re-run forward to cache block inputs (TODO: optimize by caching during forward)
    float* fwd_input = d_embeddings;
    float* fwd_output = block_inputs[1];
    for (int i = 0; i < num_layers; i++) {
        blocks[i]->forward_device(fwd_input, fwd_output,
                                  d_causal_mask, batch_size, seq_len);
        if (i + 1 < num_layers) {
            fwd_input = fwd_output;
            fwd_output = block_inputs[i + 2];
        }
    }

    // Now run backward through blocks in reverse
    for (int i = num_layers - 1; i >= 0; i--) {
        auto& grads = block_grads[i];

        // Call block backward
        blocks[i]->backward_device(
            block_inputs[i],          // Input to this block from forward pass
            d_grad_block_input,       // Gradient w.r.t. block output
            d_grad_block_input_temp,  // Gradient w.r.t. block input (output)
            grads.d_grad_ln1_gamma, grads.d_grad_ln1_beta,
            grads.d_grad_ln2_gamma, grads.d_grad_ln2_beta,
            grads.d_grad_attn_W_Q, grads.d_grad_attn_b_Q,
            grads.d_grad_attn_W_K, grads.d_grad_attn_b_K,
            grads.d_grad_attn_W_V, grads.d_grad_attn_b_V,
            grads.d_grad_attn_W_O, grads.d_grad_attn_b_O,
            grads.d_grad_ffn_W1, grads.d_grad_ffn_b1,
            grads.d_grad_ffn_W2, grads.d_grad_ffn_b2,
            batch_size, seq_len
        );

        // Swap gradient buffers for next iteration
        float* temp = d_grad_block_input;
        d_grad_block_input = d_grad_block_input_temp;
        d_grad_block_input_temp = temp;
    }

    // After all blocks, d_grad_block_input contains gradient w.r.t. embeddings
    CUDA_CHECK(cudaMemcpy(d_grad_embeddings, d_grad_block_input,
                         total_tokens * d_model * sizeof(float),
                         cudaMemcpyDeviceToDevice));

    // ========================================================================
    // 5. Backward through embeddings
    // ========================================================================

    // Backward through token embedding lookup
    // d_grad_embeddings contains gradient w.r.t. (token_emb + pos_emb)
    // This gradient flows to both token and position embeddings
    embedding_backward(d_grad_embeddings, d_token_ids, d_grad_token_embeddings,
                      batch_size, seq_len, vocab_size, d_model);

    // Accumulate position embedding gradients
    // Position embeddings are added element-wise at each position
    // So we need to accumulate gradients for each position across the batch
    dim3 grid_size_pos(seq_len, (d_model + 255) / 256);
    dim3 block_size_pos(256);
    accumulate_position_gradients_kernel<<<grid_size_pos, block_size_pos>>>(
        d_grad_embeddings, d_grad_position_embeddings,
        batch_size, seq_len, d_model
    );
    CUDA_CHECK(cudaGetLastError());

    // Cleanup temporary buffers
    for (int i = 1; i <= num_layers; i++) {
        cudaFree(block_inputs[i]);
    }
    CUDA_CHECK(cudaFree(d_grad_block_input));
    CUDA_CHECK(cudaFree(d_grad_block_input_temp));
    CUDA_CHECK(cudaFree(d_grad_logits));
    CUDA_CHECK(cudaFree(d_grad_normed));
    CUDA_CHECK(cudaFree(d_grad_block_out));
    CUDA_CHECK(cudaFree(d_grad_embeddings));
}

float Transformer::train_step(
    const int* h_token_ids,
    const int* h_targets,
    int batch_size,
    int seq_len,
    float learning_rate,
    float grad_clip_norm
) {
    // Copy inputs to device
    int* d_token_ids_train;
    int* d_targets_train;
    CUDA_CHECK(cudaMalloc(&d_token_ids_train, batch_size * seq_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_targets_train, batch_size * seq_len * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_token_ids_train, h_token_ids, batch_size * seq_len * sizeof(int),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets_train, h_targets, batch_size * seq_len * sizeof(int),
                         cudaMemcpyHostToDevice));

    // Forward pass
    forward_device(d_token_ids_train, d_logits_buffer, batch_size, seq_len);

    // Compute loss
    float loss = lm_cross_entropy_loss(
        d_logits_buffer, d_targets_train,
        batch_size, seq_len, vocab_size,
        nullptr
    );

    // Backward pass
    backward(d_token_ids_train, d_targets_train, batch_size, seq_len);

    // Gradient clipping
    if (grad_clip_norm > 0.0f) {
        clip_all_gradients(grad_clip_norm);
    }

    // ========================================================================
    // Apply Adam optimizer updates
    // ========================================================================

    // Allocate optimizer state if not already allocated
    if (d_m_token_embeddings == nullptr) {
        allocate_optimizer_state();
    }

    // Increment training step and compute bias correction terms
    training_step++;
    float beta1_t = powf(beta1, training_step);
    float beta2_t = powf(beta2, training_step);

    // Update embeddings and output projection
    adam_update(d_token_embeddings, d_grad_token_embeddings,
               d_m_token_embeddings, d_v_token_embeddings,
               learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
               vocab_size * d_model);

    adam_update(d_position_embeddings, d_grad_position_embeddings,
               d_m_position_embeddings, d_v_position_embeddings,
               learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
               max_seq_len * d_model);

    adam_update(d_output_weights, d_grad_output_weights,
               d_m_output_weights, d_v_output_weights,
               learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
               d_model * vocab_size);

    adam_update(d_output_bias, d_grad_output_bias,
               d_m_output_bias, d_v_output_bias,
               learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
               vocab_size);

    adam_update(d_ln_final_gamma, d_grad_ln_final_gamma,
               d_m_ln_final_gamma, d_v_ln_final_gamma,
               learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
               d_model);

    adam_update(d_ln_final_beta, d_grad_ln_final_beta,
               d_m_ln_final_beta, d_v_ln_final_beta,
               learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
               d_model);

    // Update each TransformerBlock's parameters
    for (int i = 0; i < num_layers; i++) {
        auto& grads = block_grads[i];
        auto& optim = block_optim_state[i];
        TransformerBlock* block = blocks[i];

        // Layer norm 1
        adam_update(block->d_ln1_gamma, grads.d_grad_ln1_gamma,
                   optim.d_m_ln1_gamma, optim.d_v_ln1_gamma,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model);
        adam_update(block->d_ln1_beta, grads.d_grad_ln1_beta,
                   optim.d_m_ln1_beta, optim.d_v_ln1_beta,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model);

        // Layer norm 2
        adam_update(block->d_ln2_gamma, grads.d_grad_ln2_gamma,
                   optim.d_m_ln2_gamma, optim.d_v_ln2_gamma,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model);
        adam_update(block->d_ln2_beta, grads.d_grad_ln2_beta,
                   optim.d_m_ln2_beta, optim.d_v_ln2_beta,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model);

        // Attention parameters
        adam_update(block->attention->d_W_Q, grads.d_grad_attn_W_Q,
                   optim.d_m_attn_W_Q, optim.d_v_attn_W_Q,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model * d_model);
        adam_update(block->attention->d_b_Q, grads.d_grad_attn_b_Q,
                   optim.d_m_attn_b_Q, optim.d_v_attn_b_Q,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model);

        adam_update(block->attention->d_W_K, grads.d_grad_attn_W_K,
                   optim.d_m_attn_W_K, optim.d_v_attn_W_K,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model * d_model);
        adam_update(block->attention->d_b_K, grads.d_grad_attn_b_K,
                   optim.d_m_attn_b_K, optim.d_v_attn_b_K,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model);

        adam_update(block->attention->d_W_V, grads.d_grad_attn_W_V,
                   optim.d_m_attn_W_V, optim.d_v_attn_W_V,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model * d_model);
        adam_update(block->attention->d_b_V, grads.d_grad_attn_b_V,
                   optim.d_m_attn_b_V, optim.d_v_attn_b_V,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model);

        adam_update(block->attention->d_W_O, grads.d_grad_attn_W_O,
                   optim.d_m_attn_W_O, optim.d_v_attn_W_O,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model * d_model);
        adam_update(block->attention->d_b_O, grads.d_grad_attn_b_O,
                   optim.d_m_attn_b_O, optim.d_v_attn_b_O,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model);

        // FFN parameters
        adam_update(block->ffn->d_W1, grads.d_grad_ffn_W1,
                   optim.d_m_ffn_W1, optim.d_v_ffn_W1,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_ff * d_model);
        adam_update(block->ffn->d_b1, grads.d_grad_ffn_b1,
                   optim.d_m_ffn_b1, optim.d_v_ffn_b1,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_ff);

        adam_update(block->ffn->d_W2, grads.d_grad_ffn_W2,
                   optim.d_m_ffn_W2, optim.d_v_ffn_W2,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model * d_ff);
        adam_update(block->ffn->d_b2, grads.d_grad_ffn_b2,
                   optim.d_m_ffn_b2, optim.d_v_ffn_b2,
                   learning_rate, beta1, beta2, epsilon, beta1_t, beta2_t,
                   d_model);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_token_ids_train));
    CUDA_CHECK(cudaFree(d_targets_train));

    return loss;
}

float Transformer::compute_gradient_norm() {
    // Compute total gradient norm across all parameters
    float total_norm_sq = 0.0f;

    // Embeddings and output
    float norm = gradient_norm(d_grad_token_embeddings, vocab_size * d_model);
    total_norm_sq += norm * norm;

    norm = gradient_norm(d_grad_position_embeddings, max_seq_len * d_model);
    total_norm_sq += norm * norm;

    norm = gradient_norm(d_grad_output_weights, d_model * vocab_size);
    total_norm_sq += norm * norm;

    norm = gradient_norm(d_grad_output_bias, vocab_size);
    total_norm_sq += norm * norm;

    norm = gradient_norm(d_grad_ln_final_gamma, d_model);
    total_norm_sq += norm * norm;

    norm = gradient_norm(d_grad_ln_final_beta, d_model);
    total_norm_sq += norm * norm;

    // Block gradients
    for (int i = 0; i < num_layers; i++) {
        auto& grads = block_grads[i];

        norm = gradient_norm(grads.d_grad_ln1_gamma, d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_ln1_beta, d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_ln2_gamma, d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_ln2_beta, d_model);
        total_norm_sq += norm * norm;

        // Attention
        norm = gradient_norm(grads.d_grad_attn_W_Q, d_model * d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_attn_b_Q, d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_attn_W_K, d_model * d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_attn_b_K, d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_attn_W_V, d_model * d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_attn_b_V, d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_attn_W_O, d_model * d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_attn_b_O, d_model);
        total_norm_sq += norm * norm;

        // FFN
        norm = gradient_norm(grads.d_grad_ffn_W1, d_ff * d_model);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_ffn_b1, d_ff);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_ffn_W2, d_model * d_ff);
        total_norm_sq += norm * norm;

        norm = gradient_norm(grads.d_grad_ffn_b2, d_model);
        total_norm_sq += norm * norm;
    }

    return sqrtf(total_norm_sq);
}

void Transformer::clip_all_gradients(float max_norm) {
    float total_norm = compute_gradient_norm();

    if (total_norm > max_norm) {
        // Compute single scale factor for ALL gradients
        float scale = max_norm / total_norm;

        // Scale all gradients by this single factor (global norm clipping)
        scale_gradients(d_grad_token_embeddings, vocab_size * d_model, scale);
        scale_gradients(d_grad_position_embeddings, max_seq_len * d_model, scale);
        scale_gradients(d_grad_output_weights, d_model * vocab_size, scale);
        scale_gradients(d_grad_output_bias, vocab_size, scale);
        scale_gradients(d_grad_ln_final_gamma, d_model, scale);
        scale_gradients(d_grad_ln_final_beta, d_model, scale);

        for (int i = 0; i < num_layers; i++) {
            auto& grads = block_grads[i];

            scale_gradients(grads.d_grad_ln1_gamma, d_model, scale);
            scale_gradients(grads.d_grad_ln1_beta, d_model, scale);
            scale_gradients(grads.d_grad_ln2_gamma, d_model, scale);
            scale_gradients(grads.d_grad_ln2_beta, d_model, scale);

            scale_gradients(grads.d_grad_attn_W_Q, d_model * d_model, scale);
            scale_gradients(grads.d_grad_attn_b_Q, d_model, scale);
            scale_gradients(grads.d_grad_attn_W_K, d_model * d_model, scale);
            scale_gradients(grads.d_grad_attn_b_K, d_model, scale);
            scale_gradients(grads.d_grad_attn_W_V, d_model * d_model, scale);
            scale_gradients(grads.d_grad_attn_b_V, d_model, scale);
            scale_gradients(grads.d_grad_attn_W_O, d_model * d_model, scale);
            scale_gradients(grads.d_grad_attn_b_O, d_model, scale);

            scale_gradients(grads.d_grad_ffn_W1, d_ff * d_model, scale);
            scale_gradients(grads.d_grad_ffn_b1, d_ff, scale);
            scale_gradients(grads.d_grad_ffn_W2, d_model * d_ff, scale);
            scale_gradients(grads.d_grad_ffn_b2, d_model, scale);
        }
    }
}

std::vector<float> Transformer::compute_per_layer_gradient_norms() {
    std::vector<float> layer_norms;

    // 1. Embeddings norm (token + position + final LN)
    float emb_norm_sq = 0.0f;
    float norm;

    norm = gradient_norm(d_grad_token_embeddings, vocab_size * d_model);
    emb_norm_sq += norm * norm;

    norm = gradient_norm(d_grad_position_embeddings, max_seq_len * d_model);
    emb_norm_sq += norm * norm;

    norm = gradient_norm(d_grad_ln_final_gamma, d_model);
    emb_norm_sq += norm * norm;

    norm = gradient_norm(d_grad_ln_final_beta, d_model);
    emb_norm_sq += norm * norm;

    layer_norms.push_back(sqrtf(emb_norm_sq));

    // 2. Per-transformer-layer norms
    for (int i = 0; i < num_layers; i++) {
        auto& grads = block_grads[i];
        float layer_norm_sq = 0.0f;

        // Layer norm parameters
        norm = gradient_norm(grads.d_grad_ln1_gamma, d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_ln1_beta, d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_ln2_gamma, d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_ln2_beta, d_model);
        layer_norm_sq += norm * norm;

        // Attention parameters
        norm = gradient_norm(grads.d_grad_attn_W_Q, d_model * d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_attn_b_Q, d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_attn_W_K, d_model * d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_attn_b_K, d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_attn_W_V, d_model * d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_attn_b_V, d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_attn_W_O, d_model * d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_attn_b_O, d_model);
        layer_norm_sq += norm * norm;

        // FFN parameters
        norm = gradient_norm(grads.d_grad_ffn_W1, d_ff * d_model);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_ffn_b1, d_ff);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_ffn_W2, d_model * d_ff);
        layer_norm_sq += norm * norm;
        norm = gradient_norm(grads.d_grad_ffn_b2, d_model);
        layer_norm_sq += norm * norm;

        layer_norms.push_back(sqrtf(layer_norm_sq));
    }

    // 3. Output projection norm
    float out_norm_sq = 0.0f;
    norm = gradient_norm(d_grad_output_weights, d_model * vocab_size);
    out_norm_sq += norm * norm;
    norm = gradient_norm(d_grad_output_bias, vocab_size);
    out_norm_sq += norm * norm;

    layer_norms.push_back(sqrtf(out_norm_sq));

    return layer_norms;
}

std::vector<float> Transformer::compute_forward_activation_stats() {
    std::vector<float> max_vals;

    // Note: This function should be called after a forward pass
    // to get meaningful statistics on the activations

    // 1. Embeddings (after adding token + position embeddings)
    TensorStats emb_stats = compute_tensor_stats(d_embeddings, max_batch_size * max_seq_len * d_model);
    max_vals.push_back(fabsf(emb_stats.max) > fabsf(emb_stats.min) ? fabsf(emb_stats.max) : fabsf(emb_stats.min));

    // 2. Per-layer activations
    for (int i = 0; i < num_layers; i++) {
        auto& block = blocks[i];

        // Attention output (after residual)
        TensorStats attn_stats = compute_tensor_stats(block->d_attn_output, max_batch_size * max_seq_len * d_model);
        max_vals.push_back(fabsf(attn_stats.max) > fabsf(attn_stats.min) ? fabsf(attn_stats.max) : fabsf(attn_stats.min));

        // FFN output (block output, after second residual)
        TensorStats ffn_stats = compute_tensor_stats(block->d_ffn_output, max_batch_size * max_seq_len * d_model);
        max_vals.push_back(fabsf(ffn_stats.max) > fabsf(ffn_stats.min) ? fabsf(ffn_stats.max) : fabsf(ffn_stats.min));
    }

    // 3. Final layer norm output
    TensorStats ln_stats = compute_tensor_stats(d_normed, max_batch_size * max_seq_len * d_model);
    max_vals.push_back(fabsf(ln_stats.max) > fabsf(ln_stats.min) ? fabsf(ln_stats.max) : fabsf(ln_stats.min));

    return max_vals;
}
