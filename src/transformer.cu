#include "transformer.h"
#include "transformer_block.h"
#include "transformer_layers.h"
#include "matrix_ops.h"
#include "loss.h"
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
    int max_batch_size
) : vocab_size(vocab_size), d_model(d_model), num_layers(num_layers),
    num_heads(num_heads), d_ff(d_ff), max_seq_len(max_seq_len),
    max_batch_size(max_batch_size)
{
    allocate_memory();

    // Create transformer blocks
    for (int i = 0; i < num_layers; i++) {
        blocks.push_back(new TransformerBlock(
            d_model, num_heads, d_ff, max_batch_size, max_seq_len
        ));
    }

    initialize_parameters();
}

Transformer::~Transformer() {
    for (auto block : blocks) {
        delete block;
    }
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
    float* h_pos_emb = new float[max_seq_len * d_model];
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < d_model / 2; i++) {
            float angle = (float)pos / powf(10000.0f, 2.0f * i / d_model);
            h_pos_emb[pos * d_model + 2 * i] = sinf(angle);
            h_pos_emb[pos * d_model + 2 * i + 1] = cosf(angle);
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
    // Forward pass
    forward(h_token_ids, nullptr, batch_size, seq_len);

    // Compute cross-entropy loss
    // TODO: Implement proper loss computation
    fprintf(stderr, "Loss computation not yet implemented\n");
    return 0.0f;
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
