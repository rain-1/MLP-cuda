#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "transformer_block.h"
#include "tokenizer.h"
#include <cuda_runtime.h>
#include <vector>

// GPT-style decoder-only transformer for language modeling
class Transformer {
public:
    Transformer(
        int vocab_size,
        int d_model,
        int num_layers,
        int num_heads,
        int d_ff,
        int max_seq_len,
        int max_batch_size = 32
    );

    ~Transformer();

    void initialize_parameters();

    // Forward pass (training mode - returns logits for all positions)
    // Input: [batch_size, seq_len] token IDs
    // Output: [batch_size, seq_len, vocab_size] logits
    void forward_device(
        const int* d_token_ids,
        float* d_logits,
        int batch_size,
        int seq_len
    );

    // Forward pass with host memory
    void forward(
        const int* h_token_ids,
        float* h_logits,
        int batch_size,
        int seq_len
    );

    // Text generation - single sequence
    std::vector<int> generate(
        const std::vector<int>& prompt,
        int max_new_tokens,
        float temperature = 1.0f,
        int top_k = 0,          // 0 means no top-k filtering
        float top_p = 1.0f,     // 1.0 means no nucleus sampling
        int seed = 0
    );

    // Batch text generation
    void generate_batch(
        const std::vector<std::vector<int>>& prompts,
        std::vector<std::vector<int>>& outputs,
        int max_new_tokens,
        float temperature = 1.0f,
        int top_k = 0,
        float top_p = 1.0f,
        int seed = 0
    );

    // Compute loss (cross-entropy)
    float compute_loss(
        const int* h_token_ids,    // [batch_size, seq_len]
        const int* h_targets,      // [batch_size, seq_len]
        int batch_size,
        int seq_len
    );

    void save_parameters(const char* filename);
    void load_parameters(const char* filename);

    // Getters
    int get_vocab_size() const { return vocab_size; }
    int get_d_model() const { return d_model; }
    int get_num_layers() const { return num_layers; }
    int get_max_seq_len() const { return max_seq_len; }

private:
    // Model dimensions
    int vocab_size;
    int d_model;
    int num_layers;
    int num_heads;
    int d_ff;
    int max_seq_len;
    int max_batch_size;

    // Embedding parameters
    float* d_token_embeddings;      // [vocab_size, d_model]
    float* d_position_embeddings;   // [max_seq_len, d_model]

    // Transformer blocks
    std::vector<TransformerBlock*> blocks;

    // Final layer norm
    float* d_ln_final_gamma;
    float* d_ln_final_beta;

    // Output projection (tied with token embeddings by default)
    float* d_output_weights;        // [d_model, vocab_size]
    float* d_output_bias;           // [vocab_size]

    // Buffers
    float* d_embeddings;            // [batch_size, seq_len, d_model]
    float* d_block_output;          // [batch_size, seq_len, d_model]
    float* d_normed;                // [batch_size, seq_len, d_model]
    float* d_causal_mask;           // [max_seq_len, max_seq_len]

    // Generation buffers
    int* d_token_ids;               // [batch_size, max_seq_len]
    float* d_logits_buffer;         // [batch_size, vocab_size]

    void allocate_memory();
    void free_memory();

    // Helper for generating next token
    int sample_token(
        const float* logits,
        int vocab_size,
        float temperature,
        int top_k,
        float top_p,
        unsigned int& rng_state
    );
};

#endif // TRANSFORMER_H
