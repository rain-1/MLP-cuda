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

    // Backward pass - computes gradients
    void backward(
        const int* d_token_ids,    // Input tokens [batch_size, seq_len]
        const int* d_targets,      // Target tokens [batch_size, seq_len]
        int batch_size,
        int seq_len
    );

    // Training step - forward + backward + optimizer update
    float train_step(
        const int* h_token_ids,    // [batch_size, seq_len]
        const int* h_targets,      // [batch_size, seq_len]
        int batch_size,
        int seq_len,
        float learning_rate
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
    float* d_logits_buffer;         // [batch_size, max_seq_len, vocab_size]

    // Gradient buffers (allocated on demand for training)
    float* d_grad_token_embeddings;   // [vocab_size, d_model]
    float* d_grad_position_embeddings; // [max_seq_len, d_model]
    float* d_grad_output_weights;     // [d_model, vocab_size]
    float* d_grad_output_bias;        // [vocab_size]
    float* d_grad_ln_final_gamma;     // [d_model]
    float* d_grad_ln_final_beta;      // [d_model]

    // Gradient buffers for TransformerBlocks (per-layer)
    struct BlockGradients {
        float *d_grad_ln1_gamma, *d_grad_ln1_beta;
        float *d_grad_ln2_gamma, *d_grad_ln2_beta;
        float *d_grad_attn_W_Q, *d_grad_attn_b_Q;
        float *d_grad_attn_W_K, *d_grad_attn_b_K;
        float *d_grad_attn_W_V, *d_grad_attn_b_V;
        float *d_grad_attn_W_O, *d_grad_attn_b_O;
        float *d_grad_ffn_W1, *d_grad_ffn_b1;
        float *d_grad_ffn_W2, *d_grad_ffn_b2;
    };
    std::vector<BlockGradients> block_grads;

    // Adam optimizer state
    int training_step;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    // Adam momentum buffers for embeddings and output
    float *d_m_token_embeddings, *d_v_token_embeddings;
    float *d_m_position_embeddings, *d_v_position_embeddings;
    float *d_m_output_weights, *d_v_output_weights;
    float *d_m_output_bias, *d_v_output_bias;
    float *d_m_ln_final_gamma, *d_v_ln_final_gamma;
    float *d_m_ln_final_beta, *d_v_ln_final_beta;

    // Adam momentum buffers for blocks (per-layer)
    struct BlockOptimState {
        float *d_m_ln1_gamma, *d_v_ln1_gamma;
        float *d_m_ln1_beta, *d_v_ln1_beta;
        float *d_m_ln2_gamma, *d_v_ln2_gamma;
        float *d_m_ln2_beta, *d_v_ln2_beta;
        float *d_m_attn_W_Q, *d_v_attn_W_Q;
        float *d_m_attn_b_Q, *d_v_attn_b_Q;
        float *d_m_attn_W_K, *d_v_attn_W_K;
        float *d_m_attn_b_K, *d_v_attn_b_K;
        float *d_m_attn_W_V, *d_v_attn_W_V;
        float *d_m_attn_b_V, *d_v_attn_b_V;
        float *d_m_attn_W_O, *d_v_attn_W_O;
        float *d_m_attn_b_O, *d_v_attn_b_O;
        float *d_m_ffn_W1, *d_v_ffn_W1;
        float *d_m_ffn_b1, *d_v_ffn_b1;
        float *d_m_ffn_W2, *d_v_ffn_W2;
        float *d_m_ffn_b2, *d_v_ffn_b2;
    };
    std::vector<BlockOptimState> block_optim_state;

    void allocate_memory();
    void free_memory();
    void allocate_gradient_buffers();
    void free_gradient_buffers();
    void allocate_optimizer_state();
    void free_optimizer_state();

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
