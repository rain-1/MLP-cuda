#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "multi_head_attention.h"
#include <cuda_runtime.h>

// Helper function for residual connections
void add_residual(
    const float* d_input,
    const float* d_residual,
    float* d_output,
    int size
);

// Feed-Forward Network
// Two-layer MLP with GELU activation
// FFN(x) = GELU(x·W1 + b1)·W2 + b2
class FeedForwardNetwork {
public:
    FeedForwardNetwork(
        int d_model,
        int d_ff,
        int max_batch_size,
        int max_seq_len
    );

    ~FeedForwardNetwork();

    void initialize_parameters();

    // Forward pass
    void forward_device(
        const float* d_input,   // [B, N, d_model]
        float* d_output,        // [B, N, d_model]
        int batch_size,
        int seq_len
    );

    void save_parameters(const char* filename);
    void load_parameters(const char* filename);

private:
    int d_model;
    int d_ff;
    int max_batch_size;
    int max_seq_len;

    // Parameters
    float *d_W1, *d_b1;  // First layer: [d_model, d_ff]
    float *d_W2, *d_b2;  // Second layer: [d_ff, d_model]

    // Intermediate buffers
    float *d_hidden;     // After first layer: [B, N, d_ff]

    void allocate_memory();
    void free_memory();
};

// Transformer Decoder Block
// Combines multi-head self-attention and feed-forward network
// with residual connections and layer normalization
class TransformerBlock {
public:
    TransformerBlock(
        int d_model,
        int num_heads,
        int d_ff,
        int max_batch_size,
        int max_seq_len
    );

    ~TransformerBlock();

    void initialize_parameters();

    // Forward pass
    void forward_device(
        const float* d_input,   // [B, N, d_model]
        float* d_output,        // [B, N, d_model]
        const float* d_mask,    // Causal mask [N, N] or [B*h, N, N]
        int batch_size,
        int seq_len
    );

    void save_parameters(const char* filename);
    void load_parameters(const char* filename);

private:
    int d_model;
    int num_heads;
    int d_ff;
    int max_batch_size;
    int max_seq_len;

    // Components
    MultiHeadAttention* attention;
    FeedForwardNetwork* ffn;

    // Layer norm parameters
    float *d_ln1_gamma, *d_ln1_beta;  // After attention
    float *d_ln2_gamma, *d_ln2_beta;  // After FFN

    // Intermediate buffers
    float *d_attn_output;     // Attention output
    float *d_attn_normed;     // After first layer norm + residual
    float *d_ffn_output;      // FFN output
    float *d_ffn_normed;      // After second layer norm + residual

    void allocate_memory();
    void free_memory();
};

#endif // TRANSFORMER_BLOCK_H
