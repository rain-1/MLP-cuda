#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "multi_head_attention.h"
#include <cuda_runtime.h>

// Helper function for residual connections with scaling
void add_residual(
    const float* d_input,
    const float* d_residual,
    float* d_output,
    int size,
    float scale = 1.0f  // Default to 1.0 (no scaling) for backward compatibility
);

// Feed-Forward Network
// Two-layer MLP with GELU activation
// FFN(x) = GELU(x·W1 + b1)·W2 + b2
class FeedForwardNetwork {
    friend class Transformer;  // Allow Transformer to access private members for training

public:
    FeedForwardNetwork(
        int d_model,
        int d_ff,
        int max_batch_size,
        int max_seq_len,
        float init_scale = 1.0f  // Scale for W2 initialization (for residual depth scaling)
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

    // Backward pass
    void backward_device(
        const float* d_input,        // Input from forward pass [B, N, d_model]
        const float* d_grad_output,  // Gradient w.r.t. output [B, N, d_model]
        float* d_grad_input,         // Gradient w.r.t. input [B, N, d_model]
        float* d_grad_W1,            // Gradient w.r.t. W1 [d_ff, d_model]
        float* d_grad_b1,            // Gradient w.r.t. b1 [d_ff]
        float* d_grad_W2,            // Gradient w.r.t. W2 [d_model, d_ff]
        float* d_grad_b2,            // Gradient w.r.t. b2 [d_model]
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
    float init_scale;  // Scale factor for W2 initialization

    // Parameters
    float *d_W1, *d_b1;  // First layer: [d_ff, d_model]
    float *d_W2, *d_b2;  // Second layer: [d_model, d_ff]

    // Intermediate buffers
    float *d_z1;         // Pre-GELU activations: [B, N, d_ff]
    float *d_hidden;     // Post-GELU activations: [B, N, d_ff]

    void allocate_memory();
    void free_memory();
};

// Transformer Decoder Block
// Combines multi-head self-attention and feed-forward network
// with residual connections and layer normalization
class TransformerBlock {
    friend class Transformer;  // Allow Transformer to access private members for training

public:
    TransformerBlock(
        int d_model,
        int num_heads,
        int d_ff,
        int max_batch_size,
        int max_seq_len,
        float residual_scale = 1.0f  // Scale factor for residual connections
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

    // Backward pass
    void backward_device(
        const float* d_input,        // Input from forward pass [B, N, d_model]
        const float* d_grad_output,  // Gradient w.r.t. output [B, N, d_model]
        float* d_grad_input,         // Gradient w.r.t. input [B, N, d_model]
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
    );

    void save_parameters(const char* filename);
    void load_parameters(const char* filename);

private:
    int d_model;
    int num_heads;
    int d_ff;
    int max_batch_size;
    int max_seq_len;
    float residual_scale;  // Scale factor for residual connections

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
