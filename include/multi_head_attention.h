#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <cuda_runtime.h>

class MultiHeadAttention {
    friend class Transformer;  // Allow Transformer to access private members for training

public:
    // Constructor
    // d_model: Model dimension (e.g., 512)
    // num_heads: Number of attention heads (e.g., 8)
    // max_seq_len: Maximum sequence length
    // max_batch_size: Maximum batch size
    MultiHeadAttention(
        int d_model,
        int num_heads,
        int max_seq_len,
        int max_batch_size
    );

    // Destructor
    ~MultiHeadAttention();

    // Initialize parameters (Xavier initialization)
    void initialize_parameters();

    // Forward pass - Self-attention (Q = K = V = X)
    // h_X: host input [B, N, d_model]
    // h_output: host output [B, N, d_model]
    // batch_size: actual batch size
    // seq_len: actual sequence length
    // h_mask: optional attention mask [B, N, N] (1 = attend, 0 = ignore)
    void forward(
        const float* h_X,
        float* h_output,
        int batch_size,
        int seq_len,
        const float* h_mask = nullptr
    );

    // Forward pass - Cross-attention (separate Q and K/V)
    // h_Q: query input [B, N, d_model]
    // h_KV: key/value input [B, M, d_model]
    // h_output: output [B, N, d_model]
    void forward_cross(
        const float* h_Q,
        const float* h_KV,
        float* h_output,
        int batch_size,
        int seq_len_q,
        int seq_len_kv,
        const float* h_mask = nullptr
    );

    // Forward pass - Device to device (for use in larger models)
    // d_input: input [B, N, d_model] (already on device)
    // d_output: output [B, N, d_model] (already on device)
    // d_mask_input: optional mask [N, N] (already on device, can be nullptr)
    void forward_device_to_device(
        const float* d_input,
        float* d_output,
        int batch_size,
        int seq_len,
        const float* d_mask_input = nullptr
    );

    // Backward pass - Device to device (for training)
    // d_input: input from forward pass [B, N, d_model]
    // d_grad_output: gradient w.r.t. output [B, N, d_model]
    // d_grad_input: gradient w.r.t. input [B, N, d_model] (output)
    // d_grad_W_*,  d_grad_b_*: parameter gradients (output)
    void backward_device_to_device(
        const float* d_input,
        const float* d_grad_output,
        float* d_grad_input,
        float* d_grad_W_Q, float* d_grad_b_Q,
        float* d_grad_W_K, float* d_grad_b_K,
        float* d_grad_W_V, float* d_grad_b_V,
        float* d_grad_W_O, float* d_grad_b_O,
        int batch_size,
        int seq_len
    );

    // Save/load parameters
    void save_parameters(const char* filename);
    void load_parameters(const char* filename);

    // Getters
    int get_d_model() const { return d_model; }
    int get_num_heads() const { return num_heads; }
    int get_d_k() const { return d_k; }
    int get_max_seq_len() const { return max_seq_len; }
    int get_max_batch_size() const { return max_batch_size; }

private:
    // Architecture
    int d_model;
    int num_heads;
    int d_k, d_v;  // d_k = d_v = d_model / num_heads
    int max_seq_len;
    int max_batch_size;

    // Device pointers - Parameters
    float *d_W_Q, *d_W_K, *d_W_V, *d_W_O;  // Projection weights
    float *d_b_Q, *d_b_K, *d_b_V, *d_b_O;  // Projection biases

    // Device pointers - Intermediate buffers
    float *d_Q_proj, *d_K_proj, *d_V_proj;     // After projection [B, N, h*d_k]
    float *d_Q_heads, *d_K_heads, *d_V_heads;  // After reshape [B*h, N, d_k]
    float *d_scores;                            // Attention scores [B*h, N, M]
    float *d_attn_weights;                      // After softmax [B*h, N, M]
    float *d_context;                           // After attention [B*h, N, d_v]
    float *d_output_concat;                     // After concat [B, N, h*d_v]

    // Input/mask buffers
    float *d_X, *d_KV;
    float *d_mask;

    // Helper methods
    void allocate_memory();
    void free_memory();
    void forward_device(int batch_size, int seq_len_q, int seq_len_kv);
};

#endif // MULTI_HEAD_ATTENTION_H
