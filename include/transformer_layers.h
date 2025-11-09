#ifndef TRANSFORMER_LAYERS_H
#define TRANSFORMER_LAYERS_H

#include <cuda_runtime.h>

// Layer Normalization
// Normalizes across the feature dimension for each sample in the batch
// Input/Output: [B, N, d_model]
void layer_norm(
    const float* d_input,
    float* d_output,
    const float* d_gamma,  // Scale parameter [d_model]
    const float* d_beta,   // Shift parameter [d_model]
    int B, int N, int d_model,
    float epsilon = 1e-5f
);

// Layer Norm backward pass (for training)
void layer_norm_backward(
    const float* d_grad_output,
    const float* d_input,
    const float* d_gamma,
    float* d_grad_input,
    float* d_grad_gamma,
    float* d_grad_beta,
    int B, int N, int d_model,
    float epsilon = 1e-5f
);

// GELU activation (Gaussian Error Linear Unit)
// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
void gelu_forward(
    const float* d_input,
    float* d_output,
    int size
);

void gelu_backward(
    const float* d_grad_output,
    const float* d_input,
    float* d_grad_input,
    int size
);

// Dropout (for training)
void dropout_forward(
    const float* d_input,
    float* d_output,
    float* d_mask,  // Binary mask for backward pass
    int size,
    float drop_prob,
    unsigned long long seed
);

void dropout_backward(
    const float* d_grad_output,
    const float* d_mask,
    float* d_grad_input,
    int size,
    float drop_prob
);

// Embedding lookup
// token_ids: [B, N] (integer token IDs)
// embeddings: [vocab_size, d_model]
// output: [B, N, d_model]
void embedding_forward(
    const int* d_token_ids,
    const float* d_embeddings,
    float* d_output,
    int B, int N,
    int vocab_size, int d_model
);

// Embedding backward (accumulate gradients)
void embedding_backward(
    const float* d_grad_output,  // [B, N, d_model]
    const int* d_token_ids,      // [B, N]
    float* d_grad_embeddings,    // [vocab_size, d_model]
    int B, int N,
    int vocab_size, int d_model
);

// Create causal attention mask (lower triangular)
// mask[i,j] = 1 if i >= j, else 0
// Output: [seq_len, seq_len]
void create_causal_mask(
    float* d_mask,
    int seq_len
);

#endif // TRANSFORMER_LAYERS_H
