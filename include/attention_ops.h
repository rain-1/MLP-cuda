#ifndef ATTENTION_OPS_H
#define ATTENTION_OPS_H

#include <cuda_runtime.h>

// Reshape and transpose: [B, N, h*d] -> [B*h, N, d]
// Used to separate heads after QKV projection
void reshape_BNhd_to_BhNd(
    const float* d_input,   // [B, N, h*d]
    float* d_output,        // [B*h, N, d]
    int B, int N, int h, int d
);

// Reshape and transpose: [B*h, N, d] -> [B, N, h*d]
// Used to concatenate heads back together
void reshape_BhNd_to_BNhd(
    const float* d_input,   // [B*h, N, d]
    float* d_output,        // [B, N, h*d]
    int B, int N, int h, int d
);

// Batched softmax along last dimension
// Input/Output: [B, N, M]
void batched_softmax(
    const float* d_input,
    float* d_output,
    int B, int N, int M
);

// Scaled dot-product attention
// Q: [Bh, N, d_k], K: [Bh, M, d_k], V: [Bh, M, d_v]
// Output: [Bh, N, d_v]
// Computes: softmax(Q·K^T / √d_k) · V
void scaled_dot_product_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_output,
    float* d_scores_buffer,      // Temporary [Bh, N, M]
    float* d_attn_weights_buffer,// Temporary [Bh, N, M]
    int Bh, int N, int M, int d_k, int d_v,
    const float* d_mask = nullptr // Optional mask [Bh, N, M]
);

// Apply attention mask (add large negative value where mask == 0)
// scores: [B, N, M], mask: [B, N, M] (1 = attend, 0 = ignore)
void apply_attention_mask(
    float* d_scores,
    const float* d_mask,
    int B, int N, int M
);

// Backward passes

// Batched softmax backward
// grad_output, softmax_output: [B, N, M]
// Computes: grad_input[i] = softmax[i] * (grad_output[i] - Σ_j(grad_output[j] * softmax[j]))
void batched_softmax_backward(
    const float* d_grad_output,
    const float* d_softmax_output,
    float* d_grad_input,
    int B, int N, int M
);

// Scaled dot-product attention backward
void scaled_dot_product_attention_backward(
    const float* d_Q,                    // Forward input [Bh, N, d_k]
    const float* d_K,                    // Forward input [Bh, M, d_k]
    const float* d_V,                    // Forward input [Bh, M, d_v]
    const float* d_attn_weights,         // Saved from forward [Bh, N, M]
    const float* d_grad_output,          // Gradient w.r.t. output [Bh, N, d_v]
    float* d_grad_Q,                     // Gradient w.r.t. Q [Bh, N, d_k]
    float* d_grad_K,                     // Gradient w.r.t. K [Bh, M, d_k]
    float* d_grad_V,                     // Gradient w.r.t. V [Bh, M, d_v]
    float* d_grad_scores_buffer,         // Temporary [Bh, N, M]
    float* d_grad_attn_buffer,           // Temporary [Bh, N, M]
    int Bh, int N, int M, int d_k, int d_v,
    const float* d_mask = nullptr
);

#endif // ATTENTION_OPS_H
