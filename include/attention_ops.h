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

#endif // ATTENTION_OPS_H
