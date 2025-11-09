#include "attention_ops.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Kernel to reshape [B, N, h*d] -> [B*h, N, d]
__global__ void reshape_BNhd_to_BhNd_kernel(
    const float* input,   // [B, N, h*d]
    float* output,        // [B*h, N, d]
    int B, int N, int h, int d
) {
    int batch_head = blockIdx.z;  // 0 to B*h-1
    int b = batch_head / h;
    int head = batch_head % h;

    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int dim = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N && dim < d) {
        // Input: [b][n][head * d + dim]
        int in_idx = b * (N * h * d) + n * (h * d) + head * d + dim;

        // Output: [batch_head][n][dim]
        int out_idx = batch_head * (N * d) + n * d + dim;

        output[out_idx] = input[in_idx];
    }
}

// Kernel to reshape [B*h, N, d] -> [B, N, h*d]
__global__ void reshape_BhNd_to_BNhd_kernel(
    const float* input,   // [B*h, N, d]
    float* output,        // [B, N, h*d]
    int B, int N, int h, int d
) {
    int batch_head = blockIdx.z;  // 0 to B*h-1
    int b = batch_head / h;
    int head = batch_head % h;

    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int dim = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N && dim < d) {
        // Input: [batch_head][n][dim]
        int in_idx = batch_head * (N * d) + n * d + dim;

        // Output: [b][n][head * d + dim]
        int out_idx = b * (N * h * d) + n * (h * d) + head * d + dim;

        output[out_idx] = input[in_idx];
    }
}

// Batched softmax kernel
__global__ void batched_softmax_kernel(
    const float* input,   // [B, N, M]
    float* output,        // [B, N, M]
    int B, int N, int M
) {
    extern __shared__ float sdata[];

    int batch_seq = blockIdx.x;  // 0 to B*N-1
    int tid = threadIdx.x;

    if (batch_seq >= B * N) return;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int m = tid; m < M; m += blockDim.x) {
        float val = input[batch_seq * M + m];
        max_val = fmaxf(max_val, val);
    }

    sdata[tid] = max_val;
    __syncthreads();

    // Reduction to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (int m = tid; m < M; m += blockDim.x) {
        float val = expf(input[batch_seq * M + m] - max_val);
        output[batch_seq * M + m] = val;
        sum += val;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction to find global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    // Normalize
    for (int m = tid; m < M; m += blockDim.x) {
        output[batch_seq * M + m] /= sum;
    }
}

// Kernel to apply attention mask
__global__ void apply_attention_mask_kernel(
    float* scores,        // [B, N, M]
    const float* mask,    // [B, N, M] (1 = attend, 0 = ignore)
    int B, int N, int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N * M;

    if (idx < total) {
        if (mask[idx] == 0.0f) {
            scores[idx] = -1e9f;  // Large negative value
        }
    }
}

// Host functions

void reshape_BNhd_to_BhNd(
    const float* d_input,
    float* d_output,
    int B, int N, int h, int d
) {
    dim3 blockDim(16, 16);
    dim3 gridDim((d + 15) / 16, (N + 15) / 16, B * h);

    reshape_BNhd_to_BhNd_kernel<<<gridDim, blockDim>>>(
        d_input, d_output, B, N, h, d
    );
    CUDA_CHECK(cudaGetLastError());
}

void reshape_BhNd_to_BNhd(
    const float* d_input,
    float* d_output,
    int B, int N, int h, int d
) {
    dim3 blockDim(16, 16);
    dim3 gridDim((d + 15) / 16, (N + 15) / 16, B * h);

    reshape_BhNd_to_BNhd_kernel<<<gridDim, blockDim>>>(
        d_input, d_output, B, N, h, d
    );
    CUDA_CHECK(cudaGetLastError());
}

void batched_softmax(
    const float* d_input,
    float* d_output,
    int B, int N, int M
) {
    int blockSize = 256;
    int gridSize = B * N;
    size_t sharedMemSize = blockSize * sizeof(float);

    batched_softmax_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_input, d_output, B, N, M
    );
    CUDA_CHECK(cudaGetLastError());
}

void apply_attention_mask(
    float* d_scores,
    const float* d_mask,
    int B, int N, int M
) {
    int total = B * N * M;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    apply_attention_mask_kernel<<<gridSize, blockSize>>>(
        d_scores, d_mask, B, N, M
    );
    CUDA_CHECK(cudaGetLastError());
}

void scaled_dot_product_attention(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_output,
    float* d_scores_buffer,
    float* d_attn_weights_buffer,
    int Bh, int N, int M, int d_k, int d_v,
    const float* d_mask
) {
    float scale = 1.0f / sqrtf((float)d_k);

    // 1. Compute attention scores: Scores = Q · K^T
    //    Q: [Bh, N, d_k], K: [Bh, M, d_k]
    //    Scores: [Bh, N, M]
    matmul_transB(d_Q, d_K, d_scores_buffer, Bh * N, d_k, M);

    // 2. Scale scores
    scale_matrix(d_scores_buffer, scale, Bh * N * M);

    // 3. Apply mask if provided
    if (d_mask != nullptr) {
        apply_attention_mask(d_scores_buffer, d_mask, Bh, N, M);
    }

    // 4. Apply softmax: Attn = softmax(Scores)
    batched_softmax(d_scores_buffer, d_attn_weights_buffer, Bh, N, M);

    // 5. Compute context: Output = Attn · V
    //    Attn: [Bh, N, M], V: [Bh, M, d_v]
    //    Output: [Bh, N, d_v]
    matmul(d_attn_weights_buffer, d_V, d_output, Bh * N, M, d_v);
}

// Backward passes

// Batched softmax backward kernel
__global__ void batched_softmax_backward_kernel(
    const float* grad_output,      // [B, N, M]
    const float* softmax_output,   // [B, N, M]
    float* grad_input,             // [B, N, M]
    int B, int N, int M
) {
    extern __shared__ float sdata[];

    int batch_seq = blockIdx.x;  // 0 to B*N-1
    int tid = threadIdx.x;

    if (batch_seq >= B * N) return;

    const float* y = softmax_output + batch_seq * M;
    const float* dy = grad_output + batch_seq * M;
    float* dx = grad_input + batch_seq * M;

    // Compute sum = Σ_j (dy[j] * y[j])
    float sum = 0.0f;
    for (int m = tid; m < M; m += blockDim.x) {
        sum += dy[m] * y[m];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    // Compute gradient: dx[i] = y[i] * (dy[i] - sum)
    for (int m = tid; m < M; m += blockDim.x) {
        dx[m] = y[m] * (dy[m] - sum);
    }
}

void batched_softmax_backward(
    const float* d_grad_output,
    const float* d_softmax_output,
    float* d_grad_input,
    int B, int N, int M
) {
    int blockSize = 256;
    int gridSize = B * N;
    size_t sharedMemSize = blockSize * sizeof(float);

    batched_softmax_backward_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_grad_output, d_softmax_output, d_grad_input, B, N, M
    );
    CUDA_CHECK(cudaGetLastError());
}

void scaled_dot_product_attention_backward(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    const float* d_attn_weights,
    const float* d_grad_output,
    float* d_grad_Q,
    float* d_grad_K,
    float* d_grad_V,
    float* d_grad_scores_buffer,
    float* d_grad_attn_buffer,
    int Bh, int N, int M, int d_k, int d_v,
    const float* d_mask
) {
    float scale = 1.0f / sqrtf((float)d_k);

    // Backward through: Output = Attn · V
    // grad_Attn = grad_Output · V^T
    // grad_V = Attn^T · grad_Output
    matmul_backward(
        d_grad_output,       // grad_C [Bh*N x d_v]
        d_attn_weights,      // A [Bh*N x M]
        d_V,                 // B [M x d_v] -- wait, V is [Bh*M x d_v]
        d_grad_attn_buffer,  // grad_A [Bh*N x M]
        d_grad_V,            // grad_B [M x d_v]
        Bh * N,              // M_mat
        M,                   // K_mat
        d_v                  // N_mat
    );

    // Backward through softmax
    batched_softmax_backward(
        d_grad_attn_buffer,    // grad_output [Bh, N, M]
        d_attn_weights,        // softmax_output [Bh, N, M]
        d_grad_scores_buffer,  // grad_input [Bh, N, M]
        Bh, N, M
    );

    // Backward through scaling
    scale_matrix(d_grad_scores_buffer, scale, Bh * N * M);

    // Note: Mask backward is not needed (mask is constant)

    // Backward through: Scores = Q · K^T
    // grad_Q = grad_Scores · K
    // grad_K = grad_Scores^T · Q
    matmul_transB_backward(
        d_grad_scores_buffer,  // grad_C [Bh*N x M]
        d_Q,                   // A [Bh*N x d_k]
        d_K,                   // B [M x d_k] -- wait, K is [Bh*M x d_k]
        d_grad_Q,              // grad_A [Bh*N x d_k]
        d_grad_K,              // grad_B [M x d_k]
        Bh * N,                // M_mat
        d_k,                   // K_mat
        M                      // N_mat
    );
}
