#include "transformer_layers.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

// Layer Normalization kernel
__global__ void layer_norm_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int B, int N, int d_model,
    float epsilon
) {
    int batch_seq = blockIdx.x;  // 0 to B*N-1
    if (batch_seq >= B * N) return;

    extern __shared__ float sdata[];
    float* s_mean = sdata;
    float* s_var = sdata + blockDim.x;

    int tid = threadIdx.x;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        sum += input[batch_seq * d_model + i];
    }
    s_mean[tid] = sum;
    __syncthreads();

    // Reduce mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mean[tid] += s_mean[tid + s];
        }
        __syncthreads();
    }
    float mean = s_mean[0] / d_model;
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float diff = input[batch_seq * d_model + i] - mean;
        var_sum += diff * diff;
    }
    s_var[tid] = var_sum;
    __syncthreads();

    // Reduce variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_var[tid] += s_var[tid + s];
        }
        __syncthreads();
    }
    float variance = s_var[0] / d_model;
    float std_dev = sqrtf(variance + epsilon);
    __syncthreads();

    // Normalize and scale
    for (int i = tid; i < d_model; i += blockDim.x) {
        float normalized = (input[batch_seq * d_model + i] - mean) / std_dev;
        output[batch_seq * d_model + i] = gamma[i] * normalized + beta[i];
    }
}

// GELU activation kernel
__global__ void gelu_forward_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// Embedding lookup kernel
__global__ void embedding_forward_kernel(
    const int* token_ids,
    const float* embeddings,
    float* output,
    int B, int N,
    int vocab_size, int d_model
) {
    int batch = blockIdx.y;
    int seq = blockIdx.x;
    int dim = threadIdx.x;

    if (batch < B && seq < N && dim < d_model) {
        int token_id = token_ids[batch * N + seq];
        if (token_id >= 0 && token_id < vocab_size) {
            output[batch * N * d_model + seq * d_model + dim] =
                embeddings[token_id * d_model + dim];
        } else {
            output[batch * N * d_model + seq * d_model + dim] = 0.0f;
        }
    }
}

// Causal mask creation kernel
__global__ void create_causal_mask_kernel(
    float* mask,
    int seq_len
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && j < seq_len) {
        // Lower triangular: 1 if i >= j, else 0
        mask[i * seq_len + j] = (i >= j) ? 1.0f : 0.0f;
    }
}

// Host functions

void layer_norm(
    const float* d_input,
    float* d_output,
    const float* d_gamma,
    const float* d_beta,
    int B, int N, int d_model,
    float epsilon
) {
    int block_size = 256;
    int grid_size = B * N;
    size_t shared_mem_size = 2 * block_size * sizeof(float);

    layer_norm_kernel<<<grid_size, block_size, shared_mem_size>>>(
        d_input, d_output, d_gamma, d_beta, B, N, d_model, epsilon
    );
    CUDA_CHECK(cudaGetLastError());
}

void gelu_forward(
    const float* d_input,
    float* d_output,
    int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    gelu_forward_kernel<<<grid_size, block_size>>>(
        d_input, d_output, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void embedding_forward(
    const int* d_token_ids,
    const float* d_embeddings,
    float* d_output,
    int B, int N,
    int vocab_size, int d_model
) {
    dim3 block_size(d_model);
    dim3 grid_size(N, B);

    embedding_forward_kernel<<<grid_size, block_size>>>(
        d_token_ids, d_embeddings, d_output, B, N, vocab_size, d_model
    );
    CUDA_CHECK(cudaGetLastError());
}

void create_causal_mask(
    float* d_mask,
    int seq_len
) {
    dim3 block_size(16, 16);
    dim3 grid_size((seq_len + 15) / 16, (seq_len + 15) / 16);

    create_causal_mask_kernel<<<grid_size, block_size>>>(
        d_mask, seq_len
    );
    CUDA_CHECK(cudaGetLastError());
}

// Placeholder implementations for backward passes (to be implemented for training)
void layer_norm_backward(
    const float* d_grad_output,
    const float* d_input,
    const float* d_gamma,
    float* d_grad_input,
    float* d_grad_gamma,
    float* d_grad_beta,
    int B, int N, int d_model,
    float epsilon
) {
    // TODO: Implement for training
    fprintf(stderr, "layer_norm_backward not yet implemented\n");
}

void gelu_backward(
    const float* d_grad_output,
    const float* d_input,
    float* d_grad_input,
    int size
) {
    // TODO: Implement for training
    fprintf(stderr, "gelu_backward not yet implemented\n");
}

void dropout_forward(
    const float* d_input,
    float* d_output,
    float* d_mask,
    int size,
    float drop_prob,
    unsigned long long seed
) {
    // TODO: Implement
    fprintf(stderr, "dropout_forward not yet implemented\n");
}

void dropout_backward(
    const float* d_grad_output,
    const float* d_mask,
    float* d_grad_input,
    int size,
    float drop_prob
) {
    // TODO: Implement
    fprintf(stderr, "dropout_backward not yet implemented\n");
}

void embedding_backward(
    const float* d_grad_output,
    const int* d_token_ids,
    float* d_grad_embeddings,
    int B, int N,
    int vocab_size, int d_model
) {
    // TODO: Implement for training
    fprintf(stderr, "embedding_backward not yet implemented\n");
}
