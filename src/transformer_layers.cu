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

// Layer norm backward kernel
__global__ void layer_norm_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* gamma,
    float* grad_input,
    float* grad_gamma_partial,
    float* grad_beta_partial,
    int B, int N, int d_model,
    float epsilon
) {
    int batch_seq = blockIdx.x;
    if (batch_seq >= B * N) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    const float* x = input + batch_seq * d_model;
    const float* dy = grad_output + batch_seq * d_model;
    float* dx = grad_input + batch_seq * d_model;

    // Recompute mean and variance
    float sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        sum += x[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / d_model;
    __syncthreads();

    float var_sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    sdata[tid] = var_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float variance = sdata[0] / d_model;
    float std_dev = sqrtf(variance + epsilon);
    float inv_std = 1.0f / std_dev;
    __syncthreads();

    // Compute sums needed for gradient
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float xhat = (x[i] - mean) * inv_std;
        sum_dy += dy[i] * gamma[i];
        sum_dy_xhat += dy[i] * gamma[i] * xhat;
    }
    sdata[tid] = sum_dy;
    sdata[tid + blockDim.x] = sum_dy_xhat;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
        }
        __syncthreads();
    }
    sum_dy = sdata[0];
    sum_dy_xhat = sdata[blockDim.x];
    __syncthreads();

    // Compute gradient w.r.t. input
    for (int i = tid; i < d_model; i += blockDim.x) {
        float xhat = (x[i] - mean) * inv_std;
        float dx_val = gamma[i] * dy[i];
        dx_val -= sum_dy / d_model;
        dx_val -= xhat * sum_dy_xhat / d_model;
        dx_val *= inv_std;
        dx[i] = dx_val;

        // Accumulate gradients for gamma and beta
        atomicAdd(&grad_gamma_partial[i], dy[i] * xhat);
        atomicAdd(&grad_beta_partial[i], dy[i]);
    }
}

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
    int block_size = 256;
    int grid_size = B * N;
    size_t shared_mem = 2 * block_size * sizeof(float);

    CUDA_CHECK(cudaMemset(d_grad_gamma, 0, d_model * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_beta, 0, d_model * sizeof(float)));

    layer_norm_backward_kernel<<<grid_size, block_size, shared_mem>>>(
        d_grad_output, d_input, d_gamma, d_grad_input,
        d_grad_gamma, d_grad_beta, B, N, d_model, epsilon
    );
    CUDA_CHECK(cudaGetLastError());
}

// GELU backward kernel
__global__ void gelu_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        const float sqrt_2_over_pi = 0.7978845608f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
        float tanh_inner = tanhf(inner);

        // Derivative: sech²(inner) = 1 - tanh²(inner)
        float sech_sq = 1.0f - tanh_inner * tanh_inner;
        float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x);
        float d_gelu = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_sq * d_inner;

        grad_input[idx] = grad_output[idx] * d_gelu;
    }
}

void gelu_backward(
    const float* d_grad_output,
    const float* d_input,
    float* d_grad_input,
    int size
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    gelu_backward_kernel<<<grid_size, block_size>>>(d_grad_output, d_input, d_grad_input, size);
    CUDA_CHECK(cudaGetLastError());
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

// Embedding backward kernel
__global__ void embedding_backward_kernel(
    const float* grad_output,
    const int* token_ids,
    float* grad_embeddings,
    int B, int N, int vocab_size, int d_model
) {
    int batch = blockIdx.y;
    int seq = blockIdx.x;
    int dim = threadIdx.x;

    if (batch < B && seq < N && dim < d_model) {
        int token_id = token_ids[batch * N + seq];
        if (token_id >= 0 && token_id < vocab_size) {
            float grad = grad_output[batch * N * d_model + seq * d_model + dim];
            atomicAdd(&grad_embeddings[token_id * d_model + dim], grad);
        }
    }
}

void embedding_backward(
    const float* d_grad_output,
    const int* d_token_ids,
    float* d_grad_embeddings,
    int B, int N,
    int vocab_size, int d_model
) {
    dim3 block_size(d_model);
    dim3 grid_size(N, B);

    embedding_backward_kernel<<<grid_size, block_size>>>(
        d_grad_output, d_token_ids, d_grad_embeddings,
        B, N, vocab_size, d_model
    );
    CUDA_CHECK(cudaGetLastError());
}
