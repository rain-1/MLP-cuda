#include "loss.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// MSE loss kernel - computes partial sums
__global__ void mse_loss_kernel(const float* pred, const float* target,
                                 float* partial_sums, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Compute local sum
    float diff = 0.0f;
    if (idx < size) {
        diff = pred[idx] - target[idx];
        diff = diff * diff;
    }
    sdata[tid] = diff;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// MSE gradient kernel
__global__ void mse_gradient_kernel(const float* pred, const float* target,
                                     float* grad, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = scale * (pred[idx] - target[idx]);
    }
}

// Cross-entropy loss kernel - computes partial sums
__global__ void cross_entropy_loss_kernel(const float* pred, const float* target,
                                           float* partial_sums, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Compute local sum
    float val = 0.0f;
    if (idx < size) {
        // Avoid log(0) by clamping pred to small positive value
        float p = fmaxf(pred[idx], 1e-7f);
        val = -target[idx] * logf(p);
    }
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Cross-entropy gradient kernel (for softmax + cross-entropy)
__global__ void cross_entropy_gradient_kernel(const float* pred, const float* target,
                                               float* grad, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = scale * (pred[idx] - target[idx]);
    }
}

// Host functions

float mse_loss(const float* d_pred, const float* d_target, int size, int batch_size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // Allocate device memory for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // Compute partial sums
    mse_loss_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_pred, d_target,
                                                             d_partial_sums, size);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums to host and reduce
    float* h_partial_sums = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total_loss += h_partial_sums[i];
    }

    // Cleanup
    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    return total_loss / (2.0f * batch_size);
}

void mse_gradient(const float* d_pred, const float* d_target, float* d_grad,
                  int size, int batch_size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    float scale = 1.0f / batch_size;
    mse_gradient_kernel<<<gridSize, blockSize>>>(d_pred, d_target, d_grad, size, scale);
    CUDA_CHECK(cudaGetLastError());
}

float cross_entropy_loss(const float* d_pred, const float* d_target,
                          int size, int batch_size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // Allocate device memory for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // Compute partial sums
    cross_entropy_loss_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_pred, d_target,
                                                                       d_partial_sums, size);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums to host and reduce
    float* h_partial_sums = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total_loss += h_partial_sums[i];
    }

    // Cleanup
    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    return total_loss / batch_size;
}

void cross_entropy_gradient(const float* d_pred, const float* d_target, float* d_grad,
                             int size, int batch_size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    float scale = 1.0f / batch_size;
    cross_entropy_gradient_kernel<<<gridSize, blockSize>>>(d_pred, d_target,
                                                           d_grad, size, scale);
    CUDA_CHECK(cudaGetLastError());
}

// Language modeling cross-entropy loss kernel
// Computes loss for a single position: -log(softmax(logits)[target])
// Note: This kernel processes one position per thread (no vocab_size limitation)
__global__ void lm_cross_entropy_loss_kernel(
    const float* logits,
    const int* targets,
    const float* mask,
    float* partial_sums,
    int batch_size,
    int seq_len,
    int vocab_size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_loss = 0.0f;

    if (idx < batch_size * seq_len) {
        // Check if this position is masked
        float m = (mask != nullptr) ? mask[idx] : 1.0f;

        if (m > 0.0f) {
            int target = targets[idx];

            if (target >= 0 && target < vocab_size) {
                // Get logits for this position
                const float* logits_ptr = logits + idx * vocab_size;

                // Compute max for numerical stability
                float max_logit = -INFINITY;
                for (int i = 0; i < vocab_size; i++) {
                    max_logit = fmaxf(max_logit, logits_ptr[i]);
                }

                // Compute log-sum-exp with Kahan summation
                float sum_exp = 0.0f;
                float c = 0.0f;  // Compensation term
                for (int i = 0; i < vocab_size; i++) {
                    float exp_val = expf(logits_ptr[i] - max_logit);
                    float y = exp_val - c;
                    float t = sum_exp + y;
                    c = (t - sum_exp) - y;
                    sum_exp = t;
                }
                float log_sum_exp = logf(sum_exp) + max_logit;

                // Cross-entropy: -log(softmax(logits)[target])
                //                = -(logits[target] - log_sum_exp)
                local_loss = -(logits_ptr[target] - log_sum_exp) * m;
            }
        }
    }

    sdata[tid] = local_loss;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Language modeling cross-entropy gradient kernel
// Computes gradient: softmax(logits) - one_hot(target), scaled by mask
// Uses Kahan summation for improved numerical precision
// Supports vocab_size > 1024 using grid-stride loop
__global__ void lm_cross_entropy_gradient_kernel(
    const float* logits,
    const int* targets,
    const float* mask,
    float* grad,
    int batch_size,
    int seq_len,
    int vocab_size,
    float scale
) {
    int idx = blockIdx.x;  // Position index (batch * seq_len)

    if (idx < batch_size * seq_len) {
        // Check if this position is masked
        float m = (mask != nullptr) ? mask[idx] : 1.0f;

        if (m > 0.0f) {
            int target = targets[idx];
            const float* logits_ptr = logits + idx * vocab_size;
            float* grad_ptr = grad + idx * vocab_size;

            // Compute max (shared across all threads in this block)
            extern __shared__ float sdata[];
            float thread_max = -INFINITY;
            for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
                thread_max = fmaxf(thread_max, logits_ptr[v]);
            }
            sdata[threadIdx.x] = thread_max;
            __syncthreads();

            // Reduce to find global max
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
                }
                __syncthreads();
            }
            float max_logit = sdata[0];
            __syncthreads();

            // Kahan summation for exp sum (each thread accumulates subset)
            float thread_sum = 0.0f;
            float c = 0.0f;  // Compensation term
            for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
                float exp_val = expf(logits_ptr[v] - max_logit);
                float y = exp_val - c;
                float t = thread_sum + y;
                c = (t - thread_sum) - y;
                thread_sum = t;
            }
            sdata[threadIdx.x] = thread_sum;
            __syncthreads();

            // Reduce to get total sum
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    sdata[threadIdx.x] += sdata[threadIdx.x + s];
                }
                __syncthreads();
            }
            float sum_exp = sdata[0];
            __syncthreads();

            // Each thread computes gradients for its subset of vocabulary
            for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
                float softmax_v = expf(logits_ptr[v] - max_logit) / sum_exp;
                float target_indicator = (v == target) ? 1.0f : 0.0f;
                grad_ptr[v] = scale * m * (softmax_v - target_indicator);
            }
        } else {
            // Zero out gradients for masked positions
            for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
                grad[idx * vocab_size + v] = 0.0f;
            }
        }
    }
}

float lm_cross_entropy_loss(
    const float* d_logits,
    const int* d_targets,
    int batch_size,
    int seq_len,
    int vocab_size,
    const float* d_mask
) {
    int total_positions = batch_size * seq_len;
    int blockSize = 256;
    int gridSize = (total_positions + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // Allocate device memory for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // Compute partial sums
    lm_cross_entropy_loss_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_logits, d_targets, d_mask, d_partial_sums,
        batch_size, seq_len, vocab_size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums to host and reduce
    float* h_partial_sums = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total_loss += h_partial_sums[i];
    }

    // Cleanup
    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    // Compute number of non-masked positions
    int num_valid = total_positions;
    if (d_mask != nullptr) {
        // Count non-masked positions
        float* h_mask = new float[total_positions];
        CUDA_CHECK(cudaMemcpy(h_mask, d_mask, total_positions * sizeof(float),
                             cudaMemcpyDeviceToHost));
        num_valid = 0;
        for (int i = 0; i < total_positions; i++) {
            if (h_mask[i] > 0.0f) num_valid++;
        }
        delete[] h_mask;
    }

    return (num_valid > 0) ? (total_loss / num_valid) : 0.0f;
}

void lm_cross_entropy_gradient(
    const float* d_logits,
    const int* d_targets,
    float* d_grad,
    int batch_size,
    int seq_len,
    int vocab_size,
    const float* d_mask
) {
    int total_positions = batch_size * seq_len;

    // Compute number of non-masked positions for scaling
    int num_valid = total_positions;
    if (d_mask != nullptr) {
        float* h_mask = new float[total_positions];
        CUDA_CHECK(cudaMemcpy(h_mask, d_mask, total_positions * sizeof(float),
                             cudaMemcpyDeviceToHost));
        num_valid = 0;
        for (int i = 0; i < total_positions; i++) {
            if (h_mask[i] > 0.0f) num_valid++;
        }
        delete[] h_mask;
    }

    float scale = (num_valid > 0) ? (1.0f / num_valid) : 0.0f;

    // Launch kernel: one block per position, with threads cooperating for large vocabularies
    // Supports vocab_size > 1024 using grid-stride loop
    dim3 gridSize(total_positions);
    int blockSize = (vocab_size <= 1024) ? vocab_size : 256;  // Use 256 threads for large vocabs
    size_t sharedMemSize = blockSize * sizeof(float);

    // DEBUG: Print launch config
    static bool printed = false;
    if (!printed) {
        printf("[DEBUG loss.cu] Using Kahan-enhanced gradient kernel: grid=%d, block=%d, shared_mem=%zu (vocab_size=%d)\n",
               total_positions, blockSize, sharedMemSize, vocab_size);
        printed = true;
    }

    lm_cross_entropy_gradient_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_logits, d_targets, d_mask, d_grad,
        batch_size, seq_len, vocab_size, scale
    );
    CUDA_CHECK(cudaGetLastError());
}
