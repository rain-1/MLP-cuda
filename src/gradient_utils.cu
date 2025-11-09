#include "gradient_utils.h"
#include "matrix_ops.h"  // For CUDA_CHECK macro
#include <cmath>
#include <float.h>

// Kernel to compute squared L2 norm
__global__ void squared_norm_kernel(
    const float* data,
    float* partial_sums,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute partial sum for this thread
    float sum = 0.0f;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float val = data[i];
        sum += val * val;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction
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

// Kernel to scale gradients
__global__ void scale_kernel(float* data, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
    }
}

// Kernel to check for NaN/Inf
__global__ void check_nan_inf_kernel(
    const float* data,
    int* has_issue,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        if (isnan(val) || isinf(val)) {
            atomicAdd(has_issue, 1);
        }
    }
}

// Kernel to compute min/max/sum for statistics
__global__ void stats_kernel(
    const float* data,
    float* partial_min,
    float* partial_max,
    float* partial_sum,
    float* partial_sum_sq,
    int* partial_nan_count,
    int* partial_inf_count,
    int size
) {
    extern __shared__ float sdata[];
    float* s_min = sdata;
    float* s_max = sdata + blockDim.x;
    float* s_sum = sdata + 2 * blockDim.x;
    float* s_sum_sq = sdata + 3 * blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int local_nan = 0;
    int local_inf = 0;

    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float val = data[i];
        if (isnan(val)) {
            local_nan++;
        } else if (isinf(val)) {
            local_inf++;
        } else {
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    s_min[tid] = local_min;
    s_max[tid] = local_max;
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_min[blockIdx.x] = s_min[0];
        partial_max[blockIdx.x] = s_max[0];
        partial_sum[blockIdx.x] = s_sum[0];
        partial_sum_sq[blockIdx.x] = s_sum_sq[0];
        atomicAdd(partial_nan_count, local_nan);
        atomicAdd(partial_inf_count, local_inf);
    }
}

float gradient_norm(const float* d_grad, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    if (gridSize > 1024) gridSize = 1024;

    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    squared_norm_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_grad, d_partial_sums, size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy and reduce on host
    float* h_partial_sums = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums,
                         gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    float total = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total += h_partial_sums[i];
    }

    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    return sqrtf(total);
}

void clip_gradients(float* d_grad, int size, float max_norm) {
    float norm = gradient_norm(d_grad, size);

    if (norm > max_norm) {
        float scale = max_norm / norm;

        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;

        scale_kernel<<<gridSize, blockSize>>>(d_grad, size, scale);
        CUDA_CHECK(cudaGetLastError());
    }
}

bool has_nan_or_inf(const float* d_tensor, int size) {
    int* d_has_issue;
    CUDA_CHECK(cudaMalloc(&d_has_issue, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_has_issue, 0, sizeof(int)));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    check_nan_inf_kernel<<<gridSize, blockSize>>>(d_tensor, d_has_issue, size);
    CUDA_CHECK(cudaGetLastError());

    int h_has_issue;
    CUDA_CHECK(cudaMemcpy(&h_has_issue, d_has_issue, sizeof(int),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_has_issue));

    return h_has_issue > 0;
}

TensorStats compute_tensor_stats(const float* d_tensor, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    if (gridSize > 1024) gridSize = 1024;

    float* d_partial_min;
    float* d_partial_max;
    float* d_partial_sum;
    float* d_partial_sum_sq;
    int* d_nan_count;
    int* d_inf_count;

    CUDA_CHECK(cudaMalloc(&d_partial_min, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_max, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_sum_sq, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nan_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_inf_count, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_nan_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_inf_count, 0, sizeof(int)));

    stats_kernel<<<gridSize, blockSize, 4 * blockSize * sizeof(float)>>>(
        d_tensor, d_partial_min, d_partial_max, d_partial_sum, d_partial_sum_sq,
        d_nan_count, d_inf_count, size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy to host and reduce
    float* h_min = new float[gridSize];
    float* h_max = new float[gridSize];
    float* h_sum = new float[gridSize];
    float* h_sum_sq = new float[gridSize];
    int h_nan_count, h_inf_count;

    CUDA_CHECK(cudaMemcpy(h_min, d_partial_min, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max, d_partial_max, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum, d_partial_sum, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_sq, d_partial_sum_sq, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_nan_count, d_nan_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_inf_count, d_inf_count, sizeof(int), cudaMemcpyDeviceToHost));

    TensorStats stats;
    stats.min = h_min[0];
    stats.max = h_max[0];
    stats.has_nan = (h_nan_count > 0);
    stats.has_inf = (h_inf_count > 0);

    float total_sum = 0.0f;
    float total_sum_sq = 0.0f;

    for (int i = 0; i < gridSize; i++) {
        stats.min = fminf(stats.min, h_min[i]);
        stats.max = fmaxf(stats.max, h_max[i]);
        total_sum += h_sum[i];
        total_sum_sq += h_sum_sq[i];
    }

    int valid_count = size - h_nan_count - h_inf_count;
    if (valid_count > 0) {
        stats.mean = total_sum / valid_count;
        float variance = (total_sum_sq / valid_count) - (stats.mean * stats.mean);
        stats.std = sqrtf(fmaxf(0.0f, variance));
    } else {
        stats.mean = 0.0f;
        stats.std = 0.0f;
    }

    delete[] h_min;
    delete[] h_max;
    delete[] h_sum;
    delete[] h_sum_sq;

    CUDA_CHECK(cudaFree(d_partial_min));
    CUDA_CHECK(cudaFree(d_partial_max));
    CUDA_CHECK(cudaFree(d_partial_sum));
    CUDA_CHECK(cudaFree(d_partial_sum_sq));
    CUDA_CHECK(cudaFree(d_nan_count));
    CUDA_CHECK(cudaFree(d_inf_count));

    return stats;
}
