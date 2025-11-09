#include "diagnostics.h"
#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

// ============================================================================
// CUDA Kernels
// ============================================================================

// Compute sum of squares kernel
__global__ void sum_of_squares_kernel(
    const float* data,
    float* partial_sums,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Compute local sum of squares
    float val = 0.0f;
    if (idx < size) {
        val = data[idx] * data[idx];
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

// Compute max absolute value kernel
__global__ void max_abs_kernel(
    const float* data,
    float* partial_max,
    int size
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Compute local max abs
    float val = 0.0f;
    if (idx < size) {
        val = fabsf(data[idx]);
    }
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

// Compute statistics kernel (sum, min, max)
__global__ void stats_kernel(
    const float* data,
    float* partial_sum,
    float* partial_min,
    float* partial_max,
    int size
) {
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_min = sdata + blockDim.x;
    float* s_max = sdata + 2 * blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize local values
    float val = 0.0f;
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    if (idx < size) {
        val = data[idx];
        min_val = val;
        max_val = val;
    }

    s_sum[tid] = val;
    s_min[tid] = min_val;
    s_max[tid] = max_val;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_sum[blockIdx.x] = s_sum[0];
        partial_min[blockIdx.x] = s_min[0];
        partial_max[blockIdx.x] = s_max[0];
    }
}

// Check for NaN or Inf kernel
__global__ void check_nan_inf_kernel(
    const float* data,
    int* has_invalid,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float val = data[idx];
        if (isnan(val) || isinf(val)) {
            atomicAdd(has_invalid, 1);
        }
    }
}

// ============================================================================
// Host Functions
// ============================================================================

float compute_l2_norm(const float* d_tensor, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // Allocate device memory for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // Compute partial sums of squares
    sum_of_squares_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_tensor, d_partial_sums, size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums to host and reduce
    float* h_partial_sums = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float sum_of_squares = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        sum_of_squares += h_partial_sums[i];
    }

    // Cleanup
    delete[] h_partial_sums;
    CUDA_CHECK(cudaFree(d_partial_sums));

    return sqrtf(sum_of_squares);
}

float compute_max_abs(const float* d_tensor, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // Allocate device memory for partial max
    float* d_partial_max;
    CUDA_CHECK(cudaMalloc(&d_partial_max, gridSize * sizeof(float)));

    // Compute partial max abs
    max_abs_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_tensor, d_partial_max, size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy partial max to host and reduce
    float* h_partial_max = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_max, d_partial_max, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        max_abs = fmaxf(max_abs, h_partial_max[i]);
    }

    // Cleanup
    delete[] h_partial_max;
    CUDA_CHECK(cudaFree(d_partial_max));

    return max_abs;
}

TensorStats compute_tensor_stats(const float* d_tensor, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = 3 * blockSize * sizeof(float);

    // Allocate device memory for partial results
    float* d_partial_sum;
    float* d_partial_min;
    float* d_partial_max;
    CUDA_CHECK(cudaMalloc(&d_partial_sum, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_min, gridSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_max, gridSize * sizeof(float)));

    // Compute partial stats
    stats_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_tensor, d_partial_sum, d_partial_min, d_partial_max, size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy partial results to host and reduce
    float* h_partial_sum = new float[gridSize];
    float* h_partial_min = new float[gridSize];
    float* h_partial_max = new float[gridSize];
    CUDA_CHECK(cudaMemcpy(h_partial_sum, d_partial_sum, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partial_min, d_partial_min, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partial_max, d_partial_max, gridSize * sizeof(float),
                         cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    for (int i = 0; i < gridSize; i++) {
        sum += h_partial_sum[i];
        min_val = fminf(min_val, h_partial_min[i]);
        max_val = fmaxf(max_val, h_partial_max[i]);
    }

    // Cleanup
    delete[] h_partial_sum;
    delete[] h_partial_min;
    delete[] h_partial_max;
    CUDA_CHECK(cudaFree(d_partial_sum));
    CUDA_CHECK(cudaFree(d_partial_min));
    CUDA_CHECK(cudaFree(d_partial_max));

    TensorStats stats;
    stats.mean = sum / size;
    stats.min_val = min_val;
    stats.max_val = max_val;
    stats.max_abs = fmaxf(fabsf(min_val), fabsf(max_val));
    stats.l2_norm = compute_l2_norm(d_tensor, size);

    return stats;
}

bool has_nan_or_inf(const float* d_tensor, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Allocate device memory for flag
    int* d_has_invalid;
    CUDA_CHECK(cudaMalloc(&d_has_invalid, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_has_invalid, 0, sizeof(int)));

    // Check for NaN or Inf
    check_nan_inf_kernel<<<gridSize, blockSize>>>(
        d_tensor, d_has_invalid, size
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy result to host
    int h_has_invalid = 0;
    CUDA_CHECK(cudaMemcpy(&h_has_invalid, d_has_invalid, sizeof(int),
                         cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_has_invalid));

    return h_has_invalid > 0;
}

// ============================================================================
// DiagnosticLogger Implementation
// ============================================================================

DiagnosticLogger::DiagnosticLogger()
    : enabled_(true),
      grad_norm_threshold_(100.0f),
      param_norm_threshold_(1000.0f),
      loss_increase_threshold_(1.5f),
      history_idx_(0),
      min_loss_(FLT_MAX),
      steps_since_improvement_(0)
{
    for (int i = 0; i < HISTORY_SIZE; i++) {
        loss_history_[i] = 0.0f;
    }
}

DiagnosticLogger::~DiagnosticLogger() {
}

void DiagnosticLogger::log_step(
    int step,
    float loss,
    float learning_rate,
    float grad_norm,
    float param_norm
) {
    if (!enabled_) return;

    printf("  [DIAG] Step %d | Loss: %.6f | LR: %.6f | GradNorm: %.6f | ParamNorm: %.6f\n",
           step, loss, learning_rate, grad_norm, param_norm);

    // Check for warnings
    if (isnan(loss) || isinf(loss)) {
        printf("  [WARN] *** LOSS IS NaN or Inf! Training has diverged! ***\n");
    }

    if (grad_norm > grad_norm_threshold_) {
        printf("  [WARN] Gradient norm %.6f exceeds threshold %.6f\n",
               grad_norm, grad_norm_threshold_);
    }

    if (param_norm > param_norm_threshold_) {
        printf("  [WARN] Parameter norm %.6f exceeds threshold %.6f\n",
               param_norm, param_norm_threshold_);
    }

    // Update loss history
    loss_history_[history_idx_] = loss;
    history_idx_ = (history_idx_ + 1) % HISTORY_SIZE;

    // Track minimum loss
    if (loss < min_loss_) {
        min_loss_ = loss;
        steps_since_improvement_ = 0;
    } else {
        steps_since_improvement_++;
    }
}

void DiagnosticLogger::log_gradient_stats(
    int step,
    const char* name,
    const float* d_grad,
    int size
) {
    if (!enabled_) return;

    TensorStats stats = compute_tensor_stats(d_grad, size);

    printf("  [GRAD] %s | L2: %.6e | Max: %.6e | Min: %.6e | Mean: %.6e\n",
           name, stats.l2_norm, stats.max_val, stats.min_val, stats.mean);

    // Check for NaN or Inf
    if (has_nan_or_inf(d_grad, size)) {
        printf("  [WARN] *** %s contains NaN or Inf! ***\n", name);
    }
}

void DiagnosticLogger::log_parameter_stats(
    int step,
    const char* name,
    const float* d_param,
    int size
) {
    if (!enabled_) return;

    TensorStats stats = compute_tensor_stats(d_param, size);

    printf("  [PARAM] %s | L2: %.6e | Max: %.6e | Min: %.6e | Mean: %.6e\n",
           name, stats.l2_norm, stats.max_val, stats.min_val, stats.mean);

    // Check for NaN or Inf
    if (has_nan_or_inf(d_param, size)) {
        printf("  [WARN] *** %s contains NaN or Inf! ***\n", name);
    }
}

bool DiagnosticLogger::check_divergence(int step, float loss) {
    if (!enabled_) return false;

    // Check for NaN or Inf
    if (isnan(loss) || isinf(loss)) {
        printf("\n");
        printf("  ========================================\n");
        printf("  DIVERGENCE DETECTED at step %d\n", step);
        printf("  Loss is NaN or Inf!\n");
        printf("  ========================================\n");
        printf("\n");
        return true;
    }

    // Check for rapid loss increase
    if (history_idx_ >= 5 && loss > loss_increase_threshold_ * min_loss_) {
        printf("\n");
        printf("  ========================================\n");
        printf("  POTENTIAL DIVERGENCE at step %d\n", step);
        printf("  Loss %.6f is %.2fx the minimum loss %.6f\n",
               loss, loss / min_loss_, min_loss_);
        printf("  This suggests training instability\n");
        printf("  ========================================\n");
        printf("\n");
        return true;
    }

    // Check for prolonged lack of improvement
    if (steps_since_improvement_ > 100) {
        printf("\n");
        printf("  [INFO] No improvement for %d steps (min loss: %.6f)\n",
               steps_since_improvement_, min_loss_);
        printf("\n");
    }

    return false;
}

void DiagnosticLogger::print_summary() {
    if (!enabled_) return;

    printf("\n");
    printf("  ========================================\n");
    printf("  DIAGNOSTIC SUMMARY\n");
    printf("  ========================================\n");
    printf("  Minimum loss achieved: %.6f\n", min_loss_);
    printf("  Steps since improvement: %d\n", steps_since_improvement_);
    printf("  ========================================\n");
    printf("\n");
}
