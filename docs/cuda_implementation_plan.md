# CUDA Implementation Plan

## Overview

This document outlines the implementation strategy for a batched MLP with CUDA kernels. We'll build modular, testable components with efficient GPU kernels.

## Project Structure

```
MLP-cuda/
├── include/
│   ├── mlp.h              # Main MLP class interface
│   ├── matrix_ops.h       # Matrix operation kernels
│   ├── activations.h      # Activation function kernels
│   ├── loss.h             # Loss function kernels
│   └── adam.h             # Adam optimizer kernels
├── src/
│   ├── mlp.cu             # Main MLP implementation
│   ├── matrix_ops.cu      # Matrix operations (matmul, transpose)
│   ├── activations.cu     # ReLU, softmax, derivatives
│   ├── loss.cu            # MSE, cross-entropy
│   └── adam.cu            # Adam optimizer update
├── tests/
│   ├── test_matrix_ops.cu
│   ├── test_activations.cu
│   ├── test_forward.cu
│   ├── test_backward.cu
│   └── test_adam.cu
├── examples/
│   ├── train_regression.cu
│   └── train_classification.cu
├── docs/
│   ├── mathematical_derivation.md
│   └── cuda_implementation_plan.md
├── CMakeLists.txt
└── README.md
```

## Core CUDA Kernels

### 1. Matrix Multiplication (Tiled)

**Kernel**: `matmul_tiled(A, B, C, M, K, N)`

**Strategy**: Use shared memory tiling for efficiency

```cuda
// C = A * B
// A: [M x K], B: [K x N], C: [M x N]

#define TILE_SIZE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int K, int N)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

**Performance Characteristics**:
- Reduces global memory access by factor of TILE_SIZE
- Coalesced memory access patterns
- O(M*N*K/TILE_SIZE) memory transactions instead of O(M*N*K)

### 2. Activation Functions

**ReLU Forward**:
```cuda
__global__ void relu_forward(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}
```

**ReLU Backward** (gradient):
```cuda
__global__ void relu_backward(float* grad_output, float* input,
                               float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}
```

### 3. Bias Addition

```cuda
// Add bias to each row of a matrix
// input: [B x N], bias: [N], output: [B x N]
__global__ void add_bias(float* input, float* bias, float* output,
                         int B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < B && col < N) {
        output[row * N + col] = input[row * N + col] + bias[col];
    }
}
```

### 4. Gradient Sum (Bias Gradient)

```cuda
// Sum gradients across batch dimension
// grad_output: [B x N], grad_bias: [N]
__global__ void sum_rows(float* grad_output, float* grad_bias, int B, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        float sum = 0.0f;
        for (int row = 0; row < B; row++) {
            sum += grad_output[row * N + col];
        }
        grad_bias[col] = sum;
    }
}
```

**Optimized version with reduction**:
```cuda
__global__ void sum_rows_optimized(float* grad_output, float* grad_bias,
                                    int B, int N) {
    extern __shared__ float sdata[];

    int col = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Each thread sums a subset of rows
    float sum = 0.0f;
    for (int row = tid; row < B; row += stride) {
        sum += grad_output[row * N + col];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        grad_bias[col] = sdata[0];
    }
}
```

### 5. Loss Functions

**MSE Loss**:
```cuda
__global__ void mse_loss(float* pred, float* target, float* loss, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float diff = 0.0f;
    if (idx < size) {
        diff = pred[idx] - target[idx];
        diff = diff * diff;
    }
    sdata[tid] = diff;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss, sdata[0]);
    }
}
```

**MSE Gradient**:
```cuda
__global__ void mse_gradient(float* pred, float* target, float* grad,
                             int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = (pred[idx] - target[idx]) / batch_size;
    }
}
```

### 6. Adam Optimizer

```cuda
__global__ void adam_update(float* param, float* grad, float* m, float* v,
                            float lr, float beta1, float beta2, float epsilon,
                            float beta1_corr, float beta2_corr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Update first moment
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];

        // Update second moment
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];

        // Bias-corrected moments
        float m_hat = m[idx] / beta1_corr;
        float v_hat = v[idx] / beta2_corr;

        // Update parameter
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}
```

## Memory Management Strategy

### Device Memory Layout

For each layer, we need to store:
- **Parameters**: W, b
- **Activations**: Z (pre-activation), A (post-activation)
- **Gradients**: dW, db, dZ, dA
- **Adam moments**: m_W, v_W, m_b, v_b

### Memory Allocation

```cpp
class MLPLayer {
    // Parameters
    float *d_W, *d_b;

    // Activations (allocated per forward pass)
    float *d_Z, *d_A;

    // Gradients (allocated per backward pass)
    float *d_dW, *d_db, *d_dZ, *d_dA;

    // Adam state
    float *d_m_W, *d_v_W, *d_m_b, *d_v_b;

    // Dimensions
    int input_size, output_size;
};
```

### Memory Optimization

1. **Reuse activation buffers**: During training, we need to store activations for backprop, but during inference we can reuse buffers
2. **Stream processing**: For very large batches, process in smaller chunks using CUDA streams
3. **Pinned memory**: Use pinned host memory for faster CPU-GPU transfers

## Kernel Launch Configurations

### Thread Block Sizes

| Kernel Type | Block Dim | Grid Dim | Notes |
|-------------|-----------|----------|-------|
| MatMul | (16, 16) | ((N+15)/16, (M+15)/16) | Tile-based |
| Element-wise | 256 | (size+255)/256 | 1D operations |
| Reduction | 256 | num_elements | With shared memory |
| Bias ops | (16, 16) | ((N+15)/16, (B+15)/16) | 2D operations |

### Optimization Tips

1. **Occupancy**: Aim for high occupancy (>50%) but not at expense of shared memory
2. **Memory coalescing**: Access memory in contiguous patterns
3. **Bank conflicts**: Avoid shared memory bank conflicts in reductions
4. **Register usage**: Monitor register spilling with `--ptxas-options=-v`

## Implementation Phases

### Phase 1: Core Kernels (Testable Units)
- [x] Matrix multiplication (with tests comparing to CPU)
- [x] Activation functions (ReLU forward/backward)
- [x] Bias operations (add, gradient sum)
- [x] Loss functions (MSE forward/gradient)

### Phase 2: Layer Operations
- [x] Forward pass for single layer
- [x] Backward pass for single layer
- [x] Test each layer independently

### Phase 3: Adam Optimizer
- [x] Adam update kernel
- [x] Test parameter updates with known gradients

### Phase 4: Full MLP
- [x] Combine layers into MLP class
- [x] End-to-end forward pass
- [x] End-to-end backward pass
- [x] Training loop

### Phase 5: Testing & Validation
- [x] Unit tests for all kernels
- [x] Integration tests for full training
- [x] Gradient checking (numerical vs analytical)
- [x] Performance benchmarks

## Testing Strategy

### Unit Tests

Each kernel should have tests that:
1. **Correctness**: Compare GPU output to CPU reference implementation
2. **Edge cases**: Test with small sizes, large sizes, non-power-of-2 sizes
3. **Numerical stability**: Test with extreme values (very small, very large)

### Integration Tests

1. **Gradient checking**: Numerical gradient vs analytical gradient
   - For each parameter, compute: `(L(θ+ε) - L(θ-ε)) / (2ε)`
   - Compare to backprop gradient
   - Should match to ~1e-5 relative error

2. **Overfitting test**: Train on tiny dataset (10 samples)
   - Should achieve near-zero loss
   - Validates both forward and backward pass

3. **Known problem test**: Train on XOR or simple regression
   - Compare final weights/loss to expected values

### Performance Tests

1. **Throughput**: Measure samples/second for various batch sizes
2. **Memory bandwidth**: Compare to theoretical peak
3. **Kernel timing**: Profile individual kernels with `nvprof` or Nsight

## Build System

### CMakeLists.txt Structure

```cmake
cmake_minimum_required(VERSION 3.18)
project(MLP_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library
add_library(mlp_cuda STATIC
    src/matrix_ops.cu
    src/activations.cu
    src/loss.cu
    src/adam.cu
    src/mlp.cu
)

# Tests
enable_testing()
add_executable(test_matrix_ops tests/test_matrix_ops.cu)
target_link_libraries(test_matrix_ops mlp_cuda)
add_test(NAME MatrixOps COMMAND test_matrix_ops)

# Examples
add_executable(train_regression examples/train_regression.cu)
target_link_libraries(train_regression mlp_cuda)
```

## Next Steps

1. Implement core kernels with tests
2. Build layer-by-layer, testing each component
3. Integrate into full MLP
4. Optimize performance
5. Add examples and documentation

## Performance Goals

For a typical configuration (B=128, h1=784, h2=256, h3=128, h4=10):
- **Forward pass**: < 1ms
- **Backward pass**: < 2ms
- **Adam update**: < 0.5ms
- **Total iteration**: < 4ms
- **Throughput**: > 30k samples/second

These are achievable with well-optimized CUDA kernels on modern GPUs (e.g., RTX 3090, A100).
