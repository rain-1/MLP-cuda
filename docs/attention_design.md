# Multi-Head Attention Design

## Overview

This document describes the mathematical formulation and CUDA implementation strategy for Multi-Head Attention, a key component of Transformer architectures.

## 1. Mathematical Formulation

### 1.1 Scaled Dot-Product Attention

Given:
- **Q** (Query): `[B × N × d_k]` - Batch size B, sequence length N, dimension d_k
- **K** (Key): `[B × M × d_k]` - Key sequence length M
- **V** (Value): `[B × M × d_v]` - Value dimension d_v

The scaled dot-product attention is computed as:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Step by step:

1. **Compute attention scores**:
   ```
   S = Q · K^T                    # [B × N × M]
   ```

2. **Scale by √d_k** (for numerical stability):
   ```
   S_scaled = S / √d_k            # [B × N × M]
   ```

3. **Apply softmax** (row-wise):
   ```
   A = softmax(S_scaled)          # [B × N × M]
   ```
   Where `softmax(S[i,j,:]) = exp(S[i,j,:]) / sum(exp(S[i,j,:]))`

4. **Compute weighted sum of values**:
   ```
   Output = A · V                 # [B × N × d_v]
   ```

### 1.2 Multi-Head Attention

Instead of performing a single attention function, multi-head attention linearly projects Q, K, V into h different subspaces and performs attention in parallel.

**Parameters**:
- `d_model`: Model dimension (e.g., 512)
- `h`: Number of attention heads (e.g., 8)
- `d_k = d_v = d_model / h`: Dimension per head (e.g., 64)

**Projection Matrices**:
- **W_Q^i**: `[d_model × d_k]` for i-th head
- **W_K^i**: `[d_model × d_k]` for i-th head
- **W_V^i**: `[d_model × d_v]` for i-th head
- **W_O**: `[h·d_v × d_model]` output projection

**Algorithm**:

1. **Project inputs to Q, K, V for each head**:
   ```
   Q_i = X · W_Q^i                # [B × N × d_k]
   K_i = X · W_K^i                # [B × M × d_k]
   V_i = X · W_V^i                # [B × M × d_v]
   ```

2. **Compute attention for each head**:
   ```
   head_i = Attention(Q_i, K_i, V_i)  # [B × N × d_v]
   ```

3. **Concatenate heads**:
   ```
   MultiHead = Concat(head_1, ..., head_h)  # [B × N × h·d_v]
   ```

4. **Apply output projection**:
   ```
   Output = MultiHead · W_O       # [B × N × d_model]
   ```

### 1.3 Efficient Implementation

Rather than computing h separate projections, we can combine them:

**Combined Projections**:
- **W_Q**: `[d_model × (h·d_k)]` - all query projections stacked
- **W_K**: `[d_model × (h·d_k)]` - all key projections stacked
- **W_V**: `[d_model × (h·d_v)]` - all value projections stacked

**Efficient Algorithm**:

1. **Single matrix multiply for all heads**:
   ```
   Q_all = X · W_Q                # [B × N × (h·d_k)]
   K_all = X · W_K                # [B × M × (h·d_k)]
   V_all = X · W_V                # [B × M × (h·d_v)]
   ```

2. **Reshape to separate heads**:
   ```
   Q_heads = reshape(Q_all, [B, N, h, d_k])  # [B × N × h × d_k]
   K_heads = reshape(K_all, [B, M, h, d_k])  # [B × M × h × d_k]
   V_heads = reshape(V_all, [B, M, h, d_v])  # [B × M × h × d_v]
   ```

3. **Transpose to process heads in parallel**:
   ```
   Q_heads = transpose(Q_heads, [0, 2, 1, 3])  # [B × h × N × d_k]
   K_heads = transpose(K_heads, [0, 2, 1, 3])  # [B × h × M × d_k]
   V_heads = transpose(V_heads, [0, 2, 1, 3])  # [B × h × M × d_v]
   ```

4. **Batched attention** (treat B×h as batch dimension):
   ```
   # Reshape to [B·h × N × d_k], [B·h × M × d_k], [B·h × M × d_v]
   Scores = Q_heads · K_heads^T / √d_k    # [B·h × N × M]
   Attn = softmax(Scores)                 # [B·h × N × M]
   Output_heads = Attn · V_heads          # [B·h × N × d_v]
   ```

5. **Reshape and concatenate**:
   ```
   Output_heads = reshape(Output_heads, [B, h, N, d_v])
   Output_heads = transpose(Output_heads, [0, 2, 1, 3])  # [B × N × h × d_v]
   Output_concat = reshape(Output_heads, [B, N, h·d_v])
   ```

6. **Output projection**:
   ```
   Output = Output_concat · W_O           # [B × N × d_model]
   ```

## 2. CUDA Implementation Strategy

### 2.1 Required Kernels

1. **Batched Matrix Multiplication**
   - Already implemented in matrix_ops.cu
   - Use for Q·W_Q, K·W_K, V·W_V, and final projection

2. **Reshape and Transpose**
   ```cuda
   // Reshape [B, N, h*d] -> [B, N, h, d]
   // Then transpose [B, N, h, d] -> [B, h, N, d]
   __global__ void reshape_transpose_4d(
       const float* input,  // [B, N, h*d]
       float* output,       // [B, h, N, d]
       int B, int N, int h, int d
   )
   ```

3. **Scaled Dot-Product Attention**
   ```cuda
   // Q: [B*h, N, d_k], K: [B*h, M, d_k], V: [B*h, M, d_v]
   // Output: [B*h, N, d_v]
   void scaled_dot_product_attention(
       const float* Q, const float* K, const float* V,
       float* output,
       int B_h, int N, int M, int d_k, int d_v,
       float* scores_buffer  // [B*h, N, M] temporary storage
   )
   ```

4. **Batched Softmax**
   ```cuda
   // Apply softmax along last dimension
   // Input/Output: [B, N, M]
   __global__ void batched_softmax(
       const float* input,
       float* output,
       int B, int N, int M
   )
   ```

### 2.2 Memory Layout

For efficient GPU computation:

**Parameters** (stored in row-major):
- W_Q: `[d_model, h*d_k]`
- W_K: `[d_model, h*d_k]`
- W_V: `[d_model, h*d_v]`
- W_O: `[h*d_v, d_model]`
- Biases (optional): b_Q, b_K, b_V, b_O

**Intermediate Buffers**:
- Q_proj: `[B, N, h*d_k]`
- K_proj: `[B, M, h*d_k]`
- V_proj: `[B, M, h*d_v]`
- Q_heads: `[B*h, N, d_k]` (after reshape+transpose)
- K_heads: `[B*h, M, d_k]`
- V_heads: `[B*h, M, d_v]`
- Scores: `[B*h, N, M]`
- Attn_weights: `[B*h, N, M]`
- Context: `[B*h, N, d_v]`
- Output_concat: `[B, N, h*d_v]`

### 2.3 Kernel Implementations

#### Reshape and Transpose Kernel

```cuda
__global__ void reshape_transpose_BNhd_to_BhNd(
    const float* input,   // [B, N, h*d]
    float* output,        // [B*h, N, d]
    int B, int N, int h, int d
) {
    int batch_head = blockIdx.z;       // 0 to B*h-1
    int b = batch_head / h;
    int head = batch_head % h;

    int n = blockIdx.y * blockDim.y + threadIdx.y;  // 0 to N-1
    int dim = blockIdx.x * blockDim.x + threadIdx.x;  // 0 to d-1

    if (n < N && dim < d) {
        // Input index: [b, n, head*d + dim]
        int in_idx = b * N * h * d + n * h * d + head * d + dim;

        // Output index: [batch_head, n, dim]
        int out_idx = batch_head * N * d + n * d + dim;

        output[out_idx] = input[in_idx];
    }
}
```

#### Batched Softmax Kernel

```cuda
__global__ void batched_softmax_kernel(
    const float* input,   // [B, N, M]
    float* output,        // [B, N, M]
    int B, int N, int M
) {
    extern __shared__ float sdata[];

    int batch_seq = blockIdx.x;  // 0 to B*N-1
    int tid = threadIdx.x;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int m = tid; m < M; m += blockDim.x) {
        float val = input[batch_seq * M + m];
        max_val = fmaxf(max_val, val);
    }

    sdata[tid] = max_val;
    __syncthreads();

    // Reduce to find global max
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

    // Reduce to find global sum
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
```

#### Scaled Dot-Product Attention

This combines several operations:

```cpp
void scaled_dot_product_attention(
    const float* Q,        // [B*h, N, d_k]
    const float* K,        // [B*h, M, d_k]
    const float* V,        // [B*h, M, d_v]
    float* output,         // [B*h, N, d_v]
    float* scores,         // [B*h, N, M] temporary
    float* attn_weights,   // [B*h, N, M] temporary
    int Bh, int N, int M, int d_k, int d_v
) {
    float scale = 1.0f / sqrtf((float)d_k);

    // 1. Scores = Q · K^T
    matmul_transB(Q, K, scores, Bh * N, d_k, M);

    // 2. Scale scores
    scale_matrix(scores, scale, Bh * N * M);

    // 3. Softmax
    batched_softmax(scores, attn_weights, Bh, N, M);

    // 4. Output = Attn · V
    matmul(attn_weights, V, output, Bh * N, M, d_v);
}
```

### 2.4 Multi-Head Attention Class

```cpp
class MultiHeadAttention {
public:
    MultiHeadAttention(
        int d_model,       // Model dimension
        int num_heads,     // Number of attention heads
        int max_seq_len,   // Maximum sequence length
        int max_batch_size // Maximum batch size
    );

    ~MultiHeadAttention();

    // Forward pass
    // X: [B, N, d_model] input
    // output: [B, N, d_model]
    // mask: optional [B, N, M] attention mask (0 = attend, -inf = ignore)
    void forward(
        const float* h_X,       // Host input
        float* h_output,        // Host output
        int batch_size,
        int seq_len,
        const float* h_mask = nullptr
    );

    // Self-attention (Q=K=V=X)
    void self_attention(
        const float* h_X,
        float* h_output,
        int batch_size,
        int seq_len,
        const float* h_mask = nullptr
    );

private:
    int d_model;
    int num_heads;
    int d_k, d_v;  // d_k = d_v = d_model / num_heads
    int max_seq_len;
    int max_batch_size;

    // Parameters
    float *d_W_Q, *d_W_K, *d_W_V, *d_W_O;
    float *d_b_Q, *d_b_K, *d_b_V, *d_b_O;

    // Buffers
    float *d_Q_proj, *d_K_proj, *d_V_proj;
    float *d_Q_heads, *d_K_heads, *d_V_heads;
    float *d_scores, *d_attn_weights;
    float *d_context, *d_output_concat;

    void allocate_memory();
    void free_memory();
};
```

## 3. Optimization Strategies

### 3.1 Memory Access Patterns

- **Coalesced Access**: Ensure adjacent threads access adjacent memory locations
- **Shared Memory**: Use shared memory for softmax reduction
- **Minimize Copies**: Keep data on GPU between operations

### 3.2 Kernel Fusion

Potential fusions:
1. Reshape + Transpose → Single kernel
2. MatMul + Scale → Combined operation
3. Softmax + MatMul → Fused attention kernel

### 3.3 Flash Attention (Advanced)

For very long sequences, implement Flash Attention:
- Tile the attention computation
- Recompute on-the-fly instead of storing all scores
- Reduces memory from O(N²) to O(N)

## 4. Testing Strategy

### 4.1 Unit Tests

1. **Reshape/Transpose**: Verify correct dimension permutation
2. **Batched Softmax**: Compare to CPU implementation
3. **Scaled Dot-Product**: Test with known attention patterns
4. **Multi-Head Attention**: Verify output shape and values

### 4.2 Numerical Tests

1. **Attention to uniform distribution** (all ones)
2. **Attention to single position** (one-hot)
3. **Positional encoding compatibility**
4. **Gradient checking** (for future backward pass)

### 4.3 Integration Tests

1. **Self-attention on short sequences**
2. **Cross-attention (encoder-decoder)**
3. **Masked attention (causal/autoregressive)**

## 5. Performance Targets

For typical configurations (B=32, N=512, d_model=512, h=8):

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| QKV Projection | < 1ms | 3 matrix multiplications |
| Reshape/Transpose | < 0.2ms | Memory-bound |
| Attention Computation | < 2ms | Dominant operation |
| Output Projection | < 0.5ms | Single matmul |
| **Total Forward Pass** | **< 4ms** | On modern GPU (RTX 3090) |

Memory usage: ~100MB for parameters and intermediate buffers

## 6. Future Extensions

1. **Backward Pass**: Implement gradients for training
2. **Flash Attention**: Memory-efficient for long sequences
3. **Sparse Attention**: For very long sequences (block-sparse, local)
4. **Relative Position Encodings**: T5-style relative attention
5. **Multi-Query Attention**: Faster inference variant
6. **Grouped-Query Attention**: Balance between MHA and MQA

## 7. References

1. Vaswani et al. (2017). "Attention Is All You Need"
2. Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention"
3. Triton Implementation: https://github.com/openai/triton
4. PyTorch MultiheadAttention: Reference implementation
