# Mathematical Derivation: Batched MLP with Adam Optimizer

## 1. Problem Setup

We consider a 2-hidden-layer Multi-Layer Perceptron (MLP) with the following architecture:

- **Input layer**: size `h1`
- **Hidden layer 1**: size `h2`
- **Hidden layer 2**: size `h3`
- **Output layer**: size `h4`

For batched processing, we process `B` samples simultaneously.

### Network Parameters

- **W1**: Weight matrix connecting input to hidden layer 1, shape `[h2 × h1]`
- **b1**: Bias vector for hidden layer 1, shape `[h2]`
- **W2**: Weight matrix connecting hidden layer 1 to hidden layer 2, shape `[h3 × h2]`
- **b2**: Bias vector for hidden layer 2, shape `[h3]`
- **W3**: Weight matrix connecting hidden layer 2 to output, shape `[h4 × h3]`
- **b3**: Bias vector for output layer, shape `[h4]`

### Batch Data

- **X**: Input batch, shape `[B × h1]`
- **Y**: Target batch, shape `[B × h4]`

## 2. Forward Pass

We use ReLU activation for hidden layers and no activation (or softmax/sigmoid depending on task) for the output layer.

### Layer 1
```
Z1 = X · W1^T + b1        # Shape: [B × h2]
A1 = ReLU(Z1)              # Shape: [B × h2]
```

Where:
- `X · W1^T` is matrix multiplication: each row of X is multiplied by W1^T
- Broadcasting adds b1 to each row
- `ReLU(z) = max(0, z)` applied element-wise

### Layer 2
```
Z2 = A1 · W2^T + b2       # Shape: [B × h3]
A2 = ReLU(Z2)              # Shape: [B × h3]
```

### Output Layer
```
Z3 = A2 · W3^T + b3       # Shape: [B × h4]
Y_pred = Z3                # Shape: [B × h4] (for regression)
```

For classification with softmax:
```
Y_pred = softmax(Z3)       # Applied row-wise
```

### Loss Function

For Mean Squared Error (MSE) regression:
```
L = (1/(2B)) · Σ_i Σ_j (Y_pred[i,j] - Y[i,j])²
```

For cross-entropy classification:
```
L = -(1/B) · Σ_i Σ_j Y[i,j] · log(Y_pred[i,j])
```

## 3. Backward Pass (Backpropagation)

We derive gradients using the chain rule. Shapes are indicated for clarity.

### Output Layer Gradients

For MSE loss:
```
dL/dZ3 = (1/B) · (Y_pred - Y)                    # Shape: [B × h4]
```

Gradient w.r.t. W3:
```
dL/dW3 = (dL/dZ3)^T · A2                         # Shape: [h4 × B] · [B × h3] = [h4 × h3]
```

Gradient w.r.t. b3:
```
dL/db3 = Σ_i (dL/dZ3)[i,:]                       # Sum over batch: [h4]
```

### Hidden Layer 2 Gradients

Gradient flowing back to A2:
```
dL/dA2 = (dL/dZ3) · W3                           # Shape: [B × h4] · [h4 × h3] = [B × h3]
```

ReLU gradient:
```
dL/dZ2 = dL/dA2 ⊙ (Z2 > 0)                      # Element-wise product, [B × h3]
```
Where `⊙` denotes element-wise multiplication and `(Z2 > 0)` is a binary mask.

Gradient w.r.t. W2:
```
dL/dW2 = (dL/dZ2)^T · A1                         # Shape: [h3 × B] · [B × h2] = [h3 × h2]
```

Gradient w.r.t. b2:
```
dL/db2 = Σ_i (dL/dZ2)[i,:]                       # Sum over batch: [h3]
```

### Hidden Layer 1 Gradients

Gradient flowing back to A1:
```
dL/dA1 = (dL/dZ2) · W2                           # Shape: [B × h3] · [h3 × h2] = [B × h2]
```

ReLU gradient:
```
dL/dZ1 = dL/dA1 ⊙ (Z1 > 0)                      # Element-wise product, [B × h2]
```

Gradient w.r.t. W1:
```
dL/dW1 = (dL/dZ1)^T · X                          # Shape: [h2 × B] · [B × h1] = [h2 × h1]
```

Gradient w.r.t. b1:
```
dL/db1 = Σ_i (dL/dZ1)[i,:]                       # Sum over batch: [h2]
```

## 4. Adam Optimizer Update

Adam (Adaptive Moment Estimation) combines momentum and RMSprop. For each parameter θ (e.g., W1, b1, etc.):

### Hyperparameters
- `α`: Learning rate (e.g., 0.001)
- `β1`: Exponential decay rate for first moment (e.g., 0.9)
- `β2`: Exponential decay rate for second moment (e.g., 0.999)
- `ε`: Small constant for numerical stability (e.g., 1e-8)
- `t`: Time step (iteration counter)

### Update Equations

For each parameter θ with gradient g = dL/dθ:

1. **Update biased first moment estimate (momentum)**:
```
m_t = β1 · m_{t-1} + (1 - β1) · g
```

2. **Update biased second moment estimate (RMSprop)**:
```
v_t = β2 · v_{t-1} + (1 - β2) · g²
```
Where `g²` is element-wise square.

3. **Compute bias-corrected first moment**:
```
m̂_t = m_t / (1 - β1^t)
```

4. **Compute bias-corrected second moment**:
```
v̂_t = v_t / (1 - β2^t)
```

5. **Update parameter**:
```
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

### Batched Adam Update

For our MLP, we maintain moment estimates for all parameters:
- `(m_W1, v_W1)` for W1
- `(m_b1, v_b1)` for b1
- `(m_W2, v_W2)` for W2
- `(m_b2, v_b2)` for b2
- `(m_W3, v_W3)` for W3
- `(m_b3, v_b3)` for b3

Each update step:
1. Compute all gradients via backpropagation
2. Update all moment estimates
3. Apply bias correction
4. Update all parameters

## 5. Complete Algorithm Summary

### Training Step (one iteration)

**Input**: Batch X `[B × h1]`, targets Y `[B × h4]`

**Forward Pass**:
1. `Z1 = X · W1^T + b1, A1 = ReLU(Z1)`
2. `Z2 = A1 · W2^T + b2, A2 = ReLU(Z2)`
3. `Z3 = A2 · W3^T + b3, Y_pred = Z3`
4. Compute loss: `L = MSE(Y_pred, Y)`

**Backward Pass**:
1. `dZ3 = (1/B) · (Y_pred - Y)`
2. `dW3 = dZ3^T · A2, db3 = sum(dZ3, axis=0)`
3. `dA2 = dZ3 · W3, dZ2 = dA2 ⊙ (Z2 > 0)`
4. `dW2 = dZ2^T · A1, db2 = sum(dZ2, axis=0)`
5. `dA1 = dZ2 · W2, dZ1 = dA1 ⊙ (Z1 > 0)`
6. `dW1 = dZ1^T · X, db1 = sum(dZ1, axis=0)`

**Adam Update** (for each parameter θ ∈ {W1, b1, W2, b2, W3, b3}):
1. `m = β1 · m + (1 - β1) · dθ`
2. `v = β2 · v + (1 - β2) · dθ²`
3. `m̂ = m / (1 - β1^t)`
4. `v̂ = v / (1 - β2^t)`
5. `θ = θ - α · m̂ / (√v̂ + ε)`

### Inference (forward pass only)

**Input**: Batch X `[B × h1]`

**Output**: Y_pred `[B × h4]`

Simply perform the forward pass without computing gradients.

## 6. Matrix Multiplication Details

For CUDA implementation, matrix multiplication is the core operation. Consider `C = A · B`:
- A has shape `[M × K]`
- B has shape `[K × N]`
- C has shape `[M × N]`

The computation:
```
C[i,j] = Σ_k A[i,k] · B[k,j]
```

For efficient GPU implementation, we use:
1. **Thread blocks**: Each block computes a tile of the output
2. **Shared memory**: Cache tiles of A and B to reduce global memory access
3. **Thread cooperation**: Threads in a block collaboratively load tiles

This achieves much better performance than naive element-wise computation.
