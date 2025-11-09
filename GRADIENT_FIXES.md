# Gradient Clipping and Training Stability Fixes

## Problem
Rising loss during training indicated the model was actively getting worse, likely due to:
1. Learning rate too high (0.001 was too aggressive)
2. No gradient clipping (transformers are prone to gradient explosion)
3. Lack of diagnostic information to identify issues

## Changes Implemented

### 1. Gradient Clipping System (NEW)

**Files Added:**
- `include/gradient_utils.h` - Header for gradient utilities
- `src/gradient_utils.cu` - CUDA implementation with kernels for:
  - Computing L2 norm of gradient tensors
  - Computing global gradient norm across all parameters
  - Clipping gradients by global norm (max_norm = 1.5)
  - Computing parameter norms for monitoring

**Implementation:**
- Global gradient norm is computed across ALL model parameters
- If norm exceeds 1.5, all gradients are scaled proportionally
- Slightly relaxed from 1.0 to allow more gradient signal through
- This prevents gradient explosion while maintaining gradient direction

### 2. Enhanced Training Diagnostics

**Modified: `src/transformer.cu`**
- Added gradient norm logging (before and after clipping)
- Added parameter norm logging for monitoring parameter scale
- Diagnostics printed every 10 training steps:
  ```
  [Step X] Loss: X.XXXX | Grad norm: X.XXXX -> X.XXXX (clipped: YES/NO) | Param norm: X.XXXX | LR: X.XXXXXX
  ```

### 3. Learning Rate Reduction

**Modified: `examples/train_transformer_impl.h`**
- **Reduced base learning rate: 0.001 → 0.00005 (5e-5)** (20x reduction)
- This is much safer for transformer training from scratch
- Further reduced after initial testing showed instability

### 4. Learning Rate Warmup Schedule

**Modified: `examples/train_transformer_impl.h`**
- Added linear warmup over first 200 steps (extended from 100)
- Warmup schedule: starts at 1e-6, linearly increases to 5e-5
- Prevents early training instability from large gradient updates
- Longer warmup provides more gradual introduction to full learning rate
- Formula: `lr = warmup_init_lr + (step/warmup_steps) * (base_lr - warmup_init_lr)`

### 5. Build System Update

**Modified: `CMakeLists.txt`**
- Added `src/gradient_utils.cu` to the build

## Expected Results

With these changes, you should see:

1. **Gradient Clipping in Action**
   - Early in training, gradients may exceed 1.0 and get clipped
   - As training stabilizes, clipping should become less frequent
   - The diagnostic output shows when clipping occurs

2. **Stable Loss Decrease**
   - Loss should consistently decrease (not increase!)
   - Lower learning rate prevents overshooting
   - Warmup prevents early instability

3. **Better Monitoring**
   - Gradient norms tell you about gradient health
   - Parameter norms tell you about model scale
   - Can easily spot if gradients explode or vanish

## Usage

Build and run as usual:
```bash
cd build
cmake ..
make train_transformer
./train_transformer --data your_data.txt
```

For Weights & Biases logging:
```bash
./train_transformer --data your_data.txt --wandb --wandb-project "my-project"
```

## Diagnostic Interpretation

**Gradient Norms:**
- < 0.1: Gradients may be too small (vanishing gradients)
- 0.1 - 1.0: Healthy gradient range
- \> 1.0: Gets clipped (normal early in training)
- Consistently \>> 10 before clipping: Consider lowering LR further

**Parameter Norms:**
- Should grow gradually during training
- Sudden jumps indicate instability
- Roughly proportional to sqrt(num_parameters)

**Learning Rate:**
- First 100 steps: linearly increases from 1e-6 to 1e-4
- After step 100: constant at 1e-4
- Shows current LR in diagnostic output

## Technical Details

### Gradient Clipping Algorithm
```
1. Compute global_norm = sqrt(sum(||grad_i||^2 for all gradients))
2. If global_norm > max_norm:
   scale = max_norm / global_norm
   For each gradient: grad_i *= scale
```

This is the same approach used by major frameworks (PyTorch's `clip_grad_norm_`, TensorFlow's `clip_by_global_norm`).

### Why These Specific Values?

- **max_norm = 1.5**: Slightly relaxed from standard 1.0, allows more gradient signal while still preventing explosion
- **base_lr = 5e-5**: Very conservative for maximum stability (half the typical default)
- **warmup_steps = 200**: Extended warmup for gradual learning rate increase
- **warmup_init_lr = 1e-6**: Very conservative start, prevents early spikes

## Next Steps

**Current Configuration (v2 - More Conservative):**
- base_lr = 5e-5 (reduced from 1e-4)
- warmup_steps = 200 (increased from 100)
- max_grad_norm = 1.5 (relaxed from 1.0)

If loss still rises or oscillates excessively:
1. Try reducing base_lr to 3e-5 or 1e-5
2. Increase warmup_steps to 300-500
3. Check the loss gradient kernel implementation (potential bug in recent rewrite)
4. Verify data preprocessing (tokens, targets alignment)

If loss is decreasing but very slowly:
1. Try increasing base_lr to 7e-5 or 1e-4
2. Monitor gradient norms - if they're always < 0.3, LR might be too low
3. Consider reducing warmup_steps back to 100

## Files Modified Summary

1. **include/gradient_utils.h** (NEW)
2. **src/gradient_utils.cu** (NEW)
3. **src/transformer.cu** (MODIFIED)
   - Added gradient clipping
   - Added diagnostic logging
4. **examples/train_transformer_impl.h** (MODIFIED)
   - Reduced learning rate
   - Added warmup schedule
5. **CMakeLists.txt** (MODIFIED)
   - Added gradient_utils.cu to build
