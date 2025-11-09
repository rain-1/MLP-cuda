# Debugging Guide for MLP-CUDA Transformer

This guide explains how to use the debugging infrastructure to quickly diagnose and fix training issues.

## Quick Start: Debugging a Training Failure

When training fails (exploding/vanishing gradients, NaN values, etc.), follow this workflow:

### 1. Run Toy Problem Tests First

```bash
# Build and run toy problem tests
./build/test_toy_problems

# These tests verify:
# - Can the model copy a sequence?
# - Can the model overfit a single batch?
# - Does the model remain numerically stable?
```

If toy problems fail, the issue is in the core architecture, not hyperparameters.

### 2. Run Unit Tests

```bash
# Test individual components
./build/test_ffn_gradients        # Test FeedForwardNetwork
./build/test_attention_gradients   # Test MultiHeadAttention (if implemented)
./build/test_layer_norm           # Test LayerNorm (if implemented)
```

Unit tests with gradient checking will pinpoint which component has broken gradients.

### 3. Compare with PyTorch Reference

```bash
# Run PyTorch reference tests
python tests/pytorch_reference.py

# This verifies:
# - Forward pass matches expected behavior
# - Backward pass computes correct gradients
# - Model can overfit simple tasks
```

If PyTorch passes but CUDA fails, the issue is in the CUDA implementation.

### 4. Use Enhanced Monitoring

Enable detailed monitoring in your training code:

```cpp
#include "debug_monitor.h"

// Create monitor
ModelMonitor monitor;

// In training loop
monitor.add_activation("L0_ffn_out", d_ffn_output, size);
monitor.add_gradient("L0_W1", d_grad_W1, size);

// Check health every N steps
if (step % 100 == 0) {
    monitor.check_health(step);
    monitor.print_status();
}
```

## Debugging Tools

### 1. Gradient Checking

Verifies analytical gradients match numerical gradients:

```cpp
#include "gradient_check.h"

// Define loss function
auto loss_fn = [&](const float* d_param) -> float {
    // Forward pass with perturbed param
    // Return scalar loss
};

// Check gradients
GradientCheckResult result = check_gradients_verbose(
    loss_fn,
    d_W1,              // Parameter to check
    d_grad_W1,         // Analytical gradient
    param_size,        // Number of parameters
    1e-4f,             // Epsilon for finite difference
    1e-3f              // Relative error threshold
);

if (!result.passed) {
    printf("Gradient check failed!\n");
    printf("Max relative error: %.6e\n", result.max_rel_error);
}
```

### 2. Tensor Comparison

Compare CUDA outputs with expected values:

```cpp
// Compare two tensors
bool match = tensors_close(d_output, d_expected, size, 1e-3f, 1e-5f);

// Detailed comparison
compare_tensors(d_output, d_expected, size, "CUDA", "Expected");
```

### 3. Statistics Monitoring

Track tensor statistics over time:

```cpp
ActivationMonitor monitor("Layer0_FFN");

// Record during training
monitor.record(d_activation, size, step, "after_gelu");

// Print summary
monitor.print_summary();

// Save to file for plotting
monitor.save_to_file("layer0_ffn_stats.csv");
```

### 4. Failure Detection

Automatically detect common failure modes:

```cpp
FailureDetector::FailureType failure =
    FailureDetector::detect_failure(d_data, size, is_gradient);

if (failure != FailureDetector::NONE) {
    printf("Detected: %s\n", FailureDetector::failure_type_str(failure));
    FailureDetector::diagnose(d_data, size, "Layer0_W1");
}
```

### 5. Checkpointing for Replay

Save state when failure occurs:

```cpp
#include "debug_monitor.h"

// Save checkpoint
std::vector<std::pair<const char*, const float*>> tensors = {
    {"W1", d_W1},
    {"W2", d_W2},
    {"activations", d_activations}
};
std::vector<int> sizes = {W1_size, W2_size, act_size};

DebugCheckpoint::save_checkpoint("checkpoint_step_1000.bin", step, tensors, sizes);

// Later: load and replay
int loaded_step;
std::vector<float*> d_tensors;
std::vector<int> loaded_sizes;

if (DebugCheckpoint::load_checkpoint("checkpoint_step_1000.bin",
                                     loaded_step, d_tensors, loaded_sizes)) {
    // Replay forward/backward at exact failure point
    printf("Loaded checkpoint from step %d\n", loaded_step);
}
```

## Common Issues and Solutions

### Issue: Forward Activations Exploding

**Symptoms:**
- Activation values grow exponentially (100 → 1000 → 10000)
- Happens in deeper layers
- FFN activations larger than attention

**Diagnosis:**
```cpp
FailureDetector::diagnose(d_ffn_output, size, "FFN_Output");
// Will show: "Exploding Activations"
```

**Solutions:**
1. Check initialization scale
2. Verify residual scaling is applied
3. Add layer normalization
4. Reduce learning rate
5. Add gradient clipping

### Issue: Gradients Vanishing

**Symptoms:**
- Gradient norms very small (< 1e-6)
- Early layers have near-zero gradients
- Loss doesn't decrease

**Diagnosis:**
```cpp
for (each layer) {
    float grad_norm = gradient_norm(d_grad, size);
    printf("Layer %d gradient norm: %.6e\n", i, grad_norm);
}
```

**Solutions:**
1. Check for dead ReLUs
2. Increase learning rate
3. Check initialization
4. Verify backprop implementation

### Issue: NaN or Inf Values

**Symptoms:**
- Loss becomes NaN
- Sudden spike in activations
- Happens after several steps

**Diagnosis:**
```cpp
TensorStatistics stats = compute_statistics(d_data, size, "Activation");
if (stats.num_nans > 0 || stats.num_infs > 0) {
    printf("Found %d NaNs, %d Infs\n", stats.num_nans, stats.num_infs);
}
```

**Solutions:**
1. Add numerical stability epsilon to divisions
2. Clip softmax inputs
3. Check for overflow in exponentials
4. Reduce learning rate

## Systematic Debugging Workflow

1. **Isolate the failure:**
   ```bash
   # Does 1 layer work?
   ./test_single_layer

   # Does 2 layers work?
   ./test_two_layers

   # When does it break?
   ```

2. **Verify each component:**
   ```bash
   # Test components individually
   ./test_ffn_gradients
   ./test_attention_gradients
   ./test_layer_norm
   ```

3. **Compare to reference:**
   ```bash
   # PyTorch should pass if architecture is correct
   python tests/pytorch_reference.py
   ```

4. **Pinpoint the layer:**
   ```cpp
   // Monitor each layer
   for (int i = 0; i < num_layers; i++) {
       monitor.add_activation(layer_names[i], d_activations[i], sizes[i]);
   }
   monitor.check_health(step);
   // Will show which layer fails first
   ```

5. **Check gradients numerically:**
   ```cpp
   // Verify backprop for failing layer
   check_gradients_verbose(loss_fn, d_param, d_grad, size);
   ```

6. **Replay at failure point:**
   ```cpp
   // Save state before failure
   DebugCheckpoint::save_checkpoint(...);

   // Load and inspect exactly what's happening
   DebugCheckpoint::load_checkpoint(...);
   ```

## Performance Notes

- Gradient checking is **very slow** (O(n²)). Only use on small models/tests.
- Statistics computation is moderate cost. Enable only for debugging.
- Monitoring adds ~5-10% overhead. Disable in production training.
- Checkpointing can be done infrequently (every 100-1000 steps).

## Testing Best Practices

### When Adding New Components

1. **Write unit test with gradient checking**
   ```cpp
   bool test_new_component() {
       // Setup
       // Forward pass
       // Backward pass
       // Gradient check
       GradientCheckResult result = check_gradients(...);
       return result.passed;
   }
   ```

2. **Test on toy problem**
   ```cpp
   // Can it learn identity function?
   // Can it overfit single batch?
   ```

3. **Compare with PyTorch**
   ```python
   # Implement same component in PyTorch
   # Compare outputs for same inputs
   ```

4. **Ablation test**
   ```cpp
   // Does model work without this component?
   // Does it break with this component?
   ```

### Before Training on Real Data

- [ ] All unit tests pass
- [ ] Toy problems pass (copy, overfit)
- [ ] PyTorch reference passes
- [ ] Gradient checks pass
- [ ] No NaN/Inf in forward pass
- [ ] Gradients are finite
- [ ] Model can overfit 1 batch in < 1000 steps

## Example: Full Debugging Session

```bash
# 1. Training fails - activations explode at step 850

# 2. Run toy problems to verify basic functionality
./build/test_toy_problems
# Result: Copy task passes, overfit fails after 800 steps

# 3. Run unit tests
./build/test_ffn_gradients
# Result: FFN gradients pass

# 4. Enable monitoring and rerun
# (Add monitoring code to training loop)
./build/train_transformer
# Output shows: Layer 2 FFN output explodes at step 850

# 5. Check Layer 2 FFN initialization
# (Inspect initialization code)
# Found: Layer 2 missing residual scaling!

# 6. Fix and verify
# (Add residual scaling)
./build/test_toy_problems
# Result: All tests pass!

# 7. Train on real data
./build/train_transformer
# Success!
```

## Summary

**Quick diagnostic commands:**
```bash
# Basic sanity
./build/test_toy_problems

# Component verification
./build/test_ffn_gradients

# Reference comparison
python tests/pytorch_reference.py

# Full debugging
# (Add monitoring to training code)
./build/train_transformer
```

**Most valuable tools:**
1. **Toy problems** - Fast check if architecture works at all
2. **Gradient checking** - Catches backprop bugs immediately
3. **PyTorch reference** - Confirms expected behavior
4. **Failure detection** - Automatically identifies issue type
5. **Layer-by-layer monitoring** - Pinpoints exact location

With these tools, you should be able to diagnose any training issue in **minutes** instead of hours.
