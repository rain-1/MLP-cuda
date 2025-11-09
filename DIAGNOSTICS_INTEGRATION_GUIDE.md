# Numerical Precision & Training Diagnostics Integration Guide

This guide explains how to integrate the diagnostic systems into your training code to debug gradient and precision issues.

## Overview

Two diagnostic systems have been created:

1. **Numerical Precision Analysis** (`include/precision_analysis.h`)
   - Kahan summation for improved accuracy
   - Softmax precision comparison (float32 vs float64)
   - Gradient precision analysis
   - Numerical stability metrics

2. **Training Diagnostics** (`include/training_diagnostics.h`)
   - Real-time gradient statistics tracking
   - Layer-by-layer gradient flow analysis
   - Adam optimizer state monitoring
   - Gradient clipping statistics
   - CSV logging for long-term analysis

## Quick Start

### 1. Build the Test Suite

```bash
# Add to your Makefile or CMakeLists.txt
g++ -std=c++17 -O2 \
    tests/test_diagnostics.cpp \
    src/training_diagnostics.cpp \
    -I./include \
    -o test_diagnostics

# Run the tests
./test_diagnostics
```

### 2. Basic Integration Example

```cpp
#include "precision_analysis.h"
#include "training_diagnostics.h"

using namespace PrecisionAnalysis;
using namespace TrainingDiagnostics;

// Initialize diagnostics system
TrainingSessionDiagnostics diagnostics;

// Register all layers (do this once at startup)
diagnostics.register_layer("output", num_output_params);
diagnostics.register_layer("transformer_3", num_transformer_params);
diagnostics.register_layer("transformer_2", num_transformer_params);
diagnostics.register_layer("transformer_1", num_transformer_params);
diagnostics.register_layer("transformer_0", num_transformer_params);
diagnostics.register_layer("embedding", num_embedding_params);

// Enable CSV logging (optional but recommended)
diagnostics.enable_logging("./training_logs");
```

### 3. During Training Loop

```cpp
for (int step = 0; step < max_steps; step++) {
    // Forward pass
    float loss = forward(model, input, target);

    // Backward pass
    backward(model);

    // DIAGNOSTIC: Record training metrics
    float grad_norm = compute_gradient_norm(model.gradients);
    diagnostics.record_step(step, loss, learning_rate, grad_norm);

    // DIAGNOSTIC: Track layer gradients
    diagnostics.update_layer_gradients(0, model.output_layer.gradients);
    diagnostics.update_layer_gradients(1, model.transformer[3].gradients);
    diagnostics.update_layer_gradients(2, model.transformer[2].gradients);
    diagnostics.update_layer_gradients(3, model.transformer[1].gradients);
    diagnostics.update_layer_gradients(4, model.transformer[0].gradients);
    diagnostics.update_layer_gradients(5, model.embedding.gradients);

    // Gradient clipping
    float norm_before = grad_norm;
    if (grad_norm > max_grad_norm) {
        clip_gradients(model.gradients, max_grad_norm);
    }
    float norm_after = compute_gradient_norm(model.gradients);

    // DIAGNOSTIC: Record clipping
    diagnostics.record_clipping(norm_before, norm_after, max_grad_norm);

    // Optimizer step
    adam_step(optimizer, model.params, model.gradients);

    // DIAGNOSTIC: Monitor Adam state (sample one layer to avoid overhead)
    if (step % 10 == 0) {  // Only every 10 steps
        diagnostics.record_adam_state(
            optimizer.m,      // First moment
            optimizer.v,      // Second moment
            optimizer.updates, // Parameter updates
            model.params,     // Current parameters
            num_params
        );
    }

    // Print diagnostics
    if (step % 100 == 0) {
        diagnostics.print_quick_summary();
    }

    if (step % 1000 == 0) {
        diagnostics.print_diagnostics();  // Detailed report
    }
}
```

## Specific Use Cases

### Debugging Softmax/Cross-Entropy Gradients

```cpp
#include "precision_analysis.h"

// Compare float32 vs float64 softmax computation
float logits[34];  // Your logits from model
auto result = SoftmaxPrecisionAnalyzer::analyze(logits, 34);
SoftmaxPrecisionAnalyzer::print_comparison(result);

// Compare gradient computation in different precisions
float grad_f32[34];
double grad_f64[34];
int target = 5;

GradientPrecisionAnalyzer::compare_cross_entropy_gradient(
    logits, 34, target,
    grad_f32, grad_f64
);

// Analyze the difference
for (int i = 0; i < 34; i++) {
    float diff = std::abs(grad_f32[i] - (float)grad_f64[i]);
    if (diff > 1e-6f) {
        std::cout << "Token " << i << ": diff = " << diff << "\n";
    }
}
```

### Testing Kahan Summation in Your Loss Kernel

You can modify your loss kernel to use Kahan summation:

```cuda
// In your CUDA kernel
__global__ void compute_loss_gradient_kahan(
    const float* logits,
    const int* targets,
    float* d_logits,
    int batch_size,
    int seq_len,
    int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * seq_len) return;

    const float* logits_ptr = logits + idx * vocab_size;
    float* grad_ptr = d_logits + idx * vocab_size;
    int target = targets[idx];

    // Find max for numerical stability
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        max_logit = fmaxf(max_logit, logits_ptr[i]);
    }

    // Kahan summation for exp accumulation
    float sum_exp = 0.0f;
    float c = 0.0f;  // Compensation

    for (int i = 0; i < vocab_size; i++) {
        float exp_val = expf(logits_ptr[i] - max_logit);
        float y = exp_val - c;
        float t = sum_exp + y;
        c = (t - sum_exp) - y;
        sum_exp = t;
    }

    // Compute gradients
    for (int i = 0; i < vocab_size; i++) {
        float softmax = expf(logits_ptr[i] - max_logit) / sum_exp;
        grad_ptr[i] = softmax - (i == target ? 1.0f : 0.0f);
    }
}
```

### Detecting Vanishing/Exploding Gradients

```cpp
// After each training step, check for gradient flow issues
auto& layer_tracker = diagnostics.get_layer_tracker();
const auto& layers = layer_tracker.get_layers();

for (size_t i = 0; i < layers.size(); i++) {
    if (layers[i].is_vanishing) {
        std::cout << "⚠️  WARNING: Vanishing gradients in "
                  << layers[i].name << "\n";
        std::cout << "  Gradient scale: " << layers[i].gradient_scale
                  << "x (< 0.01x)\n";
        std::cout << "  Consider: gradient clipping, skip connections, "
                  << "or layer norm adjustments\n";
    }

    if (layers[i].is_exploding) {
        std::cout << "⚠️  WARNING: Exploding gradients in "
                  << layers[i].name << "\n";
        std::cout << "  Gradient scale: " << layers[i].gradient_scale
                  << "x (> 100x)\n";
        std::cout << "  Consider: lower learning rate or gradient clipping\n";
    }
}
```

### Monitoring Adam Optimizer Health

```cpp
auto& adam_monitor = diagnostics.get_adam_monitor();

// Check for optimizer issues
if (adam_monitor.detect_optimizer_issues()) {
    std::cout << "\n⚠️  Optimizer issues detected!\n";
    adam_monitor.print_latest();

    // Possible actions:
    // 1. Reset optimizer state
    // 2. Adjust epsilon (if v values too small)
    // 3. Check for gradient clipping issues
    // 4. Verify no inf/nan in gradients
}

// Print trend analysis every N steps
if (step % 100 == 0) {
    adam_monitor.print_trends(10);  // Last 10 steps
}
```

### Analyzing Gradient Clipping Impact

```cpp
auto& clip_monitor = diagnostics.get_clip_monitor();
const auto& history = clip_monitor.get_history();

// Calculate average signal loss from clipping
float total_signal_loss = 0.0f;
int num_clipped = 0;

for (const auto& stat : history) {
    if (stat.was_clipped) {
        total_signal_loss += (1.0f - stat.clip_ratio);
        num_clipped++;
    }
}

if (num_clipped > 0) {
    float avg_signal_loss = total_signal_loss / num_clipped * 100.0f;
    std::cout << "Average gradient signal loss from clipping: "
              << avg_signal_loss << "%\n";

    if (avg_signal_loss > 50.0f) {
        std::cout << "⚠️  WARNING: Clipping is removing >50% of gradient signal!\n";
        std::cout << "  Consider increasing clip threshold from "
                  << max_grad_norm << " to " << (max_grad_norm * 2.0f) << "\n";
    }
}
```

## Systematic Investigation of Gradient Issues

Based on the GRADIENT_BUG_ANALYSIS.md, here's how to use these tools:

### Step 1: Verify Gradient Computation Precision

```cpp
// In your gradient test
const int vocab_size = 34;
float logits[vocab_size];
int target = 5;

// Compute gradients in both precisions
float grad_f32[vocab_size];
double grad_f64[vocab_size];

GradientPrecisionAnalyzer::compare_cross_entropy_gradient(
    logits, vocab_size, target,
    grad_f32, grad_f64
);

// Check if errors are within acceptable bounds
float max_error = 0.0f;
for (int i = 0; i < vocab_size; i++) {
    float error = std::abs(grad_f32[i] - (float)grad_f64[i]);
    max_error = std::max(max_error, error);
}

std::cout << "Max gradient error (f32 vs f64): " << max_error << "\n";

if (max_error > 0.01f) {  // > 1% error
    std::cout << "❌ Float32 precision may be insufficient\n";
    std::cout << "   Consider using float64 for loss computation\n";
} else {
    std::cout << "✓ Float32 precision appears adequate\n";
}
```

### Step 2: Monitor Training Dynamics

```cpp
// Track key metrics over time
struct TrainingHealth {
    std::vector<float> loss_history;
    std::vector<float> grad_norm_history;
    std::vector<float> clip_rate_history;

    void record(float loss, float grad_norm, bool was_clipped) {
        loss_history.push_back(loss);
        grad_norm_history.push_back(grad_norm);

        // Calculate rolling clip rate (last 100 steps)
        static int window_size = 100;
        static std::deque<bool> clip_window;
        clip_window.push_back(was_clipped);
        if (clip_window.size() > window_size) {
            clip_window.pop_front();
        }

        int num_clipped = std::count(clip_window.begin(),
                                     clip_window.end(), true);
        clip_rate_history.push_back((float)num_clipped / clip_window.size());
    }

    void check_health(int step) {
        if (loss_history.size() < 100) return;

        // Check if loss is decreasing
        float recent_loss = std::accumulate(
            loss_history.end() - 10, loss_history.end(), 0.0f) / 10.0f;
        float old_loss = std::accumulate(
            loss_history.end() - 100, loss_history.end() - 90, 0.0f) / 10.0f;

        if (recent_loss >= old_loss) {
            std::cout << "⚠️  WARNING: Loss not decreasing!\n";
            std::cout << "  Old avg: " << old_loss
                      << ", Recent avg: " << recent_loss << "\n";
        }

        // Check if gradients are stable
        float recent_grad = std::accumulate(
            grad_norm_history.end() - 10, grad_norm_history.end(), 0.0f) / 10.0f;

        if (recent_grad > 10.0f) {
            std::cout << "⚠️  WARNING: Very large gradients ("
                      << recent_grad << ")\n";
        }

        // Check clipping rate
        float recent_clip_rate = clip_rate_history.back();
        if (recent_clip_rate > 0.9f) {
            std::cout << "⚠️  WARNING: >90% gradient clipping rate!\n";
            std::cout << "  This severely limits learning\n";
        }
    }
};
```

### Step 3: Root Cause Analysis

```cpp
// Use diagnostics to identify the issue
void diagnose_training_failure(TrainingSessionDiagnostics& diagnostics) {
    std::cout << "\n=== ROOT CAUSE ANALYSIS ===\n\n";

    auto& layer_tracker = diagnostics.get_layer_tracker();
    auto& clip_monitor = diagnostics.get_clip_monitor();
    auto& adam_monitor = diagnostics.get_adam_monitor();

    // Check 1: Gradient flow
    const auto& layers = layer_tracker.get_layers();
    bool has_vanishing = false;
    bool has_exploding = false;

    for (const auto& layer : layers) {
        if (layer.is_vanishing) has_vanishing = true;
        if (layer.is_exploding) has_exploding = true;
    }

    if (has_vanishing) {
        std::cout << "Issue: VANISHING GRADIENTS detected\n";
        std::cout << "Likely cause: Too many layers, improper initialization, "
                  << "or saturating activations\n";
        std::cout << "Solutions:\n";
        std::cout << "  - Add skip connections\n";
        std::cout << "  - Use proper weight initialization (Xavier/He)\n";
        std::cout << "  - Check layer norm implementation\n\n";
    }

    if (has_exploding) {
        std::cout << "Issue: EXPLODING GRADIENTS detected\n";
        std::cout << "Likely cause: High learning rate, improper initialization, "
                  << "or numerical instability\n";
        std::cout << "Solutions:\n";
        std::cout << "  - Lower learning rate\n";
        std::cout << "  - Increase gradient clipping threshold\n";
        std::cout << "  - Check for bugs in backward pass\n\n";
    }

    // Check 2: Excessive clipping
    const auto& clip_history = clip_monitor.get_history();
    if (!clip_history.empty()) {
        int recent_window = std::min(100, (int)clip_history.size());
        int num_clipped = 0;
        for (int i = clip_history.size() - recent_window;
             i < (int)clip_history.size(); i++) {
            if (clip_history[i].was_clipped) num_clipped++;
        }

        float clip_rate = (float)num_clipped / recent_window;
        if (clip_rate > 0.9f) {
            std::cout << "Issue: EXCESSIVE GRADIENT CLIPPING ("
                      << (clip_rate * 100.0f) << "%)\n";
            std::cout << "Likely cause: Gradients consistently too large\n";
            std::cout << "Solutions:\n";
            std::cout << "  - Lower learning rate (currently losing "
                      << ((1.0f - clip_history.back().clip_ratio) * 100.0f)
                      << "% of signal)\n";
            std::cout << "  - Increase clip threshold\n";
            std::cout << "  - Check for gradient computation bugs\n\n";
        }
    }

    // Check 3: Optimizer issues
    if (adam_monitor.detect_optimizer_issues()) {
        std::cout << "Issue: OPTIMIZER STATE PROBLEMS detected\n";
        adam_monitor.print_latest();
    }
}
```

## CSV Data Analysis

The diagnostics system can log to CSV for long-term analysis:

```python
# Python script to analyze the CSV logs
import pandas as pd
import matplotlib.pyplot as plt

# Load diagnostics log
df = pd.read_csv('diagnostics_logs/training_diagnostics_*.csv')

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss
axes[0, 0].plot(df['step'], df['loss'])
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Loss')

# Gradient norm
axes[0, 1].plot(df['step'], df['gradient_norm'])
axes[0, 1].set_title('Gradient Norm')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('L2 Norm')
axes[0, 1].axhline(y=1.0, color='r', linestyle='--',
                    label='Clip threshold')

# Learning rate
axes[1, 0].plot(df['step'], df['learning_rate'])
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('LR')

plt.tight_layout()
plt.savefig('training_diagnostics.png')
```

## Performance Considerations

The diagnostic tools add overhead. Here are recommendations:

1. **During debugging**: Enable all diagnostics, print every 10 steps
2. **During normal training**:
   - Record basic metrics every step
   - Update layer gradients every 10 steps
   - Record Adam state every 100 steps
   - Print summary every 1000 steps
3. **Production training**: Only log to CSV, minimal printing

```cpp
// Configure based on mode
bool debug_mode = true;  // Set via command line

if (debug_mode) {
    diagnostics.enable_logging("./debug_logs");
    print_interval = 10;
    record_interval = 1;
} else {
    diagnostics.enable_logging("./logs");
    print_interval = 1000;
    record_interval = 100;
}
```

## Next Steps

1. **Run the test suite**: `./test_diagnostics` to verify everything works
2. **Integrate into training**: Add diagnostics to your training loop
3. **Collect baseline data**: Run training with diagnostics for 1000 steps
4. **Analyze results**: Look for patterns in gradient flow, clipping, optimizer state
5. **Test hypotheses**: Based on GRADIENT_BUG_ANALYSIS.md, test float64, Kahan summation, etc.

## Troubleshooting

**Q: Compilation errors about missing headers**
- Make sure `include/` is in your include path
- Verify both `.h` files are present

**Q: CSV file not created**
- Check that the log directory exists
- Verify write permissions

**Q: Tests crash or give strange results**
- Check that vector sizes match between gradients and stats
- Verify no null pointers passed to compute functions

**Q: Diagnostics show issues but training seems fine**
- Some warnings are informational, not critical
- Focus on: inf/nan, >90% clip rate, complete gradient vanishing
