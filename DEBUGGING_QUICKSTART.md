# Debugging Quick Start

**When training fails, follow these steps in order:**

## 1. Run Toy Problems (2 minutes)

```bash
mkdir -p build && cd build
cmake ..
make test_toy_problems
./test_toy_problems
```

**What this tells you:**
- ✓ All pass → Architecture is correct, issue is hyperparameters/data
- ✗ Copy fails → Basic forward/backward broken
- ✗ Overfit fails → Model can't learn at all
- ✗ Stability fails → Numerical issues (NaN/Inf/explosion)

## 2. Run Unit Tests (5 minutes)

```bash
make test_ffn_gradients
./test_ffn_gradients
```

**What this tells you:**
- ✗ Forward shape fails → Memory/dimension bug
- ✗ Gradient check fails → Backprop implementation wrong
- Look at max relative error to see how bad it is

## 3. Compare with PyTorch (3 minutes)

```bash
# From project root
python tests/pytorch_reference.py
```

**What this tells you:**
- ✓ PyTorch passes, CUDA fails → Implementation bug in CUDA
- ✗ Both fail → Architecture issue

## 4. Add Monitoring (10 minutes)

Edit your training code:

```cpp
#include "debug_monitor.h"

ModelMonitor monitor;

// In training loop, after forward pass:
monitor.add_activation("emb", d_embeddings, emb_size);
monitor.add_activation("L0_attn", d_attn_out, attn_size);
monitor.add_activation("L0_ffn", d_ffn_out, ffn_size);
// ... add more layers ...

// After backward pass:
monitor.add_gradient("L0_W1", d_grad_W1, w1_size);
monitor.add_gradient("L0_W2", d_grad_W2, w2_size);

// Check every 100 steps:
if (step % 100 == 0) {
    monitor.check_health(step);
    monitor.print_status();
}
```

Recompile and run:
```bash
make train_transformer
./train_transformer
```

**What this tells you:**
- Exactly which layer starts to fail first
- What type of failure (explosion/vanishing/NaN)
- At what step it happens

## Common Patterns

### Pattern 1: Exploding in deeper layers
```
✓ Step 0: All healthy
✓ Step 100: All healthy
⚠ Step 200: L2_ffn exploding (max=150)
✗ Step 300: L2_ffn exploding (max=1500), NaN in L3
```
**Diagnosis:** FFN in layer 2 not properly scaled
**Fix:** Check initialization scale for L2

### Pattern 2: Gradients vanish immediately
```
✓ Forward pass: All healthy
✗ Backward pass: L0 grad=1e-8, L1 grad=1e-9, L2 grad=1e-10
```
**Diagnosis:** Gradient doesn't flow back through layers
**Fix:** Check activation functions, verify backprop chain rule

### Pattern 3: Sudden NaN after N steps
```
✓ Step 850: Loss=5.2, max_activation=45
✗ Step 851: Loss=NaN, max_activation=Inf
```
**Diagnosis:** One bad gradient update broke weights
**Fix:** Add gradient clipping, reduce learning rate

## Quick Fixes to Try

1. **Exploding activations:**
   - Reduce learning rate by 10x
   - Add/tighten gradient clipping
   - Check residual scaling is applied

2. **Vanishing gradients:**
   - Increase learning rate
   - Check for dead ReLUs
   - Verify backprop implementation

3. **NaN values:**
   - Add gradient clipping
   - Check for division by zero
   - Add epsilon to denominators
   - Reduce learning rate

4. **Can't overfit:**
   - Model too small
   - Learning rate too low
   - Backprop bug (check gradients)

## Full Debugging Guide

See [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) for comprehensive documentation.

## Emergency Diagnostic Commands

```bash
# Basic sanity check (run this first)
./test_toy_problems

# Verify gradients are correct
./test_ffn_gradients

# Compare with reference
python tests/pytorch_reference.py

# Run all tests
make test
ctest
```

## Time Estimates

- Toy problems: 2 min
- Unit tests: 5 min per component
- PyTorch comparison: 3 min
- Add monitoring: 10 min
- Full debug session: 30-60 min

**Without these tools:** Hours to days of blind trial and error.
