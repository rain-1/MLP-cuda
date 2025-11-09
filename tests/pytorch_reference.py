#!/usr/bin/env python3
"""
PyTorch reference implementation for comparing against CUDA implementation.

This provides exact same architecture and can be used to:
1. Compare forward pass outputs
2. Compare backward pass gradients
3. Verify loss computation
4. Debug layer by layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys

class FeedForwardNetwork(nn.Module):
    """Reference FFN matching CUDA implementation exactly."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Layers matching CUDA implementation
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        # x: [B, N, d_model]
        z1 = self.fc1(x)  # [B, N, d_ff]
        hidden = F.gelu(z1)  # GELU activation
        output = self.fc2(hidden)  # [B, N, d_model]
        return output

    def load_from_cuda_params(self, params):
        """Load parameters from CUDA model export."""
        with torch.no_grad():
            # CUDA stores weights transposed: W1 is [d_ff, d_model]
            # PyTorch Linear expects [out_features, in_features]
            self.fc1.weight.copy_(torch.from_numpy(params['W1']))
            self.fc1.bias.copy_(torch.from_numpy(params['b1']))
            self.fc2.weight.copy_(torch.from_numpy(params['W2']))
            self.fc2.bias.copy_(torch.from_numpy(params['b2']))

    def export_params_for_cuda(self):
        """Export parameters in CUDA format."""
        return {
            'W1': self.fc1.weight.detach().cpu().numpy(),
            'b1': self.fc1.bias.detach().cpu().numpy(),
            'W2': self.fc2.weight.detach().cpu().numpy(),
            'b2': self.fc2.bias.detach().cpu().numpy(),
        }


class TransformerBlock(nn.Module):
    """Reference transformer block matching CUDA implementation."""

    def __init__(self, d_model, num_heads, d_ff, residual_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.residual_scale = residual_scale

        # Components
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        # Layer norms (pre-norm architecture)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: [B, N, d_model]

        # 1. Layer norm + attention + residual
        normed = self.ln1(x)
        attn_output, _ = self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)
        x = x + self.residual_scale * attn_output

        # 2. Layer norm + FFN + residual
        normed = self.ln2(x)
        ffn_output = self.ffn(normed)
        x = x + self.residual_scale * ffn_output

        return x


class Transformer(nn.Module):
    """Reference transformer matching CUDA implementation."""

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_seq_len, residual_scale=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # Auto-compute residual scale if not provided
        if residual_scale is None:
            residual_scale = 1.0 / (2.0 * num_layers)
        self.residual_scale = residual_scale

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, residual_scale)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, token_ids):
        # token_ids: [B, N]
        B, N = token_ids.shape

        # Embeddings
        positions = torch.arange(N, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_final(x)

        # Output projection
        logits = self.output_proj(x)  # [B, N, vocab_size]

        return logits

    def compute_loss(self, token_ids, targets):
        """Compute cross-entropy loss matching CUDA implementation."""
        logits = self.forward(token_ids)  # [B, N, vocab_size]

        # Reshape for loss computation
        B, N, V = logits.shape
        logits_flat = logits.reshape(B * N, V)
        targets_flat = targets.reshape(B * N)

        loss = F.cross_entropy(logits_flat, targets_flat)
        return loss


def compare_with_cuda_output(pytorch_output, cuda_output_file, tolerance=1e-3):
    """Compare PyTorch output with CUDA output from file."""
    cuda_output = np.load(cuda_output_file)
    pytorch_np = pytorch_output.detach().cpu().numpy()

    abs_diff = np.abs(pytorch_np - cuda_output)
    rel_diff = abs_diff / (np.abs(cuda_output) + 1e-8)

    max_abs = np.max(abs_diff)
    max_rel = np.max(rel_diff)
    mean_abs = np.mean(abs_diff)
    mean_rel = np.mean(rel_diff)

    print(f"\n=== Comparison Results ===")
    print(f"Max absolute difference: {max_abs:.6e}")
    print(f"Max relative difference: {max_rel:.6e}")
    print(f"Mean absolute difference: {mean_abs:.6e}")
    print(f"Mean relative difference: {mean_rel:.6e}")

    if max_abs < tolerance:
        print(f"✓ Outputs match within tolerance {tolerance}")
        return True
    else:
        print(f"✗ Outputs differ beyond tolerance {tolerance}")

        # Show where differences are
        diff_indices = np.where(abs_diff > tolerance)
        print(f"\nDifferences at {len(diff_indices[0])} locations")
        print("First 10 mismatches:")
        for i in range(min(10, len(diff_indices[0]))):
            idx = tuple(d[i] for d in diff_indices)
            print(f"  Index {idx}: PyTorch={pytorch_np[idx]:.6f}, "
                  f"CUDA={cuda_output[idx]:.6f}, Diff={abs_diff[idx]:.6e}")

        return False


def test_ffn_forward():
    """Test FFN forward pass."""
    print("\n=== Test FFN Forward Pass ===")

    d_model = 64
    d_ff = 256
    batch_size = 4
    seq_len = 8

    ffn = FeedForwardNetwork(d_model, d_ff)
    ffn.eval()

    # Random input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    with torch.no_grad():
        output = ffn(x)

    # Check output shape and finiteness
    assert output.shape == (batch_size, seq_len, d_model), f"Wrong shape: {output.shape}"
    assert torch.all(torch.isfinite(output)), "Output contains non-finite values"

    # Compute some statistics
    print(f"Input: mean={x.mean():.3f}, std={x.std():.3f}")
    print(f"Output: mean={output.mean():.3f}, std={output.std():.3f}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("✓ FFN forward pass test passed")
    return True


def test_ffn_backward():
    """Test FFN backward pass."""
    print("\n=== Test FFN Backward Pass ===")

    d_model = 32
    d_ff = 128
    batch_size = 2
    seq_len = 4

    ffn = FeedForwardNetwork(d_model, d_ff)

    # Random input and target
    torch.manual_seed(123)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    target = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = ffn(x)

    # Compute MSE loss
    loss = F.mse_loss(output, target)

    # Backward pass
    loss.backward()

    # Check gradients exist and are finite
    assert x.grad is not None, "Input gradient is None"
    assert torch.all(torch.isfinite(x.grad)), "Input gradient contains non-finite values"

    for name, param in ffn.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.all(torch.isfinite(param.grad)), f"Gradient for {name} contains non-finite values"
        print(f"{name} gradient: mean={param.grad.mean():.3e}, "
              f"std={param.grad.std():.3e}, max={param.grad.abs().max():.3e}")

    print("✓ FFN backward pass test passed")
    return True


def test_transformer_forward():
    """Test full transformer forward pass."""
    print("\n=== Test Transformer Forward Pass ===")

    vocab_size = 1000
    d_model = 128
    num_layers = 4
    num_heads = 4
    d_ff = 512
    max_seq_len = 64
    batch_size = 8
    seq_len = 32

    model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)
    model.eval()

    # Random token IDs
    torch.manual_seed(42)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        logits = model(token_ids)

    # Check output
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Wrong shape: {logits.shape}"
    assert torch.all(torch.isfinite(logits)), "Logits contain non-finite values"

    # Check activations through layers
    x = model.token_embedding(token_ids)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    x = x + model.position_embedding(positions)

    print(f"After embedding: mean={x.mean():.3f}, std={x.std():.3f}, max={x.abs().max():.3f}")

    for i, block in enumerate(model.blocks):
        x = block(x)
        print(f"After block {i}: mean={x.mean():.3f}, std={x.std():.3f}, max={x.abs().max():.3f}")

    print("✓ Transformer forward pass test passed")
    return True


def test_overfit_single_batch():
    """Test if model can overfit a single batch."""
    print("\n=== Test Overfit Single Batch ===")

    vocab_size = 100
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 256
    max_seq_len = 16
    batch_size = 4
    seq_len = 8

    model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Single batch of data
    torch.manual_seed(42)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Training loop
    print("Training to overfit single batch...")
    initial_loss = None
    for step in range(100):
        optimizer.zero_grad()
        loss = model.compute_loss(token_ids, targets)
        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    final_loss = loss.item()

    # Check that loss decreased significantly
    loss_reduction = (initial_loss - final_loss) / initial_loss
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {loss_reduction*100:.1f}%")

    if loss_reduction > 0.5:  # Loss should drop by at least 50%
        print("✓ Model can overfit single batch")
        return True
    else:
        print("✗ Model failed to overfit single batch")
        return False


if __name__ == "__main__":
    print("=== PyTorch Reference Implementation Tests ===\n")

    all_passed = True

    try:
        all_passed &= test_ffn_forward()
        all_passed &= test_ffn_backward()
        all_passed &= test_transformer_forward()
        all_passed &= test_overfit_single_batch()
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n=== Final Result ===")
    print(f"All tests: {'PASSED' if all_passed else 'FAILED'}")

    sys.exit(0 if all_passed else 1)
