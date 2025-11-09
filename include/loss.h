#ifndef LOSS_H
#define LOSS_H

#include <cuda_runtime.h>

// MSE loss: L = (1/(2*B)) * sum((pred - target)^2)
float mse_loss(const float* d_pred, const float* d_target, int size, int batch_size);

// MSE gradient: grad[i] = (1/B) * (pred[i] - target[i])
void mse_gradient(const float* d_pred, const float* d_target, float* d_grad,
                  int size, int batch_size);

// Cross-entropy loss (assumes pred is already softmax): L = -(1/B) * sum(target * log(pred))
float cross_entropy_loss(const float* d_pred, const float* d_target, int size, int batch_size);

// Cross-entropy gradient (for softmax + cross-entropy): grad = (1/B) * (pred - target)
void cross_entropy_gradient(const float* d_pred, const float* d_target, float* d_grad,
                             int size, int batch_size);

// Language modeling cross-entropy loss
// d_logits: [batch_size, seq_len, vocab_size] - raw logits (NOT softmaxed)
// d_targets: [batch_size, seq_len] - integer token IDs
// mask: optional [batch_size, seq_len] - 1.0 for valid tokens, 0.0 for padding
// Returns: average loss over non-masked tokens
float lm_cross_entropy_loss(
    const float* d_logits,
    const int* d_targets,
    int batch_size,
    int seq_len,
    int vocab_size,
    const float* d_mask = nullptr
);

// Language modeling cross-entropy gradient
// d_logits: [batch_size, seq_len, vocab_size] - raw logits
// d_targets: [batch_size, seq_len] - integer token IDs
// d_grad: [batch_size, seq_len, vocab_size] - output gradients
// mask: optional [batch_size, seq_len] - 1.0 for valid tokens, 0.0 for padding
void lm_cross_entropy_gradient(
    const float* d_logits,
    const int* d_targets,
    float* d_grad,
    int batch_size,
    int seq_len,
    int vocab_size,
    const float* d_mask = nullptr
);

#endif // LOSS_H
