// This is the OLD implementation before the rewrite
// Kept for reference and testing
// This version is INEFFICIENT but CORRECT

__global__ void lm_cross_entropy_gradient_kernel_old(
    const float* logits,
    const int* targets,
    const float* mask,
    float* grad,
    int batch_size,
    int seq_len,
    int vocab_size,
    float scale
) {
    int idx = blockIdx.x;  // Position index (batch * seq_len)
    int v = threadIdx.x;   // Vocabulary index

    if (idx < batch_size * seq_len && v < vocab_size) {
        // Check if this position is masked
        float m = (mask != nullptr) ? mask[idx] : 1.0f;

        if (m > 0.0f) {
            int target = targets[idx];
            const float* logits_ptr = logits + idx * vocab_size;
            float* grad_ptr = grad + idx * vocab_size;

            // Compute softmax for this position
            // EVERY thread computes the full max independently (inefficient but simple)
            float max_logit = -INFINITY;
            for (int i = 0; i < vocab_size; i++) {
                max_logit = fmaxf(max_logit, logits_ptr[i]);
            }

            // EVERY thread computes the full sum independently
            float sum_exp = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                sum_exp += expf(logits_ptr[i] - max_logit);
            }

            // Each thread computes gradient for its vocabulary item
            float softmax_v = expf(logits_ptr[v] - max_logit) / sum_exp;

            // Gradient: softmax - one_hot(target)
            float target_indicator = (v == target) ? 1.0f : 0.0f;
            grad_ptr[v] = scale * m * (softmax_v - target_indicator);
        } else {
            grad[idx * vocab_size + v] = 0.0f;
        }
    }
}
