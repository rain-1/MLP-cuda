#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include <cuda_runtime.h>

// Diagnostic statistics structure
struct TensorStats {
    float l2_norm;      // L2 norm (sqrt of sum of squares)
    float max_abs;      // Maximum absolute value
    float mean;         // Mean value
    float min_val;      // Minimum value
    float max_val;      // Maximum value
};

// Compute L2 norm of a device tensor
float compute_l2_norm(const float* d_tensor, int size);

// Compute max absolute value of a device tensor
float compute_max_abs(const float* d_tensor, int size);

// Compute comprehensive statistics for a tensor
TensorStats compute_tensor_stats(const float* d_tensor, int size);

// Check if tensor contains NaN or Inf
bool has_nan_or_inf(const float* d_tensor, int size);

// Diagnostic logger for training
class DiagnosticLogger {
public:
    DiagnosticLogger();
    ~DiagnosticLogger();

    // Enable/disable diagnostic logging
    void set_enabled(bool enabled) { enabled_ = enabled; }

    // Set warning thresholds
    void set_grad_norm_threshold(float threshold) { grad_norm_threshold_ = threshold; }
    void set_param_norm_threshold(float threshold) { param_norm_threshold_ = threshold; }
    void set_loss_increase_threshold(float threshold) { loss_increase_threshold_ = threshold; }

    // Log training step diagnostics
    void log_step(
        int step,
        float loss,
        float learning_rate,
        float grad_norm,
        float param_norm
    );

    // Log detailed gradient statistics
    void log_gradient_stats(
        int step,
        const char* name,
        const float* d_grad,
        int size
    );

    // Log detailed parameter statistics
    void log_parameter_stats(
        int step,
        const char* name,
        const float* d_param,
        int size
    );

    // Check for divergence
    bool check_divergence(int step, float loss);

    // Print summary
    void print_summary();

private:
    bool enabled_;
    float grad_norm_threshold_;
    float param_norm_threshold_;
    float loss_increase_threshold_;

    // Track loss history for divergence detection
    static const int HISTORY_SIZE = 10;
    float loss_history_[HISTORY_SIZE];
    int history_idx_;
    float min_loss_;
    int steps_since_improvement_;
};

#endif // DIAGNOSTICS_H
