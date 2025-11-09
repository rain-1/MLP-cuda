#ifndef DEBUG_MONITOR_H
#define DEBUG_MONITOR_H

#include <cuda_runtime.h>
#include <string>
#include <vector>

// Enhanced statistics for debugging
struct TensorStatistics {
    float mean;
    float std;
    float min;
    float max;
    float percentile_50;  // median
    float percentile_95;
    float percentile_99;
    int num_nans;
    int num_infs;
    int num_zeros;
    int total_elements;

    // Gradient-specific stats
    float grad_norm_l2;
    float grad_norm_linf;

    void print(const char* name) const;
};

// Compute comprehensive statistics for a tensor
TensorStatistics compute_statistics(const float* d_data, int size, const char* name = nullptr);

// Monitor activations through a layer
class ActivationMonitor {
public:
    ActivationMonitor(const char* layer_name);
    ~ActivationMonitor();

    // Record activation statistics
    void record(const float* d_data, int size, int step, const char* sub_component = nullptr);

    // Print summary
    void print_summary() const;

    // Save statistics to file
    void save_to_file(const char* filename) const;

private:
    std::string layer_name_;
    std::vector<TensorStatistics> history_;
    std::vector<std::string> sub_components_;
    std::vector<int> steps_;
};

// Monitor for detecting common failure modes
class FailureDetector {
public:
    enum FailureType {
        NONE = 0,
        EXPLODING_ACTIVATIONS,
        VANISHING_ACTIVATIONS,
        EXPLODING_GRADIENTS,
        VANISHING_GRADIENTS,
        NAN_VALUES,
        INF_VALUES,
        DEAD_NEURONS,
        SATURATION
    };

    // Check if tensor shows signs of failure
    static FailureType detect_failure(const float* d_data, int size,
                                      bool is_gradient = false,
                                      float explosion_threshold = 100.0f,
                                      float vanishing_threshold = 1e-6f);

    // Get human-readable description
    static const char* failure_type_str(FailureType type);

    // Print detailed diagnosis
    static void diagnose(const float* d_data, int size, const char* name);
};

// Layer-wise monitoring for full model
class ModelMonitor {
public:
    ModelMonitor();
    ~ModelMonitor();

    // Add a tensor to monitor
    void add_activation(const char* layer_name, const float* d_data, int size);
    void add_gradient(const char* layer_name, const float* d_grad, int size);

    // Check for failures across all monitored tensors
    void check_health(int step);

    // Print summary of all monitored tensors
    void print_status() const;

    // Save full report
    void save_report(const char* filename) const;

private:
    struct MonitoredTensor {
        std::string name;
        TensorStatistics stats;
        bool is_gradient;
        FailureDetector::FailureType failure;
    };

    std::vector<MonitoredTensor> tensors_;
};

// Checkpoint for replay debugging
class DebugCheckpoint {
public:
    // Save full model state at current step
    static void save_checkpoint(
        const char* filename,
        int step,
        const std::vector<std::pair<const char*, const float*>>& tensors,
        const std::vector<int>& sizes
    );

    // Load checkpoint for replay
    static bool load_checkpoint(
        const char* filename,
        int& step,
        std::vector<float*>& d_tensors,
        std::vector<int>& sizes
    );

    // Compare two checkpoints
    static void compare_checkpoints(const char* file1, const char* file2);
};

#endif // DEBUG_MONITOR_H
