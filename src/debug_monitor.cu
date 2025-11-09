#include "debug_monitor.h"
#include "gradient_utils.h"
#include "matrix_ops.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <fstream>

void TensorStatistics::print(const char* name) const {
    printf("=== %s Statistics ===\n", name);
    printf("  Mean: %.6e, Std: %.6e\n", mean, std);
    printf("  Range: [%.6e, %.6e]\n", min, max);
    printf("  Percentiles: 50%%=%.6e, 95%%=%.6e, 99%%=%.6e\n",
           percentile_50, percentile_95, percentile_99);
    printf("  NaNs: %d, Infs: %d, Zeros: %d (out of %d)\n",
           num_nans, num_infs, num_zeros, total_elements);
    if (grad_norm_l2 > 0 || grad_norm_linf > 0) {
        printf("  Gradient norms: L2=%.6e, Linf=%.6e\n", grad_norm_l2, grad_norm_linf);
    }
}

__global__ void compute_stats_kernel(
    const float* data,
    int size,
    float* mean_out,
    int* nan_count,
    int* inf_count,
    int* zero_count,
    float* min_out,
    float* max_out
) {
    __shared__ float sdata[256];
    __shared__ int snan[256];
    __shared__ int sinf[256];
    __shared__ int szero[256];
    __shared__ float smin[256];
    __shared__ float smax[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    float val = (idx < size) ? data[idx] : 0.0f;
    sdata[tid] = val;
    snan[tid] = (idx < size && isnan(val)) ? 1 : 0;
    sinf[tid] = (idx < size && isinf(val)) ? 1 : 0;
    szero[tid] = (idx < size && val == 0.0f) ? 1 : 0;
    smin[tid] = (idx < size) ? val : INFINITY;
    smax[tid] = (idx < size) ? val : -INFINITY;

    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < size) {
            sdata[tid] += sdata[tid + s];
            snan[tid] += snan[tid + s];
            sinf[tid] += sinf[tid + s];
            szero[tid] += szero[tid + s];
            smin[tid] = fminf(smin[tid], smin[tid + s]);
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    // Write results
    if (tid == 0) {
        atomicAdd(mean_out, sdata[0]);
        atomicAdd(nan_count, snan[0]);
        atomicAdd(inf_count, sinf[0]);
        atomicAdd(zero_count, szero[0]);
        atomicMin((int*)min_out, __float_as_int(smin[0]));
        atomicMax((int*)max_out, __float_as_int(smax[0]));
    }
}

TensorStatistics compute_statistics(const float* d_data, int size, const char* name) {
    TensorStatistics stats = {0};
    stats.total_elements = size;

    if (size == 0) return stats;

    // Copy data to host for detailed analysis
    float* h_data = new float[size];
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute statistics on CPU (more accurate)
    double sum = 0.0;
    double sum_sq = 0.0;
    stats.min = h_data[0];
    stats.max = h_data[0];

    for (int i = 0; i < size; i++) {
        float val = h_data[i];

        if (isnan(val)) {
            stats.num_nans++;
        } else if (isinf(val)) {
            stats.num_infs++;
        } else {
            if (val == 0.0f) stats.num_zeros++;
            sum += val;
            sum_sq += val * val;
            stats.min = fminf(stats.min, val);
            stats.max = fmaxf(stats.max, val);
        }
    }

    int valid_count = size - stats.num_nans - stats.num_infs;
    if (valid_count > 0) {
        stats.mean = sum / valid_count;
        float variance = (sum_sq / valid_count) - (stats.mean * stats.mean);
        stats.std = sqrtf(fmaxf(variance, 0.0f));
    }

    // Compute percentiles (need to sort)
    std::vector<float> sorted_data;
    sorted_data.reserve(valid_count);
    for (int i = 0; i < size; i++) {
        if (isfinite(h_data[i])) {
            sorted_data.push_back(h_data[i]);
        }
    }

    if (!sorted_data.empty()) {
        std::sort(sorted_data.begin(), sorted_data.end());
        stats.percentile_50 = sorted_data[sorted_data.size() / 2];
        stats.percentile_95 = sorted_data[(int)(sorted_data.size() * 0.95)];
        stats.percentile_99 = sorted_data[(int)(sorted_data.size() * 0.99)];
    }

    // Compute gradient norms
    stats.grad_norm_l2 = gradient_norm(d_data, size);
    stats.grad_norm_linf = stats.max;

    delete[] h_data;

    if (name) {
        stats.print(name);
    }

    return stats;
}

// ActivationMonitor implementation
ActivationMonitor::ActivationMonitor(const char* layer_name)
    : layer_name_(layer_name) {}

ActivationMonitor::~ActivationMonitor() {}

void ActivationMonitor::record(const float* d_data, int size, int step, const char* sub_component) {
    TensorStatistics stats = compute_statistics(d_data, size, nullptr);
    history_.push_back(stats);
    steps_.push_back(step);
    sub_components_.push_back(sub_component ? sub_component : "");
}

void ActivationMonitor::print_summary() const {
    printf("\n=== Activation Monitor: %s ===\n", layer_name_.c_str());
    printf("Recorded %zu snapshots\n", history_.size());

    if (history_.empty()) return;

    // Print statistics over time
    printf("\n%-10s %-15s %-15s %-15s %-15s %-15s\n",
           "Step", "Component", "Mean", "Std", "Min", "Max");
    printf("--------------------------------------------------------------------------------\n");

    for (size_t i = 0; i < history_.size(); i++) {
        const auto& stats = history_[i];
        printf("%-10d %-15s %-15.6e %-15.6e %-15.6e %-15.6e\n",
               steps_[i],
               sub_components_[i].c_str(),
               stats.mean, stats.std, stats.min, stats.max);
    }
}

void ActivationMonitor::save_to_file(const char* filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    file << "layer_name,step,component,mean,std,min,max,p50,p95,p99,nans,infs,zeros\n";

    for (size_t i = 0; i < history_.size(); i++) {
        const auto& stats = history_[i];
        file << layer_name_ << ","
             << steps_[i] << ","
             << sub_components_[i] << ","
             << stats.mean << ","
             << stats.std << ","
             << stats.min << ","
             << stats.max << ","
             << stats.percentile_50 << ","
             << stats.percentile_95 << ","
             << stats.percentile_99 << ","
             << stats.num_nans << ","
             << stats.num_infs << ","
             << stats.num_zeros << "\n";
    }

    file.close();
    printf("Saved monitoring data to %s\n", filename);
}

// FailureDetector implementation
FailureDetector::FailureType FailureDetector::detect_failure(
    const float* d_data,
    int size,
    bool is_gradient,
    float explosion_threshold,
    float vanishing_threshold
) {
    TensorStatistics stats = compute_statistics(d_data, size, nullptr);

    // Check for NaN/Inf
    if (stats.num_nans > 0) return NAN_VALUES;
    if (stats.num_infs > 0) return INF_VALUES;

    // Check for dead neurons (all zeros)
    if (stats.num_zeros == size) return DEAD_NEURONS;

    // Check for explosion/vanishing
    if (is_gradient) {
        if (stats.grad_norm_l2 > explosion_threshold) return EXPLODING_GRADIENTS;
        if (stats.grad_norm_l2 < vanishing_threshold) return VANISHING_GRADIENTS;
    } else {
        if (fabsf(stats.max) > explosion_threshold || fabsf(stats.min) > explosion_threshold) {
            return EXPLODING_ACTIVATIONS;
        }
        if (fabsf(stats.max) < vanishing_threshold && fabsf(stats.min) < vanishing_threshold) {
            return VANISHING_ACTIVATIONS;
        }
    }

    return NONE;
}

const char* FailureDetector::failure_type_str(FailureType type) {
    switch (type) {
        case NONE: return "None";
        case EXPLODING_ACTIVATIONS: return "Exploding Activations";
        case VANISHING_ACTIVATIONS: return "Vanishing Activations";
        case EXPLODING_GRADIENTS: return "Exploding Gradients";
        case VANISHING_GRADIENTS: return "Vanishing Gradients";
        case NAN_VALUES: return "NaN Values";
        case INF_VALUES: return "Inf Values";
        case DEAD_NEURONS: return "Dead Neurons";
        case SATURATION: return "Saturation";
        default: return "Unknown";
    }
}

void FailureDetector::diagnose(const float* d_data, int size, const char* name) {
    printf("\n=== Diagnosis for %s ===\n", name);

    TensorStatistics stats = compute_statistics(d_data, size, name);

    FailureType failure = detect_failure(d_data, size, false);
    printf("\nDiagnosis: %s\n", failure_type_str(failure));

    if (failure != NONE) {
        printf("\nRecommendations:\n");
        switch (failure) {
            case EXPLODING_ACTIVATIONS:
                printf("  - Reduce learning rate\n");
                printf("  - Add gradient clipping\n");
                printf("  - Check initialization scale\n");
                printf("  - Add normalization layers\n");
                break;
            case VANISHING_ACTIVATIONS:
                printf("  - Check for dead activations (ReLU)\n");
                printf("  - Increase initialization scale\n");
                printf("  - Use different activation function\n");
                break;
            case NAN_VALUES:
                printf("  - Check for division by zero\n");
                printf("  - Check for log of negative numbers\n");
                printf("  - Reduce learning rate\n");
                break;
            case INF_VALUES:
                printf("  - Check for overflow in exponentials\n");
                printf("  - Add numerical stability epsilon\n");
                printf("  - Clip intermediate values\n");
                break;
            case DEAD_NEURONS:
                printf("  - All values are zero - layer may be dead\n");
                printf("  - Check initialization\n");
                printf("  - Check for gradient flow\n");
                break;
            default:
                break;
        }
    }
}

// ModelMonitor implementation
ModelMonitor::ModelMonitor() {}

ModelMonitor::~ModelMonitor() {}

void ModelMonitor::add_activation(const char* layer_name, const float* d_data, int size) {
    MonitoredTensor tensor;
    tensor.name = std::string(layer_name) + " (activation)";
    tensor.stats = compute_statistics(d_data, size, nullptr);
    tensor.is_gradient = false;
    tensor.failure = FailureDetector::detect_failure(d_data, size, false);
    tensors_.push_back(tensor);
}

void ModelMonitor::add_gradient(const char* layer_name, const float* d_grad, int size) {
    MonitoredTensor tensor;
    tensor.name = std::string(layer_name) + " (gradient)";
    tensor.stats = compute_statistics(d_grad, size, nullptr);
    tensor.is_gradient = true;
    tensor.failure = FailureDetector::detect_failure(d_grad, size, true);
    tensors_.push_back(tensor);
}

void ModelMonitor::check_health(int step) {
    printf("\n=== Model Health Check (Step %d) ===\n", step);

    int failures = 0;
    for (const auto& tensor : tensors_) {
        if (tensor.failure != FailureDetector::NONE) {
            printf("⚠ %s: %s\n", tensor.name.c_str(),
                   FailureDetector::failure_type_str(tensor.failure));
            failures++;
        }
    }

    if (failures == 0) {
        printf("✓ All monitored tensors healthy\n");
    } else {
        printf("\n✗ Found %d failures\n", failures);
    }

    tensors_.clear();  // Reset for next check
}

void ModelMonitor::print_status() const {
    printf("\n=== Model Monitor Status ===\n");
    printf("Monitoring %zu tensors\n\n", tensors_.size());

    printf("%-40s %-15s %-15s %-15s %-15s\n",
           "Tensor", "Mean", "Std", "Min", "Max");
    printf("--------------------------------------------------------------------------------\n");

    for (const auto& tensor : tensors_) {
        const char* status = (tensor.failure == FailureDetector::NONE) ? "" : " [FAIL]";
        printf("%-40s %-15.6e %-15.6e %-15.6e %-15.6e%s\n",
               tensor.name.c_str(),
               tensor.stats.mean,
               tensor.stats.std,
               tensor.stats.min,
               tensor.stats.max,
               status);
    }
}

void ModelMonitor::save_report(const char* filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    file << "name,type,mean,std,min,max,failure\n";

    for (const auto& tensor : tensors_) {
        file << tensor.name << ","
             << (tensor.is_gradient ? "gradient" : "activation") << ","
             << tensor.stats.mean << ","
             << tensor.stats.std << ","
             << tensor.stats.min << ","
             << tensor.stats.max << ","
             << FailureDetector::failure_type_str(tensor.failure) << "\n";
    }

    file.close();
    printf("Saved monitor report to %s\n", filename);
}

// DebugCheckpoint implementation
void DebugCheckpoint::save_checkpoint(
    const char* filename,
    int step,
    const std::vector<std::pair<const char*, const float*>>& tensors,
    const std::vector<int>& sizes
) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open checkpoint file: %s\n", filename);
        return;
    }

    // Write header
    fwrite(&step, sizeof(int), 1, file);
    int num_tensors = tensors.size();
    fwrite(&num_tensors, sizeof(int), 1, file);

    // Write each tensor
    for (size_t i = 0; i < tensors.size(); i++) {
        const char* name = tensors[i].first;
        const float* d_data = tensors[i].second;
        int size = sizes[i];

        // Write name
        int name_len = strlen(name);
        fwrite(&name_len, sizeof(int), 1, file);
        fwrite(name, sizeof(char), name_len, file);

        // Write size
        fwrite(&size, sizeof(int), 1, file);

        // Write data
        float* h_data = new float[size];
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
        fwrite(h_data, sizeof(float), size, file);
        delete[] h_data;
    }

    fclose(file);
    printf("Saved checkpoint at step %d to %s\n", step, filename);
}
