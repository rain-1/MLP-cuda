#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

namespace PrecisionAnalysis {

// ==============================================================================
// KAHAN SUMMATION - Compensated summation for improved numerical accuracy
// ==============================================================================

struct KahanAccumulator {
    float sum;
    float compensation;

    HOSTDEVICE KahanAccumulator() : sum(0.0f), compensation(0.0f) {}

    HOSTDEVICE void add(float value) {
        float y = value - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    HOSTDEVICE float get() const { return sum; }
};

struct KahanAccumulatorDouble {
    double sum;
    double compensation;

    HOSTDEVICE KahanAccumulatorDouble() : sum(0.0), compensation(0.0) {}

    HOSTDEVICE void add(double value) {
        double y = value - compensation;
        double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    HOSTDEVICE double get() const { return sum; }
};

// ==============================================================================
// PRECISION COMPARISON - Compare float32 vs float64 computations
// ==============================================================================

struct PrecisionComparison {
    float float32_result;
    double float64_result;
    float absolute_error;
    float relative_error;

    PrecisionComparison() : float32_result(0.0f), float64_result(0.0),
                           absolute_error(0.0f), relative_error(0.0f) {}

    void compute_errors() {
        absolute_error = std::abs(float32_result - (float)float64_result);
        if (std::abs(float64_result) > 1e-10) {
            relative_error = absolute_error / std::abs((float)float64_result);
        } else {
            relative_error = 0.0f;
        }
    }

    void print(const std::string& label) const {
        std::cout << std::fixed << std::setprecision(8);
        std::cout << label << ":\n";
        std::cout << "  Float32: " << float32_result << "\n";
        std::cout << "  Float64: " << float64_result << "\n";
        std::cout << "  Abs Error: " << absolute_error << "\n";
        std::cout << "  Rel Error: " << (relative_error * 100.0f) << "%\n";
    }
};

// ==============================================================================
// SOFTMAX PRECISION ANALYZER - Compare different softmax implementations
// ==============================================================================

class SoftmaxPrecisionAnalyzer {
public:
    struct SoftmaxResult {
        std::vector<float> naive_float32;
        std::vector<float> kahan_float32;
        std::vector<double> naive_float64;
        std::vector<double> kahan_float64;

        float naive_f32_sum;
        float kahan_f32_sum;
        double naive_f64_sum;
        double kahan_f64_sum;
    };

    // Compute softmax using naive float32 summation
    static std::vector<float> compute_naive_float32(const float* logits, int size) {
        // Find max for numerical stability
        float max_logit = logits[0];
        for (int i = 1; i < size; i++) {
            max_logit = std::max(max_logit, logits[i]);
        }

        // Compute exp and sum (naive summation)
        std::vector<float> exp_vals(size);
        float sum_exp = 0.0f;
        for (int i = 0; i < size; i++) {
            exp_vals[i] = std::exp(logits[i] - max_logit);
            sum_exp += exp_vals[i];  // Naive accumulation
        }

        // Normalize
        std::vector<float> softmax(size);
        for (int i = 0; i < size; i++) {
            softmax[i] = exp_vals[i] / sum_exp;
        }

        return softmax;
    }

    // Compute softmax using Kahan summation for float32
    static std::vector<float> compute_kahan_float32(const float* logits, int size) {
        // Find max
        float max_logit = logits[0];
        for (int i = 1; i < size; i++) {
            max_logit = std::max(max_logit, logits[i]);
        }

        // Compute exp with Kahan summation
        std::vector<float> exp_vals(size);
        KahanAccumulator sum_exp;
        for (int i = 0; i < size; i++) {
            exp_vals[i] = std::exp(logits[i] - max_logit);
            sum_exp.add(exp_vals[i]);
        }

        // Normalize
        std::vector<float> softmax(size);
        float sum = sum_exp.get();
        for (int i = 0; i < size; i++) {
            softmax[i] = exp_vals[i] / sum;
        }

        return softmax;
    }

    // Compute softmax using naive float64 summation
    static std::vector<double> compute_naive_float64(const float* logits, int size) {
        // Find max
        double max_logit = logits[0];
        for (int i = 1; i < size; i++) {
            max_logit = std::max(max_logit, (double)logits[i]);
        }

        // Compute exp and sum
        std::vector<double> exp_vals(size);
        double sum_exp = 0.0;
        for (int i = 0; i < size; i++) {
            exp_vals[i] = std::exp((double)logits[i] - max_logit);
            sum_exp += exp_vals[i];
        }

        // Normalize
        std::vector<double> softmax(size);
        for (int i = 0; i < size; i++) {
            softmax[i] = exp_vals[i] / sum_exp;
        }

        return softmax;
    }

    // Compute softmax using Kahan summation for float64
    static std::vector<double> compute_kahan_float64(const float* logits, int size) {
        // Find max
        double max_logit = logits[0];
        for (int i = 1; i < size; i++) {
            max_logit = std::max(max_logit, (double)logits[i]);
        }

        // Compute exp with Kahan summation
        std::vector<double> exp_vals(size);
        KahanAccumulatorDouble sum_exp;
        for (int i = 0; i < size; i++) {
            exp_vals[i] = std::exp((double)logits[i] - max_logit);
            sum_exp.add(exp_vals[i]);
        }

        // Normalize
        std::vector<double> softmax(size);
        double sum = sum_exp.get();
        for (int i = 0; i < size; i++) {
            softmax[i] = exp_vals[i] / sum;
        }

        return softmax;
    }

    // Compare all implementations
    static SoftmaxResult analyze(const float* logits, int size) {
        SoftmaxResult result;

        result.naive_float32 = compute_naive_float32(logits, size);
        result.kahan_float32 = compute_kahan_float32(logits, size);
        result.naive_float64 = compute_naive_float64(logits, size);
        result.kahan_float64 = compute_kahan_float64(logits, size);

        // Compute sums (should all be 1.0, but may differ due to precision)
        result.naive_f32_sum = 0.0f;
        result.kahan_f32_sum = 0.0f;
        result.naive_f64_sum = 0.0;
        result.kahan_f64_sum = 0.0;

        for (int i = 0; i < size; i++) {
            result.naive_f32_sum += result.naive_float32[i];
            result.kahan_f32_sum += result.kahan_float32[i];
            result.naive_f64_sum += result.naive_float64[i];
            result.kahan_f64_sum += result.kahan_float64[i];
        }

        return result;
    }

    static void print_comparison(const SoftmaxResult& result) {
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "\n=== Softmax Precision Comparison ===\n";
        std::cout << "Sum of probabilities (should be 1.0):\n";
        std::cout << "  Naive Float32:  " << result.naive_f32_sum
                  << " (error: " << std::abs(result.naive_f32_sum - 1.0f) << ")\n";
        std::cout << "  Kahan Float32:  " << result.kahan_f32_sum
                  << " (error: " << std::abs(result.kahan_f32_sum - 1.0f) << ")\n";
        std::cout << "  Naive Float64:  " << result.naive_f64_sum
                  << " (error: " << std::abs(result.naive_f64_sum - 1.0) << ")\n";
        std::cout << "  Kahan Float64:  " << result.kahan_f64_sum
                  << " (error: " << std::abs(result.kahan_f64_sum - 1.0) << ")\n";

        // Compute max difference from float64 reference
        if (result.naive_float32.size() > 0) {
            float max_diff_naive = 0.0f;
            float max_diff_kahan = 0.0f;

            for (size_t i = 0; i < result.naive_float32.size(); i++) {
                float diff_naive = std::abs(result.naive_float32[i] - (float)result.naive_float64[i]);
                float diff_kahan = std::abs(result.kahan_float32[i] - (float)result.kahan_float64[i]);
                max_diff_naive = std::max(max_diff_naive, diff_naive);
                max_diff_kahan = std::max(max_diff_kahan, diff_kahan);
            }

            std::cout << "\nMax difference from Float64 reference:\n";
            std::cout << "  Naive Float32: " << max_diff_naive << "\n";
            std::cout << "  Kahan Float32: " << max_diff_kahan << "\n";

            float improvement = ((max_diff_naive - max_diff_kahan) / max_diff_naive) * 100.0f;
            if (improvement > 0) {
                std::cout << "  Kahan improvement: " << improvement << "%\n";
            }
        }
    }
};

// ==============================================================================
// GRADIENT PRECISION ANALYZER - Analyze gradient computation precision
// ==============================================================================

class GradientPrecisionAnalyzer {
public:
    struct GradientStats {
        float min_gradient;
        float max_gradient;
        float mean_gradient;
        float std_gradient;
        float abs_mean;
        int num_near_zero;  // Count of gradients < 1e-6
        int num_gradients;

        void compute(const float* gradients, int size) {
            num_gradients = size;
            min_gradient = gradients[0];
            max_gradient = gradients[0];

            KahanAccumulator sum;
            KahanAccumulator abs_sum;
            num_near_zero = 0;

            for (int i = 0; i < size; i++) {
                float grad = gradients[i];
                min_gradient = std::min(min_gradient, grad);
                max_gradient = std::max(max_gradient, grad);
                sum.add(grad);
                abs_sum.add(std::abs(grad));

                if (std::abs(grad) < 1e-6f) {
                    num_near_zero++;
                }
            }

            mean_gradient = sum.get() / size;
            abs_mean = abs_sum.get() / size;

            // Compute standard deviation
            KahanAccumulator var_sum;
            for (int i = 0; i < size; i++) {
                float diff = gradients[i] - mean_gradient;
                var_sum.add(diff * diff);
            }
            std_gradient = std::sqrt(var_sum.get() / size);
        }

        void print(const std::string& label) const {
            std::cout << std::scientific << std::setprecision(6);
            std::cout << label << " Gradient Statistics:\n";
            std::cout << "  Range: [" << min_gradient << ", " << max_gradient << "]\n";
            std::cout << "  Mean: " << mean_gradient << "\n";
            std::cout << "  Abs Mean: " << abs_mean << "\n";
            std::cout << "  Std Dev: " << std_gradient << "\n";
            std::cout << "  Near-zero (<1e-6): " << num_near_zero << "/" << num_gradients
                      << " (" << (100.0f * num_near_zero / num_gradients) << "%)\n";
        }
    };

    // Compare gradient computation in float32 vs float64
    static void compare_cross_entropy_gradient(
        const float* logits, int vocab_size, int target,
        float* grad_f32, double* grad_f64) {

        // Float32 version
        auto softmax_f32 = SoftmaxPrecisionAnalyzer::compute_kahan_float32(logits, vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            grad_f32[i] = softmax_f32[i];
        }
        grad_f32[target] -= 1.0f;

        // Float64 version
        auto softmax_f64 = SoftmaxPrecisionAnalyzer::compute_kahan_float64(logits, vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            grad_f64[i] = softmax_f64[i];
        }
        grad_f64[target] -= 1.0;
    }
};

// ==============================================================================
// NUMERICAL STABILITY METRICS
// ==============================================================================

struct NumericalStabilityMetrics {
    float condition_number;
    float dynamic_range;
    float cancellation_factor;

    // Estimate condition number of a vector
    static float estimate_condition_number(const float* values, int size) {
        float max_val = 0.0f;
        float min_val = std::abs(values[0]);

        for (int i = 0; i < size; i++) {
            float abs_val = std::abs(values[i]);
            if (abs_val > 1e-10f) {  // Ignore zeros
                max_val = std::max(max_val, abs_val);
                min_val = std::min(min_val, abs_val);
            }
        }

        return (min_val > 1e-10f) ? (max_val / min_val) : INFINITY;
    }

    // Compute dynamic range (log10 of max/min ratio)
    static float compute_dynamic_range(const float* values, int size) {
        float cond = estimate_condition_number(values, size);
        return (cond < INFINITY) ? std::log10(cond) : INFINITY;
    }

    void print() const {
        std::cout << "\nNumerical Stability Metrics:\n";
        std::cout << "  Condition Number: " << condition_number << "\n";
        std::cout << "  Dynamic Range: " << dynamic_range << " orders of magnitude\n";
        std::cout << "  Cancellation Factor: " << cancellation_factor << "\n";
    }
};

} // namespace PrecisionAnalysis
