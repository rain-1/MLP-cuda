#include "training_diagnostics.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace TrainingDiagnostics {

// ==============================================================================
// GRADIENT STATISTICS IMPLEMENTATION
// ==============================================================================

void GradientStats::compute(const float* gradients, int size) {
    if (size == 0) return;

    total_elements = size;
    min_value = gradients[0];
    max_value = gradients[0];

    // Reset counters
    num_zeros = 0;
    num_near_zero = 0;
    num_small = 0;
    num_large = 0;
    num_very_large = 0;

    // Use Kahan summation for accuracy
    PrecisionAnalysis::KahanAccumulator sum;
    PrecisionAnalysis::KahanAccumulator abs_sum;
    PrecisionAnalysis::KahanAccumulator sq_sum;

    for (int i = 0; i < size; i++) {
        float grad = gradients[i];
        float abs_grad = std::abs(grad);

        min_value = std::min(min_value, grad);
        max_value = std::max(max_value, grad);

        sum.add(grad);
        abs_sum.add(abs_grad);
        sq_sum.add(grad * grad);

        // Count distribution
        if (grad == 0.0f) num_zeros++;
        if (abs_grad < 1e-6f) num_near_zero++;
        if (abs_grad < 1e-3f) num_small++;
        if (abs_grad > 1.0f) num_large++;
        if (abs_grad > 10.0f) num_very_large++;
    }

    mean = sum.get() / size;
    abs_mean = abs_sum.get() / size;
    l2_norm = std::sqrt(sq_sum.get());

    // Compute standard deviation
    PrecisionAnalysis::KahanAccumulator var_sum;
    for (int i = 0; i < size; i++) {
        float diff = gradients[i] - mean;
        var_sum.add(diff * diff);
    }
    std_dev = std::sqrt(var_sum.get() / size);

    // Compute percentiles
    std::vector<float> sorted_grads(gradients, gradients + size);
    std::sort(sorted_grads.begin(), sorted_grads.end());

    p10 = sorted_grads[size * 10 / 100];
    p50 = sorted_grads[size * 50 / 100];
    p90 = sorted_grads[size * 90 / 100];
    p99 = sorted_grads[size * 99 / 100];
}

void GradientStats::print(const std::string& name) const {
    std::cout << "\n" << name << " Statistics:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  Range:     [" << min_value << ", " << max_value << "]\n";
    std::cout << "  Mean:      " << mean << "\n";
    std::cout << "  Abs Mean:  " << abs_mean << "\n";
    std::cout << "  Std Dev:   " << std_dev << "\n";
    std::cout << "  L2 Norm:   " << l2_norm << "\n";
    std::cout << "\nPercentiles:\n";
    std::cout << "  10th: " << p10 << "\n";
    std::cout << "  50th: " << p50 << "\n";
    std::cout << "  90th: " << p90 << "\n";
    std::cout << "  99th: " << p99 << "\n";
    std::cout << "\nDistribution:\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Zeros:          " << num_zeros << " ("
              << (100.0f * num_zeros / total_elements) << "%)\n";
    std::cout << "  Near-zero:      " << num_near_zero << " ("
              << (100.0f * num_near_zero / total_elements) << "%)\n";
    std::cout << "  Small (< 1e-3): " << num_small << " ("
              << (100.0f * num_small / total_elements) << "%)\n";
    std::cout << "  Large (> 1):    " << num_large << " ("
              << (100.0f * num_large / total_elements) << "%)\n";
    std::cout << "  Very large (> 10): " << num_very_large << " ("
              << (100.0f * num_very_large / total_elements) << "%)\n";
}

std::string GradientStats::to_csv_row() const {
    std::stringstream ss;
    ss << std::scientific << std::setprecision(6);
    ss << min_value << "," << max_value << "," << mean << ","
       << std_dev << "," << abs_mean << "," << l2_norm << ","
       << num_zeros << "," << num_near_zero << "," << num_small << ","
       << num_large << "," << num_very_large;
    return ss.str();
}

// ==============================================================================
// ADAM STATE STATS IMPLEMENTATION
// ==============================================================================

void AdamStateStats::compute(const float* m, const float* v, const float* updates,
                             const float* params, int size) {
    num_denormal_m = 0;
    num_denormal_v = 0;
    num_inf_or_nan = 0;
    min_v = (size > 0) ? v[0] : 0.0f;
    max_update_ratio = 0.0f;

    float max_abs_param = 0.0f;
    float max_abs_update = 0.0f;

    for (int i = 0; i < size; i++) {
        // Check for denormal values (very small numbers near zero)
        if (m[i] != 0.0f && std::abs(m[i]) < 1e-30f) num_denormal_m++;
        if (v[i] != 0.0f && std::abs(v[i]) < 1e-30f) num_denormal_v++;

        // Check for inf or nan in updates
        if (!std::isfinite(updates[i])) num_inf_or_nan++;

        // Track minimum second moment
        if (v[i] > 0.0f) {
            min_v = std::min(min_v, v[i]);
        }

        // Track max values for ratio
        max_abs_param = std::max(max_abs_param, std::abs(params[i]));
        max_abs_update = std::max(max_abs_update, std::abs(updates[i]));
    }

    if (max_abs_param > 1e-10f) {
        max_update_ratio = max_abs_update / max_abs_param;
    }

    // Compute stats for each component
    m_stats.compute(m, size);
    v_stats.compute(v, size);
    update_stats.compute(updates, size);
}

void AdamStateStats::print() const {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              ADAM OPTIMIZER STATE ANALYSIS                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    std::cout << "\nFirst Moment (m) - Momentum:\n";
    std::cout << "  Range: [" << std::scientific << std::setprecision(4)
              << m_stats.min_value << ", " << m_stats.max_value << "]\n";
    std::cout << "  L2 Norm: " << m_stats.l2_norm << "\n";
    std::cout << "  Denormals: " << num_denormal_m << "\n";

    std::cout << "\nSecond Moment (v) - Variance:\n";
    std::cout << "  Range: [" << v_stats.min_value << ", " << v_stats.max_value << "]\n";
    std::cout << "  Min v: " << min_v << " (should be > 0)\n";
    std::cout << "  L2 Norm: " << v_stats.l2_norm << "\n";
    std::cout << "  Denormals: " << num_denormal_v << "\n";

    std::cout << "\nParameter Updates:\n";
    std::cout << "  Range: [" << update_stats.min_value << ", "
              << update_stats.max_value << "]\n";
    std::cout << "  L2 Norm: " << update_stats.l2_norm << "\n";
    std::cout << "  Max update/param ratio: " << std::fixed << std::setprecision(6)
              << max_update_ratio << "\n";
    std::cout << "  Inf/NaN count: " << num_inf_or_nan << "\n";

    // Health check
    bool healthy = true;
    if (num_denormal_m > 0 || num_denormal_v > 0) {
        std::cout << "\n⚠️  WARNING: Denormal values detected (may indicate underflow)\n";
        healthy = false;
    }
    if (num_inf_or_nan > 0) {
        std::cout << "\n❌ ERROR: Inf/NaN detected in updates!\n";
        healthy = false;
    }
    if (min_v < 1e-20f) {
        std::cout << "\n⚠️  WARNING: Very small v values (may cause numerical instability)\n";
        healthy = false;
    }
    if (healthy) {
        std::cout << "\n✓ Optimizer state appears healthy\n";
    }
}

// ==============================================================================
// ADAM MONITOR IMPLEMENTATION
// ==============================================================================

void AdamMonitor::print_trends(int last_n) const {
    int n = std::min(last_n, (int)history_.size());
    if (n == 0) return;

    std::cout << "\nAdam State Trends (last " << n << " steps):\n";
    std::cout << std::string(60, '-') << "\n";

    // Compute average statistics
    float avg_m_norm = 0.0f;
    float avg_v_norm = 0.0f;
    float avg_update_norm = 0.0f;
    int total_denormal = 0;
    int total_inf_nan = 0;

    for (int i = history_.size() - n; i < (int)history_.size(); i++) {
        avg_m_norm += history_[i].m_stats.l2_norm;
        avg_v_norm += history_[i].v_stats.l2_norm;
        avg_update_norm += history_[i].update_stats.l2_norm;
        total_denormal += history_[i].num_denormal_m + history_[i].num_denormal_v;
        total_inf_nan += history_[i].num_inf_or_nan;
    }

    avg_m_norm /= n;
    avg_v_norm /= n;
    avg_update_norm /= n;

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  Avg m norm:      " << avg_m_norm << "\n";
    std::cout << "  Avg v norm:      " << avg_v_norm << "\n";
    std::cout << "  Avg update norm: " << avg_update_norm << "\n";
    std::cout << "  Total denormals: " << total_denormal << "\n";
    std::cout << "  Total inf/nan:   " << total_inf_nan << "\n";
}

bool AdamMonitor::detect_optimizer_issues() const {
    if (history_.empty()) return false;

    const auto& latest = history_.back();

    bool has_issues = false;

    if (latest.num_inf_or_nan > 0) {
        std::cout << "❌ CRITICAL: Inf/NaN in optimizer updates!\n";
        has_issues = true;
    }

    if (latest.num_denormal_m + latest.num_denormal_v > 0) {
        std::cout << "⚠️  WARNING: Denormal values in optimizer state\n";
        has_issues = true;
    }

    if (latest.min_v < 1e-20f) {
        std::cout << "⚠️  WARNING: Extremely small v values\n";
        has_issues = true;
    }

    // Check if update norms are decreasing over time (might indicate vanishing gradients)
    if (history_.size() >= 10) {
        float recent_avg = 0.0f;
        float old_avg = 0.0f;
        int n = std::min(5, (int)history_.size() / 2);

        for (int i = 0; i < n; i++) {
            old_avg += history_[i].update_stats.l2_norm;
            recent_avg += history_[history_.size() - 1 - i].update_stats.l2_norm;
        }
        old_avg /= n;
        recent_avg /= n;

        if (recent_avg < old_avg * 0.1f) {
            std::cout << "⚠️  WARNING: Update magnitudes decreasing rapidly\n";
            has_issues = true;
        }
    }

    return has_issues;
}

} // namespace TrainingDiagnostics
