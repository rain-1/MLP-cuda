#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "precision_analysis.h"

namespace TrainingDiagnostics {

// ==============================================================================
// GRADIENT STATISTICS - Real-time tracking of gradient properties
// ==============================================================================

struct GradientStats {
    // Basic statistics
    float min_value;
    float max_value;
    float mean;
    float std_dev;
    float abs_mean;

    // L2 norm (important for gradient clipping)
    float l2_norm;

    // Distribution analysis
    int num_zeros;
    int num_near_zero;     // |grad| < 1e-6
    int num_small;         // |grad| < 1e-3
    int num_large;         // |grad| > 1.0
    int num_very_large;    // |grad| > 10.0
    int total_elements;

    // Percentiles
    float p10, p50, p90, p99;

    GradientStats() : min_value(0), max_value(0), mean(0), std_dev(0), abs_mean(0),
                      l2_norm(0), num_zeros(0), num_near_zero(0), num_small(0),
                      num_large(0), num_very_large(0), total_elements(0),
                      p10(0), p50(0), p90(0), p99(0) {}

    void compute(const float* gradients, int size);
    void print(const std::string& name) const;
    std::string to_csv_row() const;
};

// ==============================================================================
// LAYER GRADIENT TRACKER - Track gradients for each layer
// ==============================================================================

class LayerGradientTracker {
public:
    struct LayerInfo {
        std::string name;
        int num_parameters;
        GradientStats stats;

        // Gradient flow metrics
        float gradient_scale;  // Relative to output layer (1.0 = output)
        bool is_vanishing;     // gradient_scale < 0.01
        bool is_exploding;     // gradient_scale > 100.0
    };

private:
    std::vector<LayerInfo> layers_;
    int step_;

public:
    LayerGradientTracker() : step_(0) {}

    void add_layer(const std::string& name, int num_params) {
        LayerInfo info;
        info.name = name;
        info.num_parameters = num_params;
        info.gradient_scale = 1.0f;
        info.is_vanishing = false;
        info.is_exploding = false;
        layers_.push_back(info);
    }

    void update_layer(int layer_idx, const float* gradients) {
        if (layer_idx >= 0 && layer_idx < (int)layers_.size()) {
            layers_[layer_idx].stats.compute(gradients, layers_[layer_idx].num_parameters);

            // Update gradient flow metrics (relative to first layer as reference)
            if (layer_idx > 0 && layers_[0].stats.l2_norm > 1e-10f) {
                layers_[layer_idx].gradient_scale =
                    layers_[layer_idx].stats.l2_norm / layers_[0].stats.l2_norm;
                layers_[layer_idx].is_vanishing = (layers_[layer_idx].gradient_scale < 0.01f);
                layers_[layer_idx].is_exploding = (layers_[layer_idx].gradient_scale > 100.0f);
            }
        }
    }

    void print_summary() const {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          LAYER-BY-LAYER GRADIENT FLOW ANALYSIS                ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

        for (size_t i = 0; i < layers_.size(); i++) {
            const auto& layer = layers_[i];
            std::cout << "Layer " << i << ": " << layer.name
                      << " (" << layer.num_parameters << " params)\n";
            std::cout << "  L2 Norm: " << std::scientific << std::setprecision(4)
                      << layer.stats.l2_norm;
            std::cout << "  Scale: " << std::fixed << std::setprecision(3)
                      << layer.gradient_scale << "x";

            if (layer.is_vanishing) {
                std::cout << "  [VANISHING ⚠️]";
            } else if (layer.is_exploding) {
                std::cout << "  [EXPLODING ⚠️]";
            }
            std::cout << "\n";

            std::cout << "  Range: [" << std::scientific << std::setprecision(3)
                      << layer.stats.min_value << ", " << layer.stats.max_value << "]\n";
            std::cout << "  Mean: " << layer.stats.mean
                      << ", Std: " << layer.stats.std_dev << "\n";
            std::cout << "\n";
        }
    }

    void print_gradient_flow_summary() const {
        std::cout << "\nGradient Flow Summary (relative to output layer):\n";
        std::cout << std::string(60, '-') << "\n";
        std::cout << std::left << std::setw(30) << "Layer"
                  << std::right << std::setw(15) << "L2 Norm"
                  << std::setw(15) << "Flow Scale\n";
        std::cout << std::string(60, '-') << "\n";

        for (const auto& layer : layers_) {
            std::cout << std::left << std::setw(30) << layer.name
                      << std::right << std::setw(15) << std::scientific << std::setprecision(4)
                      << layer.stats.l2_norm
                      << std::setw(15) << std::fixed << std::setprecision(3)
                      << layer.gradient_scale;

            if (layer.is_vanishing) std::cout << "  ⚠️ VANISH";
            if (layer.is_exploding) std::cout << "  ⚠️ EXPLODE";

            std::cout << "\n";
        }
        std::cout << std::string(60, '-') << "\n";
    }

    const std::vector<LayerInfo>& get_layers() const { return layers_; }

    void increment_step() { step_++; }
    int get_step() const { return step_; }
};

// ==============================================================================
// ADAM OPTIMIZER STATE MONITOR - Track Adam's internal state
// ==============================================================================

struct AdamStateStats {
    // First moment (momentum) statistics
    GradientStats m_stats;

    // Second moment (variance) statistics
    GradientStats v_stats;

    // Update statistics (actual parameter updates)
    GradientStats update_stats;

    // Numerical health indicators
    int num_denormal_m;      // Denormalized values in m
    int num_denormal_v;      // Denormalized values in v
    int num_inf_or_nan;      // Inf or NaN in updates
    float min_v;             // Minimum second moment (should be > 0)
    float max_update_ratio;  // max(|update|) / max(|param|)

    AdamStateStats() : num_denormal_m(0), num_denormal_v(0), num_inf_or_nan(0),
                       min_v(0), max_update_ratio(0) {}

    void compute(const float* m, const float* v, const float* updates,
                 const float* params, int size);
    void print() const;
};

class AdamMonitor {
private:
    std::vector<AdamStateStats> history_;
    int max_history_;

public:
    AdamMonitor(int max_history = 100) : max_history_(max_history) {}

    void record(const float* m, const float* v, const float* updates,
                const float* params, int size) {
        AdamStateStats stats;
        stats.compute(m, v, updates, params, size);

        history_.push_back(stats);
        if ((int)history_.size() > max_history_) {
            history_.erase(history_.begin());
        }
    }

    void print_latest() const {
        if (!history_.empty()) {
            history_.back().print();
        }
    }

    void print_trends(int last_n = 10) const;

    bool detect_optimizer_issues() const;
};

// ==============================================================================
// GRADIENT CLIPPING MONITOR - Track gradient clipping statistics
// ==============================================================================

struct GradientClippingStats {
    int step;
    float norm_before;
    float norm_after;
    float clip_threshold;
    bool was_clipped;
    float clip_ratio;  // norm_after / norm_before

    GradientClippingStats() : step(0), norm_before(0), norm_after(0),
                              clip_threshold(0), was_clipped(false), clip_ratio(1.0f) {}

    void record(int step_num, float norm_pre, float norm_post, float threshold) {
        step = step_num;
        norm_before = norm_pre;
        norm_after = norm_post;
        clip_threshold = threshold;
        was_clipped = (norm_pre > threshold);
        clip_ratio = (norm_pre > 1e-10f) ? (norm_post / norm_pre) : 1.0f;
    }

    void print() const {
        std::cout << "Step " << step << " Gradient Clipping:\n";
        std::cout << "  Norm before: " << norm_before << "\n";
        std::cout << "  Norm after:  " << norm_after << "\n";
        std::cout << "  Threshold:   " << clip_threshold << "\n";
        std::cout << "  Clipped:     " << (was_clipped ? "YES" : "NO") << "\n";
        if (was_clipped) {
            std::cout << "  Ratio:       " << clip_ratio << " ("
                      << ((1.0f - clip_ratio) * 100.0f) << "% reduction)\n";
        }
    }
};

class GradientClippingMonitor {
private:
    std::vector<GradientClippingStats> history_;
    int total_steps_;
    int num_clipped_;

public:
    GradientClippingMonitor() : total_steps_(0), num_clipped_(0) {}

    void record(int step, float norm_before, float norm_after, float threshold) {
        GradientClippingStats stats;
        stats.record(step, norm_before, norm_after, threshold);
        history_.push_back(stats);

        total_steps_++;
        if (stats.was_clipped) num_clipped_++;
    }

    void print_summary() const {
        if (total_steps_ == 0) return;

        float clip_rate = (float)num_clipped_ / total_steps_ * 100.0f;

        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              GRADIENT CLIPPING SUMMARY                         ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
        std::cout << "Total steps: " << total_steps_ << "\n";
        std::cout << "Clipped:     " << num_clipped_ << " (" << clip_rate << "%)\n";

        if (!history_.empty()) {
            // Recent clipping rate (last 100 steps)
            int recent_window = std::min(100, (int)history_.size());
            int recent_clipped = 0;
            for (int i = history_.size() - recent_window; i < (int)history_.size(); i++) {
                if (history_[i].was_clipped) recent_clipped++;
            }
            float recent_rate = (float)recent_clipped / recent_window * 100.0f;
            std::cout << "Recent rate: " << recent_rate << "% (last " << recent_window << " steps)\n";

            // Average clipping ratio when clipped
            if (num_clipped_ > 0) {
                float avg_ratio = 0.0f;
                for (const auto& stat : history_) {
                    if (stat.was_clipped) {
                        avg_ratio += stat.clip_ratio;
                    }
                }
                avg_ratio /= num_clipped_;
                std::cout << "Avg clip ratio: " << avg_ratio << " when clipped\n";
                std::cout << "Avg signal loss: " << ((1.0f - avg_ratio) * 100.0f) << "%\n";
            }
        }
    }

    const std::vector<GradientClippingStats>& get_history() const { return history_; }
};

// ==============================================================================
// TRAINING SESSION DIAGNOSTICS - Complete training session monitoring
// ==============================================================================

class TrainingSessionDiagnostics {
private:
    LayerGradientTracker layer_tracker_;
    AdamMonitor adam_monitor_;
    GradientClippingMonitor clip_monitor_;

    std::ofstream csv_file_;
    bool logging_enabled_;
    int current_step_;

    // Training metrics history
    struct TrainingMetrics {
        int step;
        float loss;
        float learning_rate;
        float gradient_norm;
        std::chrono::system_clock::time_point timestamp;
    };
    std::vector<TrainingMetrics> metrics_history_;

public:
    TrainingSessionDiagnostics() : logging_enabled_(false), current_step_(0) {}

    ~TrainingSessionDiagnostics() {
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
    }

    // Initialize layer tracking
    void register_layer(const std::string& name, int num_params) {
        layer_tracker_.add_layer(name, num_params);
    }

    // Enable CSV logging
    void enable_logging(const std::string& log_dir = "diagnostics_logs") {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream filename;
        filename << log_dir << "/training_diagnostics_" << time_t << ".csv";

        csv_file_.open(filename.str());
        if (csv_file_.is_open()) {
            logging_enabled_ = true;
            write_csv_header();
            std::cout << "Logging enabled: " << filename.str() << "\n";
        }
    }

    // Record training step
    void record_step(int step, float loss, float learning_rate, float grad_norm) {
        current_step_ = step;

        TrainingMetrics metrics;
        metrics.step = step;
        metrics.loss = loss;
        metrics.learning_rate = learning_rate;
        metrics.gradient_norm = grad_norm;
        metrics.timestamp = std::chrono::system_clock::now();
        metrics_history_.push_back(metrics);

        if (logging_enabled_) {
            write_csv_row(metrics);
        }
    }

    // Update layer gradients
    void update_layer_gradients(int layer_idx, const float* gradients) {
        layer_tracker_.update_layer(layer_idx, gradients);
    }

    // Record Adam optimizer state
    void record_adam_state(const float* m, const float* v,
                          const float* updates, const float* params, int size) {
        adam_monitor_.record(m, v, updates, params, size);
    }

    // Record gradient clipping
    void record_clipping(float norm_before, float norm_after, float threshold) {
        clip_monitor_.record(current_step_, norm_before, norm_after, threshold);
    }

    // Print comprehensive diagnostics
    void print_diagnostics(bool include_layer_details = true) {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           TRAINING DIAGNOSTICS REPORT                          ║\n";
        std::cout << "║           Step: " << std::setw(5) << current_step_
                  << "                                              ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

        if (include_layer_details) {
            layer_tracker_.print_summary();
        } else {
            layer_tracker_.print_gradient_flow_summary();
        }

        clip_monitor_.print_summary();

        adam_monitor_.print_latest();
    }

    void print_quick_summary() {
        print_diagnostics(false);
    }

    LayerGradientTracker& get_layer_tracker() { return layer_tracker_; }
    AdamMonitor& get_adam_monitor() { return adam_monitor_; }
    GradientClippingMonitor& get_clip_monitor() { return clip_monitor_; }

private:
    void write_csv_header() {
        csv_file_ << "step,loss,learning_rate,gradient_norm,timestamp\n";
    }

    void write_csv_row(const TrainingMetrics& m) {
        auto time_t = std::chrono::system_clock::to_time_t(m.timestamp);
        csv_file_ << m.step << "," << m.loss << "," << m.learning_rate << ","
                  << m.gradient_norm << "," << time_t << "\n";
        csv_file_.flush();
    }
};

} // namespace TrainingDiagnostics
