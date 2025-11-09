#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "precision_analysis.h"
#include "training_diagnostics.h"

using namespace PrecisionAnalysis;
using namespace TrainingDiagnostics;

// ==============================================================================
// TEST 1: Precision Analysis - Compare Summation Algorithms
// ==============================================================================

void test_kahan_summation() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       TEST 1: KAHAN SUMMATION PRECISION ANALYSIS               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    // Create a case where naive summation loses precision
    std::vector<float> values;

    // Add a large value
    values.push_back(1e8f);

    // Add many small values (1000 values of 0.1)
    for (int i = 0; i < 1000; i++) {
        values.push_back(0.1f);
    }

    // Subtract the large value
    values.push_back(-1e8f);

    // Expected result: 1000 * 0.1 = 100.0

    // Naive summation
    float naive_sum = 0.0f;
    for (float v : values) {
        naive_sum += v;
    }

    // Kahan summation
    KahanAccumulator kahan;
    for (float v : values) {
        kahan.add(v);
    }
    float kahan_sum = kahan.get();

    // Double precision reference
    double reference = 0.0;
    for (float v : values) {
        reference += (double)v;
    }

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Expected result: 100.0\n";
    std::cout << "Naive sum:       " << naive_sum
              << " (error: " << std::abs(naive_sum - 100.0f) << ")\n";
    std::cout << "Kahan sum:       " << kahan_sum
              << " (error: " << std::abs(kahan_sum - 100.0f) << ")\n";
    std::cout << "Double ref:      " << reference
              << " (error: " << std::abs(reference - 100.0) << ")\n";

    float improvement = (std::abs(naive_sum - 100.0f) - std::abs(kahan_sum - 100.0f))
                       / std::abs(naive_sum - 100.0f) * 100.0f;
    std::cout << "\nKahan improvement: " << improvement << "%\n";
}

// ==============================================================================
// TEST 2: Softmax Precision Analysis
// ==============================================================================

void test_softmax_precision() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       TEST 2: SOFTMAX PRECISION COMPARISON                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    // Create logits with varying magnitudes
    std::vector<float> logits = {
        -5.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f, 10.0f
    };

    std::cout << "Logits: [";
    for (size_t i = 0; i < logits.size(); i++) {
        std::cout << logits[i];
        if (i < logits.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";

    auto result = SoftmaxPrecisionAnalyzer::analyze(logits.data(), logits.size());
    SoftmaxPrecisionAnalyzer::print_comparison(result);
}

// ==============================================================================
// TEST 3: Gradient Statistics
// ==============================================================================

void test_gradient_statistics() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       TEST 3: GRADIENT STATISTICS ANALYSIS                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.01f);  // Mean 0, std 0.01

    // Create synthetic gradients
    std::vector<float> gradients(10000);
    for (size_t i = 0; i < gradients.size(); i++) {
        gradients[i] = dist(gen);

        // Add some large outliers (5% of data)
        if (i % 20 == 0) {
            gradients[i] *= 10.0f;
        }

        // Add some near-zero values (2% of data)
        if (i % 50 == 0) {
            gradients[i] = 1e-7f;
        }
    }

    GradientStats stats;
    stats.compute(gradients.data(), gradients.size());
    stats.print("Test Gradients");
}

// ==============================================================================
// TEST 4: Layer Gradient Tracking
// ==============================================================================

void test_layer_gradient_tracking() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       TEST 4: LAYER-BY-LAYER GRADIENT TRACKING                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    LayerGradientTracker tracker;

    // Register layers
    tracker.add_layer("output_layer", 1000);
    tracker.add_layer("hidden_layer_2", 2048);
    tracker.add_layer("hidden_layer_1", 2048);
    tracker.add_layer("embedding", 5000);

    std::mt19937 gen(123);

    // Simulate gradient flow with varying magnitudes (normal in output, vanishing in early layers)
    for (int layer = 0; layer < 4; layer++) {
        // Create gradients with decreasing magnitude as we go backwards
        float scale = std::pow(0.5f, layer);  // Each layer has half the gradient magnitude
        std::normal_distribution<float> dist(0.0f, 0.01f * scale);

        int size = (layer == 3) ? 5000 : (layer == 0) ? 1000 : 2048;
        std::vector<float> gradients(size);

        for (int i = 0; i < size; i++) {
            gradients[i] = dist(gen);
        }

        tracker.update_layer(layer, gradients.data());
    }

    tracker.print_summary();
    tracker.print_gradient_flow_summary();
}

// ==============================================================================
// TEST 5: Adam Optimizer State Monitoring
// ==============================================================================

void test_adam_monitoring() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       TEST 5: ADAM OPTIMIZER STATE MONITORING                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    AdamMonitor monitor;
    std::mt19937 gen(456);
    std::normal_distribution<float> dist(0.0f, 0.01f);

    const int size = 1000;
    std::vector<float> m(size);
    std::vector<float> v(size);
    std::vector<float> updates(size);
    std::vector<float> params(size);

    // Initialize parameters
    for (int i = 0; i < size; i++) {
        params[i] = dist(gen);
    }

    // Simulate several optimizer steps
    for (int step = 0; step < 5; step++) {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float lr = 0.001f;
        float eps = 1e-8f;

        for (int i = 0; i < size; i++) {
            float grad = dist(gen);

            // Update moments
            m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
            v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;

            // Compute update
            updates[i] = lr * m[i] / (std::sqrt(v[i]) + eps);
        }

        monitor.record(m.data(), v.data(), updates.data(), params.data(), size);
    }

    monitor.print_latest();
    monitor.print_trends(5);

    if (monitor.detect_optimizer_issues()) {
        std::cout << "\n⚠️  Issues detected in optimizer state\n";
    } else {
        std::cout << "\n✓ No issues detected\n";
    }
}

// ==============================================================================
// TEST 6: Gradient Clipping Monitor
// ==============================================================================

void test_gradient_clipping() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       TEST 6: GRADIENT CLIPPING MONITORING                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    GradientClippingMonitor monitor;
    std::mt19937 gen(789);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float clip_threshold = 1.0f;

    // Simulate training with varying gradient magnitudes
    for (int step = 0; step < 100; step++) {
        // Gradually increase gradient magnitude to simulate instability
        float magnitude_scale = 1.0f + step * 0.05f;
        float grad_norm = std::abs(dist(gen)) * magnitude_scale;

        // Clip if necessary
        float clipped_norm = std::min(grad_norm, clip_threshold);

        monitor.record(step, grad_norm, clipped_norm, clip_threshold);
    }

    monitor.print_summary();

    std::cout << "\nRecent clipping events:\n";
    const auto& history = monitor.get_history();
    int num_recent = std::min(5, (int)history.size());
    for (int i = history.size() - num_recent; i < (int)history.size(); i++) {
        std::cout << "  Step " << history[i].step << ": "
                  << std::fixed << std::setprecision(4)
                  << history[i].norm_before << " -> " << history[i].norm_after;
        if (history[i].was_clipped) {
            std::cout << " ✂️ CLIPPED";
        }
        std::cout << "\n";
    }
}

// ==============================================================================
// TEST 7: Complete Training Session Diagnostics
// ==============================================================================

void test_training_session_diagnostics() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       TEST 7: COMPLETE TRAINING SESSION DIAGNOSTICS            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    TrainingSessionDiagnostics diagnostics;

    // Register layers
    diagnostics.register_layer("loss_output", 34);
    diagnostics.register_layer("transformer_block_3", 2048);
    diagnostics.register_layer("transformer_block_2", 2048);
    diagnostics.register_layer("transformer_block_1", 2048);
    diagnostics.register_layer("transformer_block_0", 2048);
    diagnostics.register_layer("embedding", 1024);

    std::mt19937 gen(999);
    std::normal_distribution<float> dist(0.0f, 0.01f);

    // Simulate a few training steps
    for (int step = 0; step < 3; step++) {
        std::cout << "\n\n=== Simulating Training Step " << step << " ===\n";

        float loss = 4.0f - step * 0.1f;  // Decreasing loss
        float lr = 0.001f;
        float grad_norm = 2.0f + step * 0.5f;  // Increasing gradient norm

        diagnostics.record_step(step, loss, lr, grad_norm);

        // Simulate layer gradients (with vanishing gradients in early layers)
        for (int layer = 0; layer < 6; layer++) {
            int sizes[] = {34, 2048, 2048, 2048, 2048, 1024};
            float scales[] = {1.0f, 0.8f, 0.5f, 0.3f, 0.1f, 0.05f};

            std::vector<float> gradients(sizes[layer]);
            for (int i = 0; i < sizes[layer]; i++) {
                gradients[i] = dist(gen) * scales[layer];
            }

            diagnostics.update_layer_gradients(layer, gradients.data());
        }

        // Simulate gradient clipping
        float norm_before = grad_norm;
        float norm_after = std::min(grad_norm, 1.0f);
        diagnostics.record_clipping(norm_before, norm_after, 1.0f);

        // Simulate Adam state (only for last step to save output)
        if (step == 2) {
            std::vector<float> m(2048), v(2048), updates(2048), params(2048);
            for (int i = 0; i < 2048; i++) {
                m[i] = dist(gen) * 0.9f;
                v[i] = std::abs(dist(gen)) * 0.001f;
                updates[i] = dist(gen) * 0.0001f;
                params[i] = dist(gen);
            }
            diagnostics.record_adam_state(m.data(), v.data(), updates.data(),
                                         params.data(), 2048);
        }

        // Print diagnostics for this step
        if (step == 2) {
            diagnostics.print_diagnostics();
        } else {
            diagnostics.print_quick_summary();
        }
    }
}

// ==============================================================================
// TEST 8: Cross-Entropy Gradient Precision (Float32 vs Float64)
// ==============================================================================

void test_cross_entropy_gradient_precision() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       TEST 8: CROSS-ENTROPY GRADIENT PRECISION                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    const int vocab_size = 34;
    const int target = 5;

    // Create logits with realistic range
    std::vector<float> logits(vocab_size);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    std::cout << "\nLogits: [";
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = dist(gen);
        if (i < 5 || i >= vocab_size - 2) {
            std::cout << std::fixed << std::setprecision(2) << logits[i];
            if (i == 4 && vocab_size > 7) std::cout << " ... ";
            else if (i < vocab_size - 1) std::cout << ", ";
        }
    }
    std::cout << "]\n";
    std::cout << "Target: " << target << "\n\n";

    std::vector<float> grad_f32(vocab_size);
    std::vector<double> grad_f64(vocab_size);

    GradientPrecisionAnalyzer::compare_cross_entropy_gradient(
        logits.data(), vocab_size, target,
        grad_f32.data(), grad_f64.data()
    );

    // Compute statistics
    GradientStats stats_f32, stats_f64_as_f32;
    stats_f32.compute(grad_f32.data(), vocab_size);

    std::vector<float> grad_f64_float(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        grad_f64_float[i] = (float)grad_f64[i];
    }
    stats_f64_as_f32.compute(grad_f64_float.data(), vocab_size);

    std::cout << "Float32 gradients:\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  L2 norm: " << stats_f32.l2_norm << "\n";
    std::cout << "  Range: [" << stats_f32.min_value << ", " << stats_f32.max_value << "]\n";

    std::cout << "\nFloat64 gradients:\n";
    std::cout << "  L2 norm: " << stats_f64_as_f32.l2_norm << "\n";
    std::cout << "  Range: [" << stats_f64_as_f32.min_value << ", "
              << stats_f64_as_f32.max_value << "]\n";

    // Compute differences
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float diff = std::abs(grad_f32[i] - (float)grad_f64[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= vocab_size;

    std::cout << "\nPrecision comparison:\n";
    std::cout << "  Max difference: " << max_diff << "\n";
    std::cout << "  Avg difference: " << avg_diff << "\n";
    std::cout << "  Relative error: " << (avg_diff / stats_f64_as_f32.abs_mean * 100.0f) << "%\n";
}

// ==============================================================================
// MAIN
// ==============================================================================

int main() {
    std::cout << "\n";
    std::cout << "████████████████████████████████████████████████████████████████\n";
    std::cout << "█                                                              █\n";
    std::cout << "█        NUMERICAL PRECISION & TRAINING DIAGNOSTICS            █\n";
    std::cout << "█                    TEST SUITE                                █\n";
    std::cout << "█                                                              █\n";
    std::cout << "████████████████████████████████████████████████████████████████\n";

    try {
        test_kahan_summation();
        test_softmax_precision();
        test_gradient_statistics();
        test_layer_gradient_tracking();
        test_adam_monitoring();
        test_gradient_clipping();
        test_training_session_diagnostics();
        test_cross_entropy_gradient_precision();

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    ALL TESTS COMPLETED                         ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
