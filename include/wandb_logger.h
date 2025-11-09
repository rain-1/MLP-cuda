#ifndef WANDB_LOGGER_H
#define WANDB_LOGGER_H

#include <fstream>
#include <string>
#include <map>
#include <sstream>

/**
 * Simple logger that writes metrics to a JSON file for wandb integration.
 * Works in conjunction with wandb_logger.py Python script.
 */
class WandbLogger {
public:
    WandbLogger(const std::string& metrics_file = "training_metrics.jsonl")
        : metrics_file_(metrics_file), step_(0), enabled_(true)
    {
        // Clear the file on initialization
        std::ofstream file(metrics_file_, std::ios::trunc);
        file.close();
    }

    void set_enabled(bool enabled) {
        enabled_ = enabled;
    }

    bool is_enabled() const {
        return enabled_;
    }

    // Log configuration
    void log_config(const std::map<std::string, std::string>& config) {
        if (!enabled_) return;

        std::ofstream file(metrics_file_, std::ios::app);
        file << "{\"command\": \"init\", \"config\": {";

        bool first = true;
        for (const auto& pair : config) {
            if (!first) file << ", ";
            file << "\"" << pair.first << "\": \"" << pair.second << "\"";
            first = false;
        }

        file << "}}\n";
        file.flush();
    }

    // Log numeric config
    void log_config_num(const std::map<std::string, double>& config) {
        if (!enabled_) return;

        std::ofstream file(metrics_file_, std::ios::app);
        file << "{\"command\": \"init\", \"config\": {";

        bool first = true;
        for (const auto& pair : config) {
            if (!first) file << ", ";
            file << "\"" << pair.first << "\": " << pair.second;
            first = false;
        }

        file << "}}\n";
        file.flush();
    }

    // Log metrics at current step
    void log_metrics(const std::map<std::string, double>& metrics) {
        if (!enabled_) return;

        std::ofstream file(metrics_file_, std::ios::app);
        file << "{\"step\": " << step_ << ", \"metrics\": {";

        bool first = true;
        for (const auto& pair : metrics) {
            if (!first) file << ", ";
            file << "\"" << pair.first << "\": " << pair.second;
            first = false;
        }

        file << "}}\n";
        file.flush();
    }

    // Log a single metric
    void log_metric(const std::string& key, double value) {
        if (!enabled_) return;

        std::map<std::string, double> metrics;
        metrics[key] = value;
        log_metrics(metrics);
    }

    // Log text samples
    void log_sample(const std::string& prompt, const std::string& output) {
        if (!enabled_) return;

        std::ofstream file(metrics_file_, std::ios::app);
        file << "{\"step\": " << step_ << ", \"samples\": {";
        file << "\"prompt\": \"" << escape_json(prompt) << "\", ";
        file << "\"output\": \"" << escape_json(output) << "\"";
        file << "}}\n";
        file.flush();
    }

    // Increment step counter
    void set_step(int step) {
        step_ = step;
    }

    int get_step() const {
        return step_;
    }

    // Finish logging
    void finish() {
        if (!enabled_) return;

        std::ofstream file(metrics_file_, std::ios::app);
        file << "{\"command\": \"finish\"}\n";
        file.flush();
    }

private:
    std::string metrics_file_;
    int step_;
    bool enabled_;

    // Escape special characters for JSON
    std::string escape_json(const std::string& str) {
        std::stringstream ss;
        for (char c : str) {
            switch (c) {
                case '"': ss << "\\\""; break;
                case '\\': ss << "\\\\"; break;
                case '\b': ss << "\\b"; break;
                case '\f': ss << "\\f"; break;
                case '\n': ss << "\\n"; break;
                case '\r': ss << "\\r"; break;
                case '\t': ss << "\\t"; break;
                default: ss << c; break;
            }
        }
        return ss.str();
    }
};

#endif // WANDB_LOGGER_H
