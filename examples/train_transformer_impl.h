#pragma once

#include "transformer.h"
#include "text_dataset.h"
#include "wandb_logger.h"
#include "diagnostics.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <time.h>
#include <cmath>
#include <algorithm>

// Learning rate schedule with linear warmup and optional cosine decay
inline float get_learning_rate_with_warmup(
    int step,
    float base_lr,
    int warmup_steps,
    int total_steps = -1,  // -1 means no decay
    float min_lr = 0.0f
) {
    // Linear warmup
    if (step < warmup_steps) {
        return base_lr * (static_cast<float>(step + 1) / warmup_steps);
    }

    // Constant LR after warmup (if no decay)
    if (total_steps <= 0) {
        return base_lr;
    }

    // Cosine decay after warmup
    float decay_ratio = static_cast<float>(step - warmup_steps) /
                       (total_steps - warmup_steps);
    decay_ratio = std::min(1.0f, std::max(0.0f, decay_ratio));

    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return min_lr + (base_lr - min_lr) * coeff;
}

template<typename TokenizerType>
int run_training(
    TokenizerType& tokenizer,
    const char* text_file,
    const std::string& training_text,
    bool use_wandb,
    const char* wandb_project,
    const char* wandb_run
) {
    // Model hyperparameters (small model for demo)
    int d_model = 128;
    int num_layers = 4;
    int num_heads = 4;
    int d_ff = 512;
    int max_seq_len = 64;
    int seq_len = 32;  // Actual training sequence length

    // Training hyperparameters
    int batch_size = 8;
    int num_epochs = 10;
    float base_learning_rate = 1e-3f;
    int warmup_steps = 500;  // Linear warmup for first 500 steps

    int vocab_size = tokenizer.vocab_size();

    // 2. Create dataset
    printf("2. Creating dataset...\n");
    TextDataset<TokenizerType> dataset(tokenizer, seq_len, batch_size);

    if (text_file != nullptr) {
        dataset.load_from_file(text_file);
    } else {
        dataset.load_from_text(training_text);
    }

    int num_batches = dataset.get_num_batches();
    printf("   Number of batches: %d\n\n", num_batches);

    if (num_batches == 0) {
        fprintf(stderr, "Error: Not enough data to create batches\n");
        return 1;
    }

    // 3. Create transformer model
    printf("3. Creating transformer model...\n");
    Transformer model(vocab_size, d_model, num_layers, num_heads, d_ff,
                     max_seq_len, batch_size);

    printf("   Model parameters:\n");
    printf("     - Vocabulary size: %d\n", vocab_size);
    printf("     - Model dimension: %d\n", d_model);
    printf("     - Layers: %d\n", num_layers);
    printf("     - Attention heads: %d\n", num_heads);
    printf("     - FFN dimension: %d\n", d_ff);
    printf("     - Max sequence length: %d\n\n", max_seq_len);

    // Initialize wandb logger
    WandbLogger wandb_logger("training_metrics.jsonl");
    wandb_logger.set_enabled(use_wandb);

    if (use_wandb) {
        printf("   Wandb logging enabled\n");
        printf("     - Project: %s\n", wandb_project);
        if (wandb_run) printf("     - Run: %s\n", wandb_run);
        printf("\n");

        // Log configuration
        std::map<std::string, double> config;
        config["vocab_size"] = vocab_size;
        config["d_model"] = d_model;
        config["num_layers"] = num_layers;
        config["num_heads"] = num_heads;
        config["d_ff"] = d_ff;
        config["max_seq_len"] = max_seq_len;
        config["seq_len"] = seq_len;
        config["batch_size"] = batch_size;
        config["num_epochs"] = num_epochs;
        config["base_learning_rate"] = base_learning_rate;
        config["warmup_steps"] = warmup_steps;
        wandb_logger.log_config_num(config);
    }

    // 4. Train the model
    printf("4. Training transformer...\n");
    printf("   Training for %d epochs with batch size %d\n", num_epochs, batch_size);
    printf("   Base learning rate: %.4f (with %d step warmup)\n\n", base_learning_rate, warmup_steps);

    // Initialize diagnostic logger
    DiagnosticLogger diag_logger;
    diag_logger.set_enabled(true);
    diag_logger.set_grad_norm_threshold(100.0f);
    diag_logger.set_param_norm_threshold(1000.0f);
    diag_logger.set_loss_increase_threshold(2.0f);

    time_t start_time = time(nullptr);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("   === Epoch %d/%d ===\n", epoch + 1, num_epochs);

        float total_loss = 0.0f;
        float total_grad_norm = 0.0f;
        float total_param_norm = 0.0f;
        int num_samples = 0;

        // Shuffle dataset every few epochs
        if (epoch > 0 && epoch % 5 == 0) {
            dataset.shuffle(epoch);
        }
        dataset.reset();

        // Iterate through batches
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            std::vector<int> inputs, targets;
            if (!dataset.get_batch(batch_idx, inputs, targets)) {
                break;
            }

            int global_step = epoch * num_batches + batch_idx;

            // Compute learning rate with warmup
            float current_lr = get_learning_rate_with_warmup(
                global_step, base_learning_rate, warmup_steps
            );

            // Training step with diagnostics
            // Enable detailed logging on first batch of each epoch
            bool detailed_logging = (batch_idx == 0);
            TrainingDiagnostics diag = model.train_step_with_diagnostics(
                inputs.data(), targets.data(),
                batch_size, seq_len, current_lr,
                detailed_logging
            );

            total_loss += diag.loss;
            total_grad_norm += diag.grad_norm;
            total_param_norm += diag.param_norm;
            num_samples++;

            // Log diagnostics every 10 batches
            if ((batch_idx + 1) % 10 == 0 || batch_idx == 0) {
                diag_logger.log_step(global_step, diag.loss, current_lr,
                                   diag.grad_norm, diag.param_norm);

                // Check for divergence
                if (diag_logger.check_divergence(global_step, diag.loss)) {
                    printf("\n   [ERROR] Training has diverged! Stopping...\n");
                    return 1;
                }

                // Check for NaN or Inf
                if (diag.has_nan_or_inf) {
                    printf("\n   [ERROR] NaN or Inf detected in gradients/parameters! Stopping...\n");
                    return 1;
                }
            }

            // Log to wandb
            if (use_wandb) {
                wandb_logger.set_step(global_step);

                std::map<std::string, double> metrics;
                metrics["train/loss"] = diag.loss;
                metrics["train/grad_norm"] = diag.grad_norm;
                metrics["train/param_norm"] = diag.param_norm;
                metrics["train/learning_rate"] = current_lr;
                metrics["train/epoch"] = epoch + 1;
                wandb_logger.log_metrics(metrics);
            }

            // Print progress
            if ((batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1) {
                printf("\r   Epoch %d/%d, Batch %d/%d, Loss: %.4f, GradNorm: %.2f, LR: %.6f",
                       epoch + 1, num_epochs, batch_idx + 1, num_batches,
                       diag.loss, diag.grad_norm, current_lr);
                fflush(stdout);
            }
        }

        float avg_loss = total_loss / num_samples;
        float avg_grad_norm = total_grad_norm / num_samples;
        float avg_param_norm = total_param_norm / num_samples;
        printf("\n   Epoch %d complete - Avg Loss: %.4f | Avg GradNorm: %.2f | Avg ParamNorm: %.2f\n\n",
               epoch + 1, avg_loss, avg_grad_norm, avg_param_norm);

        // Log epoch average loss to wandb
        if (use_wandb) {
            int global_step = (epoch + 1) * num_batches;
            wandb_logger.set_step(global_step);

            std::map<std::string, double> metrics;
            metrics["train/epoch_avg_loss"] = avg_loss;
            wandb_logger.log_metrics(metrics);
        }

        // Generate sample text every few epochs
        if ((epoch + 1) % 2 == 0) {
            printf("   Sample generation:\n");
            std::string prompt = "once upon";
            std::vector<int> prompt_tokens = tokenizer.encode(prompt);
            std::vector<int> generated = model.generate(prompt_tokens, 30, 0.8f, 0, 1.0f, epoch);
            std::string generated_text = tokenizer.decode(generated);
            printf("   \"%s\"\n\n", generated_text.c_str());

            // Log sample to wandb
            if (use_wandb) {
                wandb_logger.log_sample(prompt, generated_text);
            }
        }
    }

    time_t end_time = time(nullptr);
    int elapsed = (int)(end_time - start_time);
    printf("Training complete in %d seconds\n\n", elapsed);

    // Print diagnostic summary
    diag_logger.print_summary();

    // Log training time to wandb
    if (use_wandb) {
        std::map<std::string, double> metrics;
        metrics["train/total_time_seconds"] = elapsed;
        wandb_logger.log_metrics(metrics);
    }

    // 5. Save model
    const char* model_path = "trained_transformer.bin";
    printf("5. Saving model to %s...\n", model_path);
    model.save_parameters(model_path);
    tokenizer.save_vocab("tokenizer.vocab");
    printf("   Model and tokenizer saved\n\n");

    // 6. Final generation test
    printf("6. Final generation test:\n");
    std::string prompts[] = {
        "once upon",
        "she was",
        "the forest"
    };

    for (const auto& prompt : prompts) {
        std::vector<int> prompt_tokens = tokenizer.encode(prompt);
        std::vector<int> generated = model.generate(prompt_tokens, 40, 0.7f, 0, 1.0f, 42);
        std::string generated_text = tokenizer.decode(generated);
        printf("   Prompt: \"%s\"\n", prompt.c_str());
        printf("   Output: \"%s\"\n\n", generated_text.c_str());

        // Log final samples to wandb
        if (use_wandb) {
            wandb_logger.log_sample(prompt, generated_text);
        }
    }

    // Finish wandb logging
    if (use_wandb) {
        wandb_logger.finish();
        printf("   Wandb logging complete. Run: python scripts/wandb_logger.py training_metrics.jsonl %s %s\n\n",
               wandb_project, wandb_run ? wandb_run : "");
    }

    printf("=== Training demo complete! ===\n");

    return 0;
}
