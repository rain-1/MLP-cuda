#pragma once

#include "transformer.h"
#include "text_dataset.h"
#include "wandb_logger.h"
#include "gradient_utils.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <time.h>

struct TrainingConfig {
    float learning_rate = 1e-4f;  // Lowered from 1e-3 for stability
    float grad_clip_norm = 1.0f;   // Gradient clipping threshold
    int num_epochs = 10;
    bool monitor_grads = true;      // Monitor gradient norms
    bool check_nans = true;         // Check for NaN/Inf
    int log_interval = 10;          // Log every N batches
};

template<typename TokenizerType>
int run_training(
    TokenizerType& tokenizer,
    const char* text_file,
    const std::string& training_text,
    bool use_wandb,
    const char* wandb_project,
    const char* wandb_run,
    const TrainingConfig& config = TrainingConfig()
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
        std::map<std::string, double> wandb_config;
        wandb_config["vocab_size"] = vocab_size;
        wandb_config["d_model"] = d_model;
        wandb_config["num_layers"] = num_layers;
        wandb_config["num_heads"] = num_heads;
        wandb_config["d_ff"] = d_ff;
        wandb_config["max_seq_len"] = max_seq_len;
        wandb_config["seq_len"] = seq_len;
        wandb_config["batch_size"] = batch_size;
        wandb_config["num_epochs"] = config.num_epochs;
        wandb_config["learning_rate"] = config.learning_rate;
        wandb_config["grad_clip_norm"] = config.grad_clip_norm;
        wandb_logger.log_config_num(wandb_config);
    }

    // 4. Train the model
    printf("4. Training transformer...\n");
    printf("   Training for %d epochs with batch size %d\n", config.num_epochs, batch_size);
    printf("   Learning rate: %.4e (10x lower for stability)\n", config.learning_rate);
    printf("   Gradient clipping: %.2f\n", config.grad_clip_norm);
    printf("   Monitoring: grads=%s, NaNs=%s\n\n",
           config.monitor_grads ? "yes" : "no",
           config.check_nans ? "yes" : "no");

    time_t start_time = time(nullptr);

    float last_loss = 0.0f;
    bool training_diverged = false;

    for (int epoch = 0; epoch < config.num_epochs && !training_diverged; epoch++) {
        printf("   === Epoch %d/%d ===\n", epoch + 1, config.num_epochs);

        float total_loss = 0.0f;
        int num_samples = 0;
        float max_grad_norm = 0.0f;

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

            // Training step: forward + backward + gradient clipping + optimizer update
            float loss = model.train_step(inputs.data(), targets.data(),
                                         batch_size, seq_len,
                                         config.learning_rate, config.grad_clip_norm);

            // Check for NaN/Inf in loss
            if (config.check_nans && (isnan(loss) || isinf(loss))) {
                printf("\n\n   *** TRAINING DIVERGED: Loss is %s at epoch %d, batch %d ***\n",
                       isnan(loss) ? "NaN" : "Inf", epoch + 1, batch_idx + 1);
                printf("   This usually indicates:\n");
                printf("     1. Learning rate too high\n");
                printf("     2. Gradient explosion\n");
                printf("     3. Numerical instability in loss computation\n");
                printf("   Try:\n");
                printf("     - Lower learning rate (current: %.4e)\n", config.learning_rate);
                printf("     - Tighter gradient clipping (current: %.2f)\n", config.grad_clip_norm);
                printf("     - Check your data for issues\n\n");
                training_diverged = true;
                break;
            }

            total_loss += loss;
            num_samples++;

            // Monitor gradient norms (every log_interval batches)
            if (config.monitor_grads && (batch_idx % config.log_interval == 0)) {
                float grad_norm = model.compute_gradient_norm();
                max_grad_norm = fmaxf(max_grad_norm, grad_norm);

                // Log gradient norm
                if (use_wandb) {
                    int global_step = epoch * num_batches + batch_idx;
                    wandb_logger.set_step(global_step);

                    std::map<std::string, double> grad_metrics;
                    grad_metrics["train/grad_norm"] = grad_norm;
                    wandb_logger.log_metrics(grad_metrics);
                }
            }

            // Log to wandb
            if (use_wandb) {
                int global_step = epoch * num_batches + batch_idx;
                wandb_logger.set_step(global_step);

                std::map<std::string, double> metrics;
                metrics["train/loss"] = loss;
                metrics["train/learning_rate"] = config.learning_rate;
                metrics["train/epoch"] = epoch + 1;
                wandb_logger.log_metrics(metrics);
            }

            // Print progress
            if ((batch_idx + 1) % config.log_interval == 0 || batch_idx == num_batches - 1) {
                if (config.monitor_grads) {
                    printf("\r   Epoch %d/%d, Batch %d/%d, Loss: %.4f, GradNorm: %.2f   ",
                           epoch + 1, config.num_epochs, batch_idx + 1, num_batches, loss, max_grad_norm);
                } else {
                    printf("\r   Epoch %d/%d, Batch %d/%d, Loss: %.4f   ",
                           epoch + 1, config.num_epochs, batch_idx + 1, num_batches, loss);
                }
                fflush(stdout);
            }
        }

        if (training_diverged) break;

        float avg_loss = total_loss / num_samples;
        printf("\n   Epoch %d complete - Avg loss: %.4f", epoch + 1, avg_loss);

        if (config.monitor_grads) {
            printf(", Max grad norm: %.2f", max_grad_norm);
        }

        // Check for loss increase (potential divergence)
        if (epoch > 0 && avg_loss > last_loss * 1.5f) {
            printf(" [WARNING: Loss increased by >50%%!]");
        } else if (epoch > 0 && avg_loss < last_loss * 0.5f) {
            printf(" [Good: Loss decreased by >50%%]");
        }
        printf("\n\n");

        last_loss = avg_loss;

        // Log epoch average loss to wandb
        if (use_wandb) {
            int global_step = (epoch + 1) * num_batches;
            wandb_logger.set_step(global_step);

            std::map<std::string, double> metrics;
            metrics["train/epoch_avg_loss"] = avg_loss;
            if (config.monitor_grads) {
                metrics["train/epoch_max_grad_norm"] = max_grad_norm;
            }
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
    printf("Training %s in %d seconds\n\n",
           training_diverged ? "stopped (diverged)" : "complete", elapsed);

    // Log training time to wandb
    if (use_wandb) {
        std::map<std::string, double> metrics;
        metrics["train/total_time_seconds"] = elapsed;
        metrics["train/diverged"] = training_diverged ? 1.0 : 0.0;
        wandb_logger.log_metrics(metrics);
    }

    if (training_diverged) {
        printf("Training diverged - model not saved.\n");
        printf("Please try again with:\n");
        printf("  - Lower learning rate (--lr %.4e)\n", config.learning_rate / 10.0f);
        printf("  - Tighter clipping (--grad-clip %.2f)\n\n", config.grad_clip_norm / 2.0f);
        return 1;
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
