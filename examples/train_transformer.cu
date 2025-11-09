#include "transformer.h"
#include "tokenizer.h"
#include "text_dataset.h"
#include "loss.h"
#include "wandb_logger.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <time.h>
#include <cstring>

int main(int argc, char** argv) {
    printf("=== Transformer Training Demo ===\n\n");

    // Parse command line arguments
    const char* text_file = nullptr;
    bool use_wandb = false;
    const char* wandb_project = "transformer-training";
    const char* wandb_run = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--wandb") == 0) {
            use_wandb = true;
        } else if (strcmp(argv[i], "--wandb-project") == 0 && i + 1 < argc) {
            wandb_project = argv[++i];
        } else if (strcmp(argv[i], "--wandb-run") == 0 && i + 1 < argc) {
            wandb_run = argv[++i];
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            text_file = argv[++i];
        } else {
            // Backward compatibility: first positional arg is data file
            if (text_file == nullptr) {
                text_file = argv[i];
            }
        }
    }

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
    float learning_rate = 1e-3f;

    // Sample training text (if no file provided)
    std::string sample_text =
        "once upon a time there was a little girl. "
        "she lived in a small house in the woods. "
        "every day she would go for a walk in the forest. "
        "the trees were tall and the birds sang songs. "
        "she was very happy in her little home. "
        "one day she found a magical flower. "
        "the flower was blue and very pretty. "
        "she brought it home and put it in water. "
        "the flower made her home smell nice. "
        "she smiled and thanked the forest. "
        "the end. ";

    // Repeat the text to have more data
    std::string training_text;
    for (int i = 0; i < 20; i++) {
        training_text += sample_text;
    }

    // 1. Build tokenizer
    printf("1. Building tokenizer...\n");
    Tokenizer tokenizer;

    if (text_file != nullptr) {
        tokenizer.build_vocab_from_file(text_file);
    } else {
        tokenizer.build_vocab(training_text);
    }

    int vocab_size = tokenizer.vocab_size();
    printf("   Vocabulary size: %d\n", vocab_size);
    printf("   Special tokens: PAD=%d, BOS=%d, EOS=%d\n\n",
           tokenizer.pad_token(), tokenizer.bos_token(), tokenizer.eos_token());

    // 2. Create dataset
    printf("2. Creating dataset...\n");
    TextDataset dataset(tokenizer, seq_len, batch_size);

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

    // 3. Create model
    printf("3. Creating transformer model...\n");
    Transformer model(vocab_size, d_model, num_layers, num_heads,
                     d_ff, max_seq_len, batch_size);

    printf("   Model parameters:\n");
    printf("   - d_model: %d\n", d_model);
    printf("   - num_layers: %d\n", num_layers);
    printf("   - num_heads: %d\n", num_heads);
    printf("   - d_ff: %d\n", d_ff);
    printf("   - seq_len: %d\n\n", seq_len);

    // Initialize wandb logger
    WandbLogger wandb_logger("training_metrics.jsonl");
    wandb_logger.set_enabled(use_wandb);

    if (use_wandb) {
        printf("   Wandb: enabled (project=%s, run=%s)\n",
               wandb_project, wandb_run ? wandb_run : "auto");
        printf("   To view logs, run: python scripts/wandb_logger.py training_metrics.jsonl %s %s\n\n",
               wandb_project, wandb_run ? wandb_run : "");

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
        config["learning_rate"] = learning_rate;
        config["num_batches"] = num_batches;
        wandb_logger.log_config_num(config);
    }

    // 4. Training loop
    printf("4. Training...\n");
    printf("   Epochs: %d\n", num_epochs);
    printf("   Batch size: %d\n", batch_size);
    printf("   Learning rate: %.4f\n\n", learning_rate);

    time_t start_time = time(nullptr);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;
        int num_samples = 0;

        // Shuffle dataset at the start of each epoch
        if (epoch > 0) {
            dataset.shuffle(epoch);
        }
        dataset.reset();

        // Iterate through batches
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            std::vector<int> inputs, targets;
            if (!dataset.get_batch(batch_idx, inputs, targets)) {
                break;
            }

            // Training step: forward + backward + optimizer update
            float loss = model.train_step(inputs.data(), targets.data(),
                                         batch_size, seq_len, learning_rate);

            total_loss += loss;
            num_samples++;

            // Log to wandb
            if (use_wandb) {
                int global_step = epoch * num_batches + batch_idx;
                wandb_logger.set_step(global_step);

                std::map<std::string, double> metrics;
                metrics["train/loss"] = loss;
                metrics["train/learning_rate"] = learning_rate;
                metrics["train/epoch"] = epoch + 1;
                wandb_logger.log_metrics(metrics);
            }

            // Print progress
            if ((batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1) {
                printf("\r   Epoch %d/%d, Batch %d/%d, Loss: %.4f",
                       epoch + 1, num_epochs, batch_idx + 1, num_batches, loss);
                fflush(stdout);
            }
        }

        float avg_loss = total_loss / num_samples;
        printf("\n   Epoch %d complete - Average loss: %.4f\n\n", epoch + 1, avg_loss);

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
