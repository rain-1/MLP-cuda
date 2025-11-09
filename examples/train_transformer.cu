#include "transformer.h"
#include "tokenizer.h"
#include "text_dataset.h"
#include "loss.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <time.h>

int main(int argc, char** argv) {
    printf("=== Transformer Training Demo ===\n\n");

    // Training configuration
    const char* text_file = (argc > 1) ? argv[1] : nullptr;

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

            // Compute loss (forward pass only - no backward yet)
            float loss = model.compute_loss(inputs.data(), targets.data(),
                                           batch_size, seq_len);

            total_loss += loss;
            num_samples++;

            // Print progress
            if ((batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1) {
                printf("\r   Epoch %d/%d, Batch %d/%d, Loss: %.4f",
                       epoch + 1, num_epochs, batch_idx + 1, num_batches, loss);
                fflush(stdout);
            }
        }

        float avg_loss = total_loss / num_samples;
        printf("\n   Epoch %d complete - Average loss: %.4f\n\n", epoch + 1, avg_loss);

        // Generate sample text every few epochs
        if ((epoch + 1) % 2 == 0) {
            printf("   Sample generation:\n");
            std::string prompt = "once upon";
            std::vector<int> prompt_tokens = tokenizer.encode(prompt);
            std::vector<int> generated = model.generate(prompt_tokens, 30, 0.8f, 0, 1.0f, epoch);
            std::string generated_text = tokenizer.decode(generated);
            printf("   \"%s\"\n\n", generated_text.c_str());
        }
    }

    time_t end_time = time(nullptr);
    int elapsed = (int)(end_time - start_time);
    printf("Training complete in %d seconds\n\n", elapsed);

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
    }

    printf("=== Training demo complete! ===\n");

    printf("\nNOTE: This is a forward-pass only demo. The model is not actually learning\n");
    printf("because backward passes are not yet implemented. Loss values show the\n");
    printf("performance of the randomly initialized model.\n");

    return 0;
}
