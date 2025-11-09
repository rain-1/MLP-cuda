#include "transformer.h"
#include "tokenizer.h"
#include "word_tokenizer.h"
#include "text_dataset.h"
#include "loss.h"
#include "wandb_logger.h"
#include "train_transformer_impl.h"
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
    bool use_word_tokenizer = false;
    int word_vocab_size = 2000;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--wandb") == 0) {
            use_wandb = true;
        } else if (strcmp(argv[i], "--wandb-project") == 0 && i + 1 < argc) {
            wandb_project = argv[++i];
        } else if (strcmp(argv[i], "--wandb-run") == 0 && i + 1 < argc) {
            wandb_run = argv[++i];
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            text_file = argv[++i];
        } else if (strcmp(argv[i], "--word-tokenizer") == 0) {
            use_word_tokenizer = true;
        } else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) {
            word_vocab_size = atoi(argv[++i]);
        } else {
            // Backward compatibility: first positional arg is data file
            if (text_file == nullptr) {
                text_file = argv[i];
            }
        }
    }

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

    // 1. Build tokenizer and run training
    printf("1. Building tokenizer...\n");

    if (use_word_tokenizer) {
        printf("   Using WORD-level tokenizer (max vocab_size=%d)\n", word_vocab_size);
        WordTokenizer tokenizer(word_vocab_size);

        if (text_file != nullptr) {
            tokenizer.build_vocab_from_file(text_file);
        } else {
            tokenizer.build_vocab(training_text);
        }

        int vocab_size = tokenizer.vocab_size();
        printf("   Vocabulary size: %d\n", vocab_size);
        printf("   Special tokens: PAD=%d, BOS=%d, EOS=%d, UNK=%d\n\n",
               tokenizer.pad_token(), tokenizer.bos_token(), tokenizer.eos_token(), tokenizer.unk_token());

        return run_training(tokenizer, text_file, training_text,
                          use_wandb, wandb_project, wandb_run);
    } else {
        printf("   Using CHARACTER-level tokenizer\n");
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

        return run_training(tokenizer, text_file, training_text,
                          use_wandb, wandb_project, wandb_run);
    }
}
