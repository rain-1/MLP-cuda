#include "text_dataset.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdio.h>

TextDataset::TextDataset(
    Tokenizer& tokenizer,
    int seq_len,
    int batch_size
) : tokenizer(tokenizer), seq_len(seq_len), batch_size(batch_size),
    num_batches(0), current_batch(0), rng(0)
{
}

void TextDataset::load_from_file(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    load_from_text(buffer.str());

    file.close();
}

void TextDataset::load_from_text(const std::string& text) {
    // Encode the text
    tokens = tokenizer.encode(text);

    // Calculate number of complete batches
    // Each sequence needs seq_len + 1 tokens (seq_len for input, 1 for target)
    int tokens_per_batch = batch_size * (seq_len + 1);
    num_batches = tokens.size() / tokens_per_batch;

    // Truncate to fit complete batches
    int total_tokens_needed = num_batches * tokens_per_batch;
    if (tokens.size() > total_tokens_needed) {
        tokens.resize(total_tokens_needed);
    }

    printf("Dataset loaded: %d tokens, %d batches of %d sequences (seq_len=%d)\n",
           (int)tokens.size(), num_batches, batch_size, seq_len);
}

bool TextDataset::get_batch(int batch_idx, std::vector<int>& inputs, std::vector<int>& targets) {
    if (batch_idx >= num_batches) {
        return false;
    }

    // Allocate output
    inputs.resize(batch_size * seq_len);
    targets.resize(batch_size * seq_len);

    // Fill the batch
    int tokens_per_batch = batch_size * (seq_len + 1);
    int start_idx = batch_idx * tokens_per_batch;

    for (int b = 0; b < batch_size; b++) {
        int seq_start = start_idx + b * (seq_len + 1);

        for (int i = 0; i < seq_len; i++) {
            inputs[b * seq_len + i] = tokens[seq_start + i];
            targets[b * seq_len + i] = tokens[seq_start + i + 1];
        }
    }

    return true;
}

void TextDataset::shuffle(unsigned int seed) {
    // For simplicity, we'll shuffle at the sequence level
    // More sophisticated shuffling could be done at the batch level

    if (seed != 0) {
        rng.seed(seed);
    }

    // Create sequence indices
    int num_sequences = num_batches * batch_size;
    std::vector<int> seq_indices(num_sequences);
    for (int i = 0; i < num_sequences; i++) {
        seq_indices[i] = i;
    }

    // Shuffle indices
    std::shuffle(seq_indices.begin(), seq_indices.end(), rng);

    // Create new token array with shuffled sequences
    std::vector<int> shuffled_tokens;
    shuffled_tokens.reserve(tokens.size());

    for (int idx : seq_indices) {
        int start = idx * (seq_len + 1);
        for (int i = 0; i < seq_len + 1; i++) {
            shuffled_tokens.push_back(tokens[start + i]);
        }
    }

    tokens = shuffled_tokens;
}
