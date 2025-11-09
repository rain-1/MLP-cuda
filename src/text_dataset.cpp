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
    std::vector<int> all_tokens = tokenizer.encode(text);

    // Split into documents at EOS tokens and create sequences
    // that don't cross document boundaries
    tokens.clear();

    int eos_id = tokenizer.eos_token();
    std::vector<int> current_doc;
    int total_sequences = 0;

    auto process_document = [&](const std::vector<int>& doc) {
        // For each document, extract all possible sequences of length (seq_len + 1)
        if (doc.size() >= seq_len + 1) {
            int num_seqs = doc.size() - seq_len;
            for (int i = 0; i < num_seqs; i++) {
                // Add this sequence to our token pool
                for (int j = 0; j < seq_len + 1; j++) {
                    tokens.push_back(doc[i + j]);
                }
                total_sequences++;
            }
        }
    };

    // Process tokens, splitting at EOS boundaries
    for (size_t i = 0; i < all_tokens.size(); i++) {
        if (all_tokens[i] == eos_id) {
            // Add EOS to current document
            current_doc.push_back(all_tokens[i]);

            // Process this document
            process_document(current_doc);

            // Start new document
            current_doc.clear();
        } else {
            current_doc.push_back(all_tokens[i]);
        }
    }

    // Process final document if it exists
    if (!current_doc.empty()) {
        process_document(current_doc);
    }

    // Calculate number of complete batches
    num_batches = total_sequences / batch_size;

    // Truncate to fit complete batches
    int total_tokens_needed = num_batches * batch_size * (seq_len + 1);
    if (tokens.size() > total_tokens_needed) {
        tokens.resize(total_tokens_needed);
    }

    printf("Dataset loaded: %d tokens, %d sequences, %d batches of %d sequences (seq_len=%d)\n",
           (int)all_tokens.size(), total_sequences, num_batches, batch_size, seq_len);
    printf("   (Sequences respect document boundaries at EOS tokens)\n");
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
