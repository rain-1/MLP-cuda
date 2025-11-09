#ifndef TEXT_DATASET_H
#define TEXT_DATASET_H

#include "tokenizer.h"
#include <vector>
#include <string>
#include <random>

// Simple text dataset for language modeling
// Creates batches of (input, target) sequences where target is input shifted by 1
class TextDataset {
public:
    TextDataset(
        Tokenizer& tokenizer,
        int seq_len,
        int batch_size
    );

    // Load text from file
    void load_from_file(const char* filename);

    // Load text from string
    void load_from_text(const std::string& text);

    // Get number of batches
    int get_num_batches() const { return num_batches; }

    // Get a batch (returns false if no more batches)
    // inputs: [batch_size, seq_len]
    // targets: [batch_size, seq_len] - targets[i, j] = inputs[i, j+1]
    bool get_batch(int batch_idx, std::vector<int>& inputs, std::vector<int>& targets);

    // Shuffle the dataset
    void shuffle(unsigned int seed = 0);

    // Reset iteration
    void reset() { current_batch = 0; }

private:
    Tokenizer& tokenizer;
    int seq_len;
    int batch_size;

    std::vector<int> tokens;  // All tokens from the dataset
    int num_batches;
    int current_batch;

    std::mt19937 rng;
};

#endif // TEXT_DATASET_H
