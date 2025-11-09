#include "word_tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>

void analyze_text_file(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();

    // Count basic statistics
    size_t total_chars = text.size();
    size_t total_lines = std::count(text.begin(), text.end(), '\n');
    size_t total_eos = 0;

    // Count <|endoftext|> markers
    const std::string eos_marker = "<|endoftext|>";
    size_t pos = 0;
    while ((pos = text.find(eos_marker, pos)) != std::string::npos) {
        total_eos++;
        pos += eos_marker.length();
    }

    printf("\n==========================================================\n");
    printf("         Vocabulary Analysis for: %s\n", filename);
    printf("==========================================================\n\n");

    printf("File Statistics:\n");
    printf("  - Total characters: %zu\n", total_chars);
    printf("  - Total lines: %zu\n", total_lines);
    printf("  - Documents (EOS markers): %zu\n", total_eos);
    printf("\n");

    // Test different vocabulary sizes
    std::vector<int> vocab_sizes = {500, 1000, 2000, 5000, 10000};

    for (int vocab_size : vocab_sizes) {
        WordTokenizer tokenizer(vocab_size);
        tokenizer.build_vocab(text);

        std::vector<int> tokens = tokenizer.encode(text);

        printf("Vocabulary size: %d\n", vocab_size);
        printf("  - Tokens in vocabulary: %d\n", tokenizer.get_vocab_size());
        printf("  - Total tokens encoded: %zu\n", tokens.size());
        printf("  - Compression ratio: %.2fx (chars/tokens)\n",
               (float)total_chars / tokens.size());
        printf("  - Average token length: %.2f chars\n",
               (float)total_chars / tokens.size());
        printf("\n");
    }

    // Show detailed stats for recommended size
    int recommended_size = 2000;
    printf("==========================================================\n");
    printf("Detailed analysis with vocabulary size: %d\n", recommended_size);
    printf("==========================================================\n");

    WordTokenizer tokenizer(recommended_size);
    tokenizer.build_vocab(text);
    tokenizer.print_vocab_stats();

    // Sample encoding/decoding
    std::string sample = "once upon a time, there was a little girl<|endoftext|>";
    printf("Sample encoding/decoding:\n");
    printf("  Original: \"%s\"\n", sample.c_str());

    std::vector<int> encoded = tokenizer.encode(sample);
    printf("  Encoded:  [");
    for (size_t i = 0; i < encoded.size(); i++) {
        printf("%d", encoded[i]);
        if (i < encoded.size() - 1) printf(", ");
    }
    printf("]\n");

    std::string decoded = tokenizer.decode(encoded);
    printf("  Decoded:  \"%s\"\n", decoded.c_str());
    printf("\n");

    // Recommendations
    printf("==========================================================\n");
    printf("Recommendations:\n");
    printf("==========================================================\n");

    if (total_chars < 10000) {
        printf("  - Small dataset: Use vocab_size=500-1000\n");
    } else if (total_chars < 100000) {
        printf("  - Medium dataset: Use vocab_size=1000-2000\n");
    } else {
        printf("  - Large dataset: Use vocab_size=2000-5000\n");
    }

    printf("  - For tinystories (varied vocabulary): vocab_size=2000-3000\n");
    printf("  - For specific domain (limited vocabulary): vocab_size=500-1000\n");
    printf("\n");

    printf("Usage:\n");
    printf("  ./train_transformer --data %s --vocab-size 2000 --word-tokenizer\n", filename);
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <text_file>\n", argv[0]);
        printf("\n");
        printf("Analyzes a text file to help determine optimal tokenizer configuration.\n");
        printf("\n");
        printf("Example:\n");
        printf("  %s data/tinystories.txt\n", argv[0]);
        return 1;
    }

    analyze_text_file(argv[1]);

    return 0;
}
