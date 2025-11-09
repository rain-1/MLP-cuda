#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <set>

Tokenizer::Tokenizer() {
    // Initialize with special tokens
    // Reserve IDs 0, 1, 2 for PAD, BOS, EOS
    // Regular characters will start from ID 3
    pad_id = 0;
    bos_id = 1;
    eos_id = 2;

    // Special tokens don't map to actual characters
    // They are handled separately in encode/decode
    id_to_char[pad_id] = '\0';  // Null char for special tokens
    id_to_char[bos_id] = '\0';
    id_to_char[eos_id] = '\0';
}

std::string Tokenizer::normalize(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    for (char c : text) {
        // Convert to lowercase
        c = std::tolower(c);

        // Keep only: a-z, space, newline, .,?!,'"
        if ((c >= 'a' && c <= 'z') ||
            c == ' ' || c == '\n' ||
            c == '.' || c == ',' ||
            c == '?' || c == '!' ||
            c == '\'' || c == '"') {
            result += c;
        }
    }

    return result;
}

void Tokenizer::build_vocab(const std::string& text) {
    std::string normalized = normalize(text);

    // Count character frequencies
    std::unordered_map<char, int> char_freq;
    for (char c : normalized) {
        char_freq[c]++;
    }

    // Sort by frequency (descending)
    std::vector<std::pair<char, int>> freq_vec(char_freq.begin(), char_freq.end());
    std::sort(freq_vec.begin(), freq_vec.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Add characters to vocabulary (after special tokens)
    int next_id = 3;  // After PAD, BOS, EOS
    for (const auto& pair : freq_vec) {
        char c = pair.first;
        if (char_to_id.find(c) == char_to_id.end()) {
            char_to_id[c] = next_id;
            id_to_char[next_id] = c;
            next_id++;
        }
    }
}

void Tokenizer::build_vocab_from_file(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    build_vocab(buffer.str());

    file.close();
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    tokens.reserve(text.size());

    // Split text by <|endoftext|> markers
    const std::string eos_marker = "<|endoftext|>";
    size_t pos = 0;
    size_t found = 0;

    while ((found = text.find(eos_marker, pos)) != std::string::npos) {
        // Encode the segment before the marker
        if (found > pos) {
            std::string segment = text.substr(pos, found - pos);
            std::string normalized = normalize(segment);

            for (char c : normalized) {
                auto it = char_to_id.find(c);
                if (it != char_to_id.end()) {
                    tokens.push_back(it->second);
                }
            }
        }

        // Add EOS token
        tokens.push_back(eos_id);

        // Move past the marker
        pos = found + eos_marker.length();
    }

    // Encode the remaining text after the last marker
    if (pos < text.length()) {
        std::string segment = text.substr(pos);
        std::string normalized = normalize(segment);

        for (char c : normalized) {
            auto it = char_to_id.find(c);
            if (it != char_to_id.end()) {
                tokens.push_back(it->second);
            }
        }
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string text;
    text.reserve(tokens.size());

    for (int token_id : tokens) {
        if (token_id == pad_id || token_id == bos_id || token_id == eos_id) {
            continue;  // Skip special tokens in output
        }

        auto it = id_to_char.find(token_id);
        if (it != id_to_char.end()) {
            text += it->second;
        }
    }

    return text;
}

char Tokenizer::get_char(int token_id) const {
    auto it = id_to_char.find(token_id);
    if (it != id_to_char.end()) {
        return it->second;
    }
    return '\0';
}

void Tokenizer::save_vocab(const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }

    // Save vocab size
    file << char_to_id.size() << "\n";

    // Save mappings
    for (const auto& pair : id_to_char) {
        int id = pair.first;
        char c = pair.second;
        file << id << " " << (int)c << "\n";
    }

    file.close();
}

void Tokenizer::load_vocab(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filename);
        return;
    }

    char_to_id.clear();
    id_to_char.clear();

    int vocab_size;
    file >> vocab_size;

    for (int i = 0; i < vocab_size; i++) {
        int id, char_code;
        file >> id >> char_code;
        char c = (char)char_code;

        id_to_char[id] = c;
        char_to_id[c] = id;
    }

    file.close();

    // Restore special token IDs (these are always fixed)
    pad_id = 0;
    bos_id = 1;
    eos_id = 2;
}
