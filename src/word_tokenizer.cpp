#include "word_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <iostream>

WordTokenizer::WordTokenizer(int vocab_size)
    : max_vocab_size(vocab_size)
{
    // Initialize special tokens
    pad_id = 0;
    bos_id = 1;
    eos_id = 2;
    unk_id = 3;

    // Reserve special tokens in vocabulary
    word_to_id["<PAD>"] = pad_id;
    word_to_id["<BOS>"] = bos_id;
    word_to_id["<EOS>"] = eos_id;
    word_to_id["<UNK>"] = unk_id;

    id_to_word[pad_id] = "<PAD>";
    id_to_word[bos_id] = "<BOS>";
    id_to_word[eos_id] = "<EOS>";
    id_to_word[unk_id] = "<UNK>";
}

bool WordTokenizer::is_punctuation(char c) {
    return c == '.' || c == ',' || c == '!' || c == '?' ||
           c == ';' || c == ':' || c == '\'' || c == '"' ||
           c == '(' || c == ')' || c == '-';
}

std::string WordTokenizer::normalize_word(const std::string& word) {
    std::string result;
    for (char c : word) {
        result += std::tolower(c);
    }
    return result;
}

std::vector<std::string> WordTokenizer::tokenize_words(const std::string& text) {
    std::vector<std::string> words;
    std::string current_word;

    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];

        if (std::isspace(c)) {
            // Whitespace - end current word
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
        } else if (is_punctuation(c)) {
            // Punctuation - end current word and add punctuation as separate token
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
            words.push_back(std::string(1, c));
        } else if (std::isalpha(c)) {
            // Letter - add to current word
            current_word += c;
        }
        // Ignore other characters (numbers, special chars, etc.)
    }

    // Add final word if any
    if (!current_word.empty()) {
        words.push_back(current_word);
    }

    return words;
}

void WordTokenizer::build_vocab(const std::string& text) {
    // Tokenize and count word frequencies
    std::map<std::string, int> word_freq;

    // Split by <|endoftext|> first to process it separately
    const std::string eos_marker = "<|endoftext|>";
    size_t pos = 0;
    size_t found = 0;

    std::string text_copy = text;
    while ((found = text_copy.find(eos_marker, pos)) != std::string::npos) {
        // Process segment before marker
        if (found > pos) {
            std::string segment = text_copy.substr(pos, found - pos);
            std::vector<std::string> words = tokenize_words(segment);

            for (const auto& word : words) {
                std::string normalized = normalize_word(word);
                if (!normalized.empty()) {
                    word_freq[normalized]++;
                }
            }
        }

        pos = found + eos_marker.length();
    }

    // Process remaining text
    if (pos < text_copy.length()) {
        std::string segment = text_copy.substr(pos);
        std::vector<std::string> words = tokenize_words(segment);

        for (const auto& word : words) {
            std::string normalized = normalize_word(word);
            if (!normalized.empty()) {
                word_freq[normalized]++;
            }
        }
    }

    // Sort by frequency (descending)
    std::vector<std::pair<std::string, int>> freq_vec(word_freq.begin(), word_freq.end());
    std::sort(freq_vec.begin(), freq_vec.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Clear existing vocabulary (except special tokens)
    word_to_id.clear();
    id_to_word.clear();

    // Re-add special tokens
    word_to_id["<PAD>"] = pad_id;
    word_to_id["<BOS>"] = bos_id;
    word_to_id["<EOS>"] = eos_id;
    word_to_id["<UNK>"] = unk_id;

    id_to_word[pad_id] = "<PAD>";
    id_to_word[bos_id] = "<BOS>";
    id_to_word[eos_id] = "<EOS>";
    id_to_word[unk_id] = "<UNK>";

    // Add top N words to vocabulary
    int next_id = 4;  // After special tokens
    int words_added = 0;

    for (const auto& pair : freq_vec) {
        if (words_added >= max_vocab_size) {
            break;
        }

        const std::string& word = pair.first;

        // Skip if already in vocab (shouldn't happen, but just in case)
        if (word_to_id.find(word) != word_to_id.end()) {
            continue;
        }

        word_to_id[word] = next_id;
        id_to_word[next_id] = word;
        next_id++;
        words_added++;
    }

    printf("Vocabulary built: %d unique words found, kept top %d (+ 4 special tokens)\n",
           (int)word_freq.size(), words_added);
}

void WordTokenizer::build_vocab_from_file(const char* filename) {
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

void WordTokenizer::print_vocab_stats() const {
    printf("\n=== Word Tokenizer Vocabulary Statistics ===\n");
    printf("Total vocabulary size: %d tokens\n", get_vocab_size());
    printf("  - Special tokens: 4 (<PAD>, <BOS>, <EOS>, <UNK>)\n");
    printf("  - Word/punctuation tokens: %d\n", get_vocab_size() - 4);
    printf("\nMost common tokens (IDs 4-23):\n");

    int count = 0;
    for (int id = 4; id < std::min(24, get_vocab_size()); id++) {
        auto it = id_to_word.find(id);
        if (it != id_to_word.end()) {
            printf("  [%3d] %s\n", id, it->second.c_str());
            count++;
        }
    }
    printf("============================================\n\n");
}

std::vector<int> WordTokenizer::encode(const std::string& text) {
    std::vector<int> tokens;

    // Split by <|endoftext|> markers
    const std::string eos_marker = "<|endoftext|>";
    size_t pos = 0;
    size_t found = 0;

    while ((found = text.find(eos_marker, pos)) != std::string::npos) {
        // Encode segment before marker
        if (found > pos) {
            std::string segment = text.substr(pos, found - pos);
            std::vector<std::string> words = tokenize_words(segment);

            for (const auto& word : words) {
                std::string normalized = normalize_word(word);
                auto it = word_to_id.find(normalized);
                if (it != word_to_id.end()) {
                    tokens.push_back(it->second);
                }
                // Unknown words are dropped (not added to tokens)
            }
        }

        // Add EOS token
        tokens.push_back(eos_id);

        pos = found + eos_marker.length();
    }

    // Encode remaining text
    if (pos < text.length()) {
        std::string segment = text.substr(pos);
        std::vector<std::string> words = tokenize_words(segment);

        for (const auto& word : words) {
            std::string normalized = normalize_word(word);
            auto it = word_to_id.find(normalized);
            if (it != word_to_id.end()) {
                tokens.push_back(it->second);
            }
            // Unknown words are dropped
        }
    }

    return tokens;
}

std::string WordTokenizer::decode(const std::vector<int>& tokens) {
    std::string text;
    bool first = true;

    for (int token_id : tokens) {
        // Skip special tokens
        if (token_id == pad_id || token_id == bos_id || token_id == unk_id) {
            continue;
        }

        // Handle EOS token
        if (token_id == eos_id) {
            text += "<|endoftext|>";
            first = true;  // Reset spacing after EOS
            continue;
        }

        auto it = id_to_word.find(token_id);
        if (it != id_to_word.end()) {
            const std::string& word = it->second;

            // Add space before word (unless it's punctuation or first word)
            if (!first && !is_punctuation(word[0])) {
                text += " ";
            }

            text += word;
            first = false;
        }
    }

    return text;
}

void WordTokenizer::save_vocab(const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }

    // Save vocabulary size
    file << word_to_id.size() << "\n";

    // Save mappings
    for (const auto& pair : id_to_word) {
        int id = pair.first;
        const std::string& word = pair.second;
        file << id << " " << word << "\n";
    }

    file.close();
    printf("Vocabulary saved to %s\n", filename);
}

void WordTokenizer::load_vocab(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file for reading: %s\n", filename);
        return;
    }

    word_to_id.clear();
    id_to_word.clear();

    int vocab_size;
    file >> vocab_size;
    file.ignore();  // Skip newline

    for (int i = 0; i < vocab_size; i++) {
        int id;
        std::string word;
        file >> id;
        file.ignore();  // Skip space
        std::getline(file, word);

        id_to_word[id] = word;
        word_to_id[word] = id;
    }

    file.close();

    // Restore special token IDs
    pad_id = 0;
    bos_id = 1;
    eos_id = 2;
    unk_id = 3;

    printf("Vocabulary loaded from %s (%d tokens)\n", filename, vocab_size);
}
