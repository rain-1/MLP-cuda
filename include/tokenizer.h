#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

// Simple character-level tokenizer
class Tokenizer {
public:
    Tokenizer();

    // Build vocabulary from text
    void build_vocab(const std::string& text);
    void build_vocab_from_file(const char* filename);

    // Encode text to token IDs
    std::vector<int> encode(const std::string& text);

    // Decode token IDs to text
    std::string decode(const std::vector<int>& tokens);

    // Get vocabulary size
    int vocab_size() const { return char_to_id.size(); }

    // Special tokens
    int pad_token() const { return pad_id; }
    int eos_token() const { return eos_id; }
    int bos_token() const { return bos_id; }

    // Save/load vocabulary
    void save_vocab(const char* filename);
    void load_vocab(const char* filename);

    // Get character for token ID
    char get_char(int token_id) const;

private:
    std::unordered_map<char, int> char_to_id;
    std::unordered_map<int, char> id_to_char;

    int pad_id;  // Padding token
    int eos_id;  // End of sequence
    int bos_id;  // Beginning of sequence

    // Normalize text (lowercase, filter characters)
    std::string normalize(const std::string& text);
};

#endif // TOKENIZER_H
