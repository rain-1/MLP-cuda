#ifndef WORD_TOKENIZER_H
#define WORD_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>

// Word-level tokenizer with fixed vocabulary
// - Builds vocabulary from corpus based on word frequency
// - Downcases all input
// - Drops unknown words
// - Always recognizes <|endoftext|> as EOS token
class WordTokenizer {
public:
    WordTokenizer(int vocab_size = 5000);

    // Build vocabulary from text corpus
    // Scans text, counts word frequencies, keeps top vocab_size words
    void build_vocab(const std::string& text);
    void build_vocab_from_file(const char* filename);

    // Print vocabulary statistics
    void print_vocab_stats() const;

    // Encode text to token IDs
    // - Downcases input
    // - Splits into words
    // - Maps known words to IDs
    // - Drops unknown words
    // - Converts <|endoftext|> to EOS token
    std::vector<int> encode(const std::string& text);

    // Decode token IDs to text
    std::string decode(const std::vector<int>& tokens);

    // Get vocabulary size
    int get_vocab_size() const { return word_to_id.size(); }
    int vocab_size() const { return word_to_id.size(); }  // Alias for compatibility

    // Special tokens
    int pad_token() const { return pad_id; }
    int eos_token() const { return eos_id; }
    int bos_token() const { return bos_id; }
    int unk_token() const { return unk_id; }

    // Save/load vocabulary
    void save_vocab(const char* filename);
    void load_vocab(const char* filename);

private:
    int max_vocab_size;  // Maximum vocabulary size (excluding special tokens)

    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;

    int pad_id;  // Padding token = 0
    int bos_id;  // Beginning of sequence = 1
    int eos_id;  // End of sequence = 2
    int unk_id;  // Unknown word = 3

    // Tokenize text into words (split by whitespace and punctuation)
    std::vector<std::string> tokenize_words(const std::string& text);

    // Normalize word (downcase)
    std::string normalize_word(const std::string& word);

    // Check if character is punctuation
    bool is_punctuation(char c);
};

#endif // WORD_TOKENIZER_H
