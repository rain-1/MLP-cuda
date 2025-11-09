#include "transformer.h"
#include "tokenizer.h"
#include <stdio.h>
#include <string>
#include <vector>

int main() {
    printf("=== Transformer Demo ===\n\n");

    // Sample training text
    const char* training_text =
        "once upon a time there was a little girl. "
        "she lived in a small house in the woods. "
        "one day she went for a walk. "
        "the sun was shining and the birds were singing. "
        "she was very happy.";

    // 1. Build tokenizer
    printf("1. Building tokenizer...\n");
    Tokenizer tokenizer;
    tokenizer.build_vocab(training_text);

    int vocab_size = tokenizer.vocab_size();
    printf("   Vocabulary size: %d\n", vocab_size);
    printf("   Special tokens: PAD=%d, BOS=%d, EOS=%d\n\n",
           tokenizer.pad_token(), tokenizer.bos_token(), tokenizer.eos_token());

    // 2. Create a small transformer
    printf("2. Creating transformer model...\n");
    int d_model = 128;
    int num_layers = 4;
    int num_heads = 4;
    int d_ff = 512;
    int max_seq_len = 64;
    int max_batch_size = 8;

    Transformer model(vocab_size, d_model, num_layers, num_heads,
                     d_ff, max_seq_len, max_batch_size);

    printf("   Model parameters:\n");
    printf("   - d_model: %d\n", d_model);
    printf("   - num_layers: %d\n", num_layers);
    printf("   - num_heads: %d\n", num_heads);
    printf("   - d_ff: %d\n", d_ff);
    printf("   - max_seq_len: %d\n\n", max_seq_len);

    // Approximate parameter count
    int params_per_block = d_model * d_model * 4 + d_model * d_ff * 2;
    int total_params = vocab_size * d_model  // embeddings
                     + max_seq_len * d_model  // positional embeddings
                     + params_per_block * num_layers
                     + d_model * vocab_size;  // output projection
    printf("   Approximate parameters: %.2fM\n\n", total_params / 1e6);

    // 3. Test forward pass
    printf("3. Testing forward pass...\n");
    std::string test_text = "once upon a time";
    std::vector<int> tokens = tokenizer.encode(test_text);

    printf("   Input text: \"%s\"\n", test_text.c_str());
    printf("   Tokens (%zu): ", tokens.size());
    for (int t : tokens) printf("%d ", t);
    printf("\n");

    // Prepare input (add batch dimension)
    int batch_size = 1;
    int seq_len = tokens.size();
    std::vector<int> input_ids(batch_size * seq_len);
    for (int i = 0; i < seq_len; i++) {
        input_ids[i] = tokens[i];
    }

    // Forward pass
    std::vector<float> logits(batch_size * seq_len * vocab_size);
    model.forward(input_ids.data(), logits.data(), batch_size, seq_len);

    printf("   Forward pass successful!\n");
    printf("   Output shape: [%d, %d, %d]\n\n", batch_size, seq_len, vocab_size);

    // 4. Test text generation
    printf("4. Testing text generation...\n");
    std::string prompt = "once upon";
    std::vector<int> prompt_tokens = tokenizer.encode(prompt);

    printf("   Prompt: \"%s\"\n", prompt.c_str());
    printf("   Generating 20 tokens...\n\n");

    // Generate with different settings
    printf("   a) Greedy (temperature=0.1):\n");
    std::vector<int> generated = model.generate(prompt_tokens, 20, 0.1f, 0, 1.0f, 42);
    std::string generated_text = tokenizer.decode(generated);
    printf("      \"%s\"\n\n", generated_text.c_str());

    printf("   b) Sampling (temperature=0.8):\n");
    generated = model.generate(prompt_tokens, 20, 0.8f, 0, 1.0f, 123);
    generated_text = tokenizer.decode(generated);
    printf("      \"%s\"\n\n", generated_text.c_str());

    printf("   c) Top-k sampling (k=5, temperature=1.0):\n");
    generated = model.generate(prompt_tokens, 20, 1.0f, 5, 1.0f, 456);
    generated_text = tokenizer.decode(generated);
    printf("      \"%s\"\n\n", generated_text.c_str());

    printf("   d) Nucleus sampling (top_p=0.9, temperature=1.0):\n");
    generated = model.generate(prompt_tokens, 20, 1.0f, 0, 0.9f, 789);
    generated_text = tokenizer.decode(generated);
    printf("      \"%s\"\n\n", generated_text.c_str());

    // 5. Test save/load
    printf("5. Testing save/load...\n");
    const char* model_path = "/tmp/transformer_test.bin";

    model.save_parameters(model_path);
    printf("   Model saved to %s\n", model_path);

    // Create a new model and load
    Transformer model2(vocab_size, d_model, num_layers, num_heads,
                      d_ff, max_seq_len, max_batch_size);
    model2.load_parameters(model_path);
    printf("   Model loaded successfully\n");

    // Verify outputs are the same
    std::vector<float> logits2(batch_size * seq_len * vocab_size);
    model2.forward(input_ids.data(), logits2.data(), batch_size, seq_len);

    float max_diff = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        float diff = fabsf(logits[i] - logits2[i]);
        max_diff = fmaxf(max_diff, diff);
    }
    printf("   Max difference after reload: %.6e\n", max_diff);

    if (max_diff < 1e-5f) {
        printf("   ✓ Save/load verified!\n");
    } else {
        printf("   ✗ Warning: outputs differ after reload\n");
    }

    printf("\n=== Demo complete! ===\n");
    return 0;
}
