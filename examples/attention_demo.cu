#include "multi_head_attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void generate_positional_encoding(float* encoding, int seq_len, int d_model) {
    // Standard positional encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    //                                PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < d_model / 2; i++) {
            float angle = (float)pos / powf(10000.0f, 2.0f * i / d_model);
            encoding[pos * d_model + 2 * i] = sinf(angle);
            encoding[pos * d_model + 2 * i + 1] = cosf(angle);
        }
    }
}

void print_attention_pattern(const float* output, int seq_len, int d_model) {
    // Print a simplified view of the first few positions and dimensions
    printf("\nAttention Output (first 4 positions, first 8 dimensions):\n");
    printf("     ");
    for (int d = 0; d < 8 && d < d_model; d++) {
        printf(" %6d", d);
    }
    printf("\n");

    for (int pos = 0; pos < 4 && pos < seq_len; pos++) {
        printf("pos%d:", pos);
        for (int d = 0; d < 8 && d < d_model; d++) {
            printf(" %6.3f", output[pos * d_model + d]);
        }
        printf("\n");
    }
}

void test_self_attention() {
    printf("======================================\n");
    printf("  Self-Attention Demo\n");
    printf("======================================\n\n");

    int d_model = 64;
    int num_heads = 8;
    int seq_len = 16;
    int batch_size = 1;
    int max_seq_len = 32;
    int max_batch_size = 4;

    printf("Configuration:\n");
    printf("  d_model:    %d\n", d_model);
    printf("  num_heads:  %d\n", num_heads);
    printf("  d_k = d_v:  %d\n", d_model / num_heads);
    printf("  seq_len:    %d\n", seq_len);
    printf("  batch_size: %d\n\n", batch_size);

    // Create attention block
    printf("Creating multi-head attention block...\n");
    MultiHeadAttention mha(d_model, num_heads, max_seq_len, max_batch_size);

    // Create input with positional encoding
    printf("Generating positional encoding...\n");
    float* h_input = new float[batch_size * seq_len * d_model];
    float* h_output = new float[batch_size * seq_len * d_model];

    generate_positional_encoding(h_input, seq_len, d_model);

    printf("Running self-attention...\n");
    mha.forward(h_input, h_output, batch_size, seq_len);

    print_attention_pattern(h_output, seq_len, d_model);

    // Compute some statistics
    float mean = 0.0f, variance = 0.0f;
    for (int i = 0; i < seq_len * d_model; i++) {
        mean += h_output[i];
    }
    mean /= (seq_len * d_model);

    for (int i = 0; i < seq_len * d_model; i++) {
        float diff = h_output[i] - mean;
        variance += diff * diff;
    }
    variance /= (seq_len * d_model);

    printf("\nOutput Statistics:\n");
    printf("  Mean:     %.6f\n", mean);
    printf("  Std Dev:  %.6f\n", sqrtf(variance));
    printf("  Min:      %.6f\n", *std::min_element(h_output, h_output + seq_len * d_model));
    printf("  Max:      %.6f\n", *std::max_element(h_output, h_output + seq_len * d_model));

    delete[] h_input;
    delete[] h_output;
}

void test_cross_attention() {
    printf("\n======================================\n");
    printf("  Cross-Attention Demo\n");
    printf("======================================\n\n");

    int d_model = 64;
    int num_heads = 4;
    int seq_len_q = 8;   // Query sequence length
    int seq_len_kv = 12; // Key/Value sequence length
    int batch_size = 1;
    int max_seq_len = 32;
    int max_batch_size = 4;

    printf("Configuration:\n");
    printf("  d_model:     %d\n", d_model);
    printf("  num_heads:   %d\n", num_heads);
    printf("  seq_len_q:   %d (query)\n", seq_len_q);
    printf("  seq_len_kv:  %d (key/value)\n\n", seq_len_kv);

    printf("Creating multi-head attention block...\n");
    MultiHeadAttention mha(d_model, num_heads, max_seq_len, max_batch_size);

    // Create inputs
    float* h_Q = new float[batch_size * seq_len_q * d_model];
    float* h_KV = new float[batch_size * seq_len_kv * d_model];
    float* h_output = new float[batch_size * seq_len_q * d_model];

    // Generate different patterns for Q and KV
    printf("Generating query and key/value sequences...\n");
    generate_positional_encoding(h_Q, seq_len_q, d_model);
    generate_positional_encoding(h_KV, seq_len_kv, d_model);

    printf("Running cross-attention...\n");
    mha.forward_cross(h_Q, h_KV, h_output, batch_size, seq_len_q, seq_len_kv);

    print_attention_pattern(h_output, seq_len_q, d_model);

    printf("\nCross-attention complete!\n");
    printf("Output shape: [%d, %d, %d]\n", batch_size, seq_len_q, d_model);

    delete[] h_Q;
    delete[] h_KV;
    delete[] h_output;
}

void test_save_load() {
    printf("\n======================================\n");
    printf("  Parameter Save/Load Demo\n");
    printf("======================================\n\n");

    int d_model = 32;
    int num_heads = 4;
    int seq_len = 8;
    int batch_size = 1;
    int max_seq_len = 16;
    int max_batch_size = 4;

    // Create and use first attention block
    printf("Creating first attention block...\n");
    MultiHeadAttention mha1(d_model, num_heads, max_seq_len, max_batch_size);

    float* h_input = new float[batch_size * seq_len * d_model];
    float* h_output1 = new float[batch_size * seq_len * d_model];
    float* h_output2 = new float[batch_size * seq_len * d_model];

    generate_positional_encoding(h_input, seq_len, d_model);

    printf("Running forward pass with original parameters...\n");
    mha1.forward(h_input, h_output1, batch_size, seq_len);

    printf("Saving parameters to 'attention_params.bin'...\n");
    mha1.save_parameters("attention_params.bin");

    // Create second attention block and load parameters
    printf("\nCreating second attention block...\n");
    MultiHeadAttention mha2(d_model, num_heads, max_seq_len, max_batch_size);

    printf("Loading parameters from 'attention_params.bin'...\n");
    mha2.load_parameters("attention_params.bin");

    printf("Running forward pass with loaded parameters...\n");
    mha2.forward(h_input, h_output2, batch_size, seq_len);

    // Compare outputs
    float max_diff = 0.0f;
    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        float diff = fabsf(h_output1[i] - h_output2[i]);
        max_diff = fmaxf(max_diff, diff);
    }

    printf("\nVerification:\n");
    printf("  Maximum difference: %.6e\n", max_diff);
    if (max_diff < 1e-5f) {
        printf("  ✓ Parameters saved and loaded successfully!\n");
    } else {
        printf("  ✗ Outputs differ - possible issue with save/load\n");
    }

    delete[] h_input;
    delete[] h_output1;
    delete[] h_output2;
}

int main() {
    printf("========================================\n");
    printf("  Multi-Head Attention Demonstration\n");
    printf("========================================\n\n");

    printf("This demo showcases the multi-head attention\n");
    printf("implementation with various configurations.\n\n");

    // Run demonstrations
    test_self_attention();
    test_cross_attention();
    test_save_load();

    printf("\n========================================\n");
    printf("  Demo Complete!\n");
    printf("========================================\n");

    return 0;
}
