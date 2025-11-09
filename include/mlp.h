#ifndef MLP_H
#define MLP_H

#include <cuda_runtime.h>

class MLP {
public:
    // Constructor
    // layer_sizes: array of 4 integers [h1, h2, h3, h4]
    // batch_size: maximum batch size for this MLP
    // learning_rate: learning rate for Adam optimizer (default: 0.001)
    // beta1: Adam beta1 parameter (default: 0.9)
    // beta2: Adam beta2 parameter (default: 0.999)
    // epsilon: Adam epsilon parameter (default: 1e-8)
    MLP(int layer_sizes[4], int batch_size,
        float learning_rate = 0.001f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f);

    // Destructor
    ~MLP();

    // Initialize parameters with Xavier/He initialization
    void initialize_parameters();

    // Forward pass (inference)
    // h_X: host input batch [B x h1]
    // h_output: host output buffer [B x h4]
    // batch_size: actual batch size (must be <= max batch_size from constructor)
    void forward(const float* h_X, float* h_output, int batch_size);

    // Training step (forward + backward + update)
    // h_X: host input batch [B x h1]
    // h_Y: host target batch [B x h4]
    // batch_size: actual batch size
    // Returns: loss value
    float train_step(const float* h_X, const float* h_Y, int batch_size);

    // Get loss on a batch (forward + loss computation)
    float evaluate(const float* h_X, const float* h_Y, int batch_size);

    // Save/load parameters
    void save_parameters(const char* filename);
    void load_parameters(const char* filename);

    // Getters
    int get_h1() const { return h1; }
    int get_h2() const { return h2; }
    int get_h3() const { return h3; }
    int get_h4() const { return h4; }
    int get_max_batch_size() const { return max_batch_size; }
    int get_step_count() const { return step_count; }

private:
    // Network architecture
    int h1, h2, h3, h4;
    int max_batch_size;

    // Adam hyperparameters
    float lr, beta1, beta2, epsilon;
    int step_count;

    // Device pointers - Layer 1
    float *d_W1, *d_b1;        // Parameters
    float *d_Z1, *d_A1;        // Activations
    float *d_dW1, *d_db1;      // Gradients
    float *d_dZ1, *d_dA1;
    float *d_m_W1, *d_v_W1;    // Adam moments
    float *d_m_b1, *d_v_b1;

    // Device pointers - Layer 2
    float *d_W2, *d_b2;
    float *d_Z2, *d_A2;
    float *d_dW2, *d_db2;
    float *d_dZ2, *d_dA2;
    float *d_m_W2, *d_v_W2;
    float *d_m_b2, *d_v_b2;

    // Device pointers - Layer 3 (output)
    float *d_W3, *d_b3;
    float *d_Z3;               // Output (no activation)
    float *d_dW3, *d_db3;
    float *d_dZ3;
    float *d_m_W3, *d_v_W3;
    float *d_m_b3, *d_v_b3;

    // Device pointers - Input/Output buffers
    float *d_X;                // Input batch
    float *d_Y;                // Target batch

    // Temporary buffers for intermediate computations
    float *d_temp1, *d_temp2;

    // Helper methods
    void allocate_memory();
    void free_memory();
    void forward_device(int batch_size);
    void backward_device(int batch_size);
    void update_parameters();
};

#endif // MLP_H
