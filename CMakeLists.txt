cmake_minimum_required(VERSION 3.18)
project(MLP_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA compilation flags
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# Library: MLP CUDA
add_library(mlp_cuda STATIC
    src/matrix_ops.cu
    src/activations.cu
    src/loss.cu
    src/adam.cu
    src/mlp.cu
    src/attention_ops.cu
    src/multi_head_attention.cu
    src/transformer_layers.cu
    src/transformer_block.cu
    src/transformer.cu
    src/tokenizer.cpp
    src/text_dataset.cpp
)

target_link_libraries(mlp_cuda curand)

# Tests
enable_testing()

# Test: Matrix Operations
add_executable(test_matrix_ops tests/test_matrix_ops.cu)
target_link_libraries(test_matrix_ops mlp_cuda)
add_test(NAME MatrixOps COMMAND test_matrix_ops)

# Test: MLP
add_executable(test_mlp tests/test_mlp.cu)
target_link_libraries(test_mlp mlp_cuda)
add_test(NAME MLP COMMAND test_mlp)

# Test: Attention
add_executable(test_attention tests/test_attention.cu)
target_link_libraries(test_attention mlp_cuda)
add_test(NAME Attention COMMAND test_attention)

# Examples
add_executable(train_regression examples/train_regression.cu)
target_link_libraries(train_regression mlp_cuda)

add_executable(attention_demo examples/attention_demo.cu)
target_link_libraries(attention_demo mlp_cuda)

add_executable(transformer_demo examples/transformer_demo.cu)
target_link_libraries(transformer_demo mlp_cuda)

add_executable(train_transformer examples/train_transformer.cu)
target_link_libraries(train_transformer mlp_cuda)

# Installation
install(TARGETS mlp_cuda DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
