cmake_minimum_required(VERSION 3.18)
project(MLP_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# Library: MLP CUDA
add_library(mlp_cuda STATIC
    src/matrix_ops.cu
    src/activations.cu
    src/loss.cu
    src/adam.cu
    src/mlp.cu
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

# Examples
add_executable(train_regression examples/train_regression.cu)
target_link_libraries(train_regression mlp_cuda)

# Installation
install(TARGETS mlp_cuda DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
