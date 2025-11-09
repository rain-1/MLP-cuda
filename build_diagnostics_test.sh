#!/bin/bash

# Build the diagnostics test without CUDA dependencies
# This tests the CPU-side diagnostic infrastructure

echo "Building diagnostics test suite..."

g++ -std=c++17 -O2 -Wall \
    tests/test_diagnostics.cpp \
    src/training_diagnostics.cpp \
    -I./include \
    -o test_diagnostics

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Run with: ./test_diagnostics"
    ls -lh test_diagnostics
else
    echo "✗ Build failed!"
    exit 1
fi
