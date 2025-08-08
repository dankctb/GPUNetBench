#!/bin/bash

set -e

echo "=== GPU Histogram Benchmark ==="
echo "Building histogram programs..."

# Clean previous builds
make clean

# Detect GPU architecture
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)

if [[ "$GPU_INFO" == *"V100"* ]]; then
    echo "Detected V100 GPU"
    make histogram_v100
    
    echo "Running V100 histogram benchmark..."
    ./histogram_v100 v100
    
elif [[ "$GPU_INFO" == *"H100"* ]]; then
    echo "Detected H100 GPU"
    make histogram_h100
    
    echo "Running H100 histogram benchmarks with different cluster sizes..."
    
    echo "--- Cluster Size 2 ---"
    ./histogram_h100 h100_c2
    
    echo "--- Cluster Size 4 ---"
    ./histogram_h100 h100_c4
    
    echo "--- Cluster Size 8 ---"
    ./histogram_h100 h100_c8
    
else
    echo "Building both versions for manual testing..."
    make all
    
    echo "Available executables:"
    echo "  ./histogram_v100 v100"
    echo "  ./histogram_h100 h100_c2"
    echo "  ./histogram_h100 h100_c4"
    echo "  ./histogram_h100 h100_c8"
fi

echo "Benchmark complete!" 