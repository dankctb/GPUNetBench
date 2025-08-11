#!/bin/bash

set -e

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

    ./histogram_h100 
    
else
    echo "No GPU Architecture detected"
fi

echo "Benchmark complete!" 