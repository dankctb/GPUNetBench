#!/bin/bash

set -e

echo "Building histogram H100 program..."
make clean
make histogram_h100
# Detect GPU architecture
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)

if [[ "$GPU_INFO" == *"H100"* ]]; then
    echo "Detected H100 GPU"
    make histogram_h100
    
    echo "Format: bin_size cluster_size latency_ms"
    echo ""

    # H100 has 256KB shared memory per SM (~64K ints)
    # Test bin sizes that exceed this limit
    bin_sizes=(16384 32768 65536 131072 262144 524288)  # 16K to 512K bins
    cluster_sizes=(0 2 4 8 16)  # 0 = no DSM, others = DSM cluster sizes

    for bins in "${bin_sizes[@]}"; do
        echo "# Testing bin_size: $bins ($(($bins * 4 / 1024)) KB)"
        for cluster in "${cluster_sizes[@]}"; do
            ./histogram_h100 $bins $cluster
        done
        echo ""
    done

    echo "Benchmark complete!" 

    
else
    echo "No Hopper GPU Architecture detected"
fi

