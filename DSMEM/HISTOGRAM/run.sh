#!/bin/bash

set -e

echo "Building histogram H100 program..."
make clean
make histogram_h100
# Detect GPU architecture
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)

nvidia-smi -pm 1 # Enable persistent mode
nvidia-smi -ac 1600,1755 

if [[ "$GPU_INFO" == *"H100"* ]]; then
    echo "Detected H100 GPU"
    make histogram_h100
    
    echo "Format: bin_size cluster_size latency_ms"
    echo ""

    # H100 has 256KB shared memory per SM (~64K ints)
    # Test bin sizes that exceed this limit
    bin_sizes=(64 256 16384 32768 65536 131072 262144 524288)  # 16K to 512K bins ~ 8 SM shared mem needed
    cluster_sizes=(0 1 2 4 8)  # Smaller clusters to utilize all 132 SMs

    for bins in "${bin_sizes[@]}"; do
        echo "# Testing bin_size: $bins elements ($(($bins * 4 / 1024)) KB)"
        for cluster in "${cluster_sizes[@]}"; do
            ./histogram_h100 $bins $cluster
        done
        echo ""
    done

    echo "Benchmark complete!" 

    echo ""
    echo "===========================================" 
    echo "Running histogram256 optimized version..."
    echo "==========================================="
    
    # Build and run the histogram256 optimized version
    make histogram256_optimize_host
    echo ""
    echo "Running histogram256 with 256 bins on 1M elements:"
    ./histogram256_optimize_host
    
    echo ""
    echo "histogram256 benchmark complete!"
    
    echo ""
    echo "=================================================" 
    echo "Running histogram256 DSMEM optimized version..."
    echo "================================================="
    
    # Build and run the histogram256 DSMEM optimized version
    make histogram256_optimize_dsmem
    echo ""
    echo "Running histogram256 DSMEM with 16 clusters (size 8) on 1M elements:"
    ./histogram256_optimize_dsmem
    
    echo ""
    echo "histogram256 DSMEM optimization benchmark complete!"
    
else
    echo "No Hopper GPU Architecture detected"
fi

