# GPU Histogram Benchmark

## Overview
High-performance histogram computation benchmark with 2048 bins, designed for V100 and H100 GPUs.

## Files
- `histogram_v100.cu` - V100 optimized kernel using shared memory
- `histogram_h100.cu` - H100 kernel with DSM and cluster cooperation  
- `main.cpp` - Driver program with data generation and verification
- `Makefile` - Build system for both architectures
- `run.sh` - Automated build and execution script

## Features
- **Large bins**: 2048 bins (>1024 requirement)
- **L2 cache optimized**: 1M elements (~4MB) fits in GPU L2 cache
- **V100**: Traditional shared memory approach
- **H100**: Distributed Shared Memory (DSM) with single GPC, varied cluster sizes (2,4,8)

## Usage

### Quick Start
```bash
chmod +x run.sh
./run.sh
```

### Manual Build & Run
```bash
# Build for V100
make histogram_v100
./histogram_v100 v100

# Build for H100  
make histogram_h100
./histogram_h100 h100_c2  # cluster size 2
./histogram_h100 h100_c4  # cluster size 4
./histogram_h100 h100_c8  # cluster size 8
```

## Architecture Details
- **V100**: Uses block-level shared memory with atomic operations
- **H100**: Leverages cooperative groups and DSM for cluster-wide reduction
- Both versions process same workload for fair comparison 