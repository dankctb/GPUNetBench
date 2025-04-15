# GPU Microbenchmarks Repository Overview

This repository contains CUDA-based benchmarks designed to evaluate various aspects of GPU memory and interconnection network on NVIDIA GPUs (V100, A100, H100). Each benchmark focuses on distinct architectural components, using as metrics bandwidth, latency and execution time.

## Repository Structure

```
GPU_Benchmarks_Repo
├── AGGREGATE
├── BISECTION
├── HBM_LAT-BW
├── L2_LAT-BW
├── MP
├── PER_SLICE
├── SM2SM
│   ├── ALL2ALL
│   ├── ONE2ONE
│   └── SM2SM+L2
├── SMEM_LAT-BW
└── SPEEDUP
```

## Benchmarks Overview

- **AGGREGATE:**
  - Measures aggregate read bandwidth from L2 cache and High-Bandwidth Memory (HBM) under various injection rates by modifying number of CTAs per SM and number of threads per CTA.

- **HBM_LAT-BW:**
  - Evaluates latency and throughput characteristics of HBM accesses under different injection rates, access patterns, and delayed injection to investigate interconnect characteristics.

- **L2_LAT-BW:**
  - Evaluates latency and throughput characteristics of L2 accesses under different injection rates, access patterns, and delayed injection to investigate interconnect characteristics.

- **MP:**
  - Performs non-coalesced memory accesses to specific (and multiple) L2 cache slices. Allows selection of Graphics Processing Clusters (GPCs) and memory partitions. We also provide the possibility to choose distributed SMs and L2 slices.

- **PER_SLICE:**
  - Targets individual L2 cache slices by a selected SM or GPC in the GPU. We evaluate for read operation and non-coalesced access pattern. The purpouse is to evaluate if the NoC provides uniform bandwidth across different sources (SMs) and destinations (L2 slices), despites a non-uniform zero load latency.

- **SM2SM:**
  - Benchmarks SM-to-SM network by using Distributed Shared Memory (DSM) and Thread-Block Cluster features introduced in Hopper Architecture for inter-SM communication within a GPC:
    - **ALL2ALL:** Evaluates bandwidth and latency for various traffic patterns using all the SMs in the GPC.
    - **ONE2ONE:** Measures bandwidth and latency across all pairs of source and destination SMs.
    - **SM2SM+L2:** Assesses interference between DSM and L2 cache traffics.

- **SMEM_LAT-BW:**
  - Benchmarks local shared memory latency and bandwidth with varying injection rates and access patterns to analyze bank conflicts and performance of shared memory.

- **SPEEDUP:**
  - Evaluates input speedup to the GPU NoC by selectively activating SMs within GPC, CPC or TPC hierarchies (evaluating the bandwith to L2 cache with streaming access).
  - Supports both read and write operations and includes profiling capabilities for multiple GPU architectures.

- **BISECTION:**
  - Measures inter-partition bisection bandwidth targeting the partitioned L2 cache architecture of NVIDIA Ampere and Hopper GPUs.
  - Utilizes SMs on one side of the chip to access remote L2 partition and measure the L2-to-L2 interconnection bisection bandwidth.

## General Requirements

- **CUDA Toolkit** with `nvcc` compiler
- Compatible NVIDIA GPUs (V100, A100, H100)
- Python 3 with libraries (`numpy`, `matplotlib`, `scipy`, `pandas`, `argparse`)
- Bash shell for execution scripts
- Make
- NVIDIA profiling tools (`nvprof`, `ncu`) for performance analysis

