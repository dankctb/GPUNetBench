# GPU Microbenchmarks

This repository contains CUDA-based benchmarks designed to evaluate various aspects of GPU memory and interconnection network on NVIDIA GPUs (V100, A100, H100). Each benchmark focuses on distinct architectural components, using as metrics bandwidth, latency and execution time.

## Repository Structure

```
GPUbench
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

## GPUs specifications

For convenience we report some useful specifications for the GPUs used in these experiments.


| Feature                       | V100     | A100     | H100 (PCIe)|
|-------------------------------|----------|----------|------------|
| **SMs**                       | 80       | 108      | 114        |
| **TPCs**                      | 40       | 54       | 57         |
| **GPCs**                      | 6        | 7        | 7          |
| **Max SMs / GPC**             | 14       | 16       | 18         |
| **L2 cache size**             | 6 MB     | 40 MB    | 50 MB      |
| **L2 cache slices**           | 32       | 80       | 80         |
| **L2 Memory Partitions**      | 4        | 10       | 10         |
| **GPU memory bandwidth**      | 0.9 TB/s | 2 TB/s   | 2 TB/s     |
| **Memory controllers**        | 8        | 10       | 10         |
| **GPU max clock speed**       | 1.38 GHz | 1.41 GHz | 1.755 GHz  |


## Benchmarks Overview

- **AGGREGATE:**
  - Measures aggregate read bandwidth from L2 cache and High-Bandwidth Memory (HBM) in a STREAM-like way. It is possible to vary number of CTAs per SM and number of threads per CTA.

- **HBM_LAT-BW:**
  - Evaluates latency and throughput characteristics of HBM accesses under different injection rates and access patterns. It is possible to add a random delay before memory requests to schedule the injection of the memory requests into the NoC.

- **L2_LAT-BW:**
  - Evaluates latency and throughput characteristics of L2 accesses under different injection rates and access patterns. It is possible to add a random delay before memory requests to schedule the injection of the memory requests into the NoC.

- **MP:**
  - Performs non-coalesced memory accesses to specific multiple L2 cache slices. Allows selection of Graphics Processing Clusters (GPCs) to use as source and L2 memory partitions (MPs) to target as destinations. We also provide the possibility to choose SMs distributed across different GPCs and L2 slices distributed across different MPs.

- **PER_SLICE:**
  - Targets a single L2 cache slice as destination. It is possible to selected a SM or a GPC in the GPU as source. We evaluate for read operation and non-coalesced access pattern. The porpouse is to evaluate if the NoC provides uniform bandwidth across different sources (SMs) and destinations (L2 slices), despites a non-uniform zero load latency.

- **SM2SM:**
  - Benchmarks SM-to-SM network by using Distributed Shared Memory (DSM) and Thread-Block Cluster features introduced in Hopper Architecture for inter-SM communication within a GPC:
    - **ALL2ALL:** Evaluates bandwidth and latency for various traffic patterns using all the SMs in the GPC.
    - **ONE2ONE:** Measures bandwidth and latency across all pairs of source and destination SMs.
    - **SM2SM+L2:** Evaluate interference between DSM and L2 cache traffics when crossing the SM-to-SM network.

- **SMEM_LAT-BW:**
  - Benchmarks local shared memory latency and bandwidth with varying injection rates. Streaming and strided access patterns are provided. Strided accesses will result in bank conflicts. The amount of bank conflicts depend on the stride chosen.

- **SPEEDUP:**
  - Evaluates input speedup to the GPU NoC by selectively activating SMs within GPC, CPC or TPC hierarchies.
  - Supports both read and write operations.
  - The evaluations are done for streaming access and by measuring the bandwidth.

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

## Common Settings

- L1 cache is disabled with the `-dlcm=cg` compiler flag.
