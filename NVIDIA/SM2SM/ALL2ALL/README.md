# SM2SM Benchmark for Various Traffic Patterns

## Overview

This benchmark targets the **Distributed Shared Memory (DSM)** feature of the NVIDIA Hopper architecture. It measures the performance of different traffic patterns between Streaming Multiprocessors (SMs) within a Graphics Processing Cluster (GPC) using the SM-to-SM network.

In this benchmark, each source block reads data from one or more destination blocks within the same cluster. The supported traffic patterns include:

- **TRAFFIC_RANDPERM**: Each block reads from a fixed random partner.
- **TRAFFIC_ROUNDROBIN**: Each block reads from the next block in the cluster.
- **TRAFFIC_UNIFORM**: Each block reads from a different random partner on each iteration.
- **TRAFFIC_ALL2ALL**: Each block reads from all other blocks in the cluster.
  - With the additional compile‑time flag `READ_ALL2ALL_FULL`, the ALL2ALL pattern can be configured to read the full shared memory size of each destination block instead of a reduced fraction.

The benchmark can measure two performance metrics:
- **Overall Bandwidth** (`CALC_BW`): Reports the aggregate cycles for completing the memory transfers.
- **Per-Request Latency** (`CALC_LATENCY`): Reports the average latency in clock cycles per request.

Other configurable parameters include:
- **STRIDE**: Controls the element stride between threads. A default of 1 indicates a streaming access, while higher values simulate bank-conflict scenarios.
- **ILP_FACTOR**: Determines the degree of instruction-level parallelism (loop unrolling).
- **ITERATION**: The number of times the read loop is executed.
- **CLUSTER_SIZE**: Number of blocks per cluster (default is 16).
- **BLOCK_SIZE**: Threads per block (default is 1024).

## Prerequisites

- **CUDA Toolkit** with `nvcc` (tested with compute capability `sm_90`)
- A Unix-like environment with a **bash** shell

## File Structure

```
.
├── main.cu          # CUDA kernel and host driver for the SM2SM benchmark
├── Makefile         # Build configuration (uses NVCC_DEFS for compile‑time flags)
├── run.sh           # Script to compile and execute the benchmark for multiple configurations
└── README.md        # Documentation for this project
```

## Compilation

The project uses a simple Makefile controlled by the environment variable `NVCC_DEFS` for compile‑time definitions. For example, to compile with a given ILP factor and measurement mode, you can run:

```bash
# For Bandwidth measurement (CALC_BW)
make clean
make NVCC_DEFS="-DILP_FACTOR=8 -DCALC_BW"

# For Latency measurement (CALC_LATENCY)
make clean
make NVCC_DEFS="-DILP_FACTOR=8 -DCALC_LATENCY"
```

The provided **Makefile** compiles the `main.cu` file into the executable `SMEM`.

## Running the Experiment

The **run.sh** script automates both the compilation and execution of the benchmark for multiple configurations. It allows you to easily select:

- **Traffic Pattern**: Options include `randperm`, `roundrobin`, `uniform`, `all2all`, and `all2all_full` (ALL2ALL with full shared memory read).
- **Measurement Mode**: Either latency or bandwidth.
- **Block Sizes**: The script loops over block sizes: **32, 64, 128, 256, 512, 1024**.

**Usage:**

```bash
chmod +x run.sh
./run.sh
```

By default, **run.sh** sets:
- `CLUSTER_SIZE = 16`
- `NUM_CLUSTERS = 1`
- `ILP_FACTOR = 8`

The executable now accepts **two runtime parameters**:
1. `<numClusters>` – number of clusters (default is 1)
2. `<blockSize>`   – threads per block

Each run produces a log file named with the measurement mode, traffic pattern, block size, number of clusters, and ILP factor. For example:

- `bw_randperm_1024_1_8.log`
- `latency_all2all_full_256_1_8.log`

These log files contain the benchmark results for each configuration.

## Experiment Settings and Customization

- **Measurement Modes**:
  - **Bandwidth**: Uses `-DCALC_BW` and reports overall bandwidth (GB/s).
  - **Latency**: Uses `-DCALC_LATENCY` and reports average latency (clock cycles).
- **Traffic Pattern Selection**:
  - Use the corresponding flag (e.g., `-DTRAFFIC_RANDPERM`, `-DTRAFFIC_ROUNDROBIN`, `-DTRAFFIC_UNIFORM`, `-DTRAFFIC_ALL2ALL`) in combination with other compile‑time options.
  - For the ALL2ALL pattern to read the full shared memory, add the flag `-DREAD_ALL2ALL_FULL`.
- **Block Size Loop**:
  - The **run.sh** script loops over block sizes (`32, 64, 128, 256, 512, 1024`), allowing you to observe the effects of varying threads per block.

To adjust any of these settings—such as changing the ILP factor, STRIDE, or iterations—modify the corresponding definitions in **main.cu** or adjust the flags in **run.sh** via the `NVCC_DEFS` variable.

## Objectives

- **Traffic Pattern Analysis**:
  - Evaluate the performance differences among various SM-to-SM communication patterns.
- **Performance Characterization**:
  - Assess both overall bandwidth and per-request latency.
- **Parameter Sensitivity**:
  - Investigate the impact of different block sizes and ILP factors on the DSM performance.

## License

This project is released under the MIT License. Feel free to use, modify, and distribute the code as needed.
