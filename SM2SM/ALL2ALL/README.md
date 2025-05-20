# SM2SM Benchmark for Various Traffic Patterns

## Overview

This benchmark measures the **bandwidth** and **latency** of SM-to-SM communication inside a GPC on an NVIDIA H100 GPU under different traffic patterns and injection rates. It leverages two H100‑specific features:

- **Distributed Shared Memory (DSM)**
- **Thread‑Block Clusters** (with a maximum cluster size of 16, guaranteed to reside on one GPC)

In this benchmark, each source block reads from one or more destination blocks within the same cluster. The supported traffic patterns are:

- **TRAFFIC_RANDPERM**: Each block reads from a fixed random partner.
- **TRAFFIC_ROUNDROBIN**: Each block reads from the next block in the cluster.
- **TRAFFIC_UNIFORM**: Each block reads from a different random partner on each iteration.
- **TRAFFIC_ALL2ALL**: Each block reads from all other blocks in the cluster.
  - With the additional compile‑time flag `READ_ALL2ALL_FULL`, the ALL2ALL pattern can be configured to read the full shared memory size of each destination block instead of a reduced fraction.

Two access patterns are supported:

- **Stream Access** (`STRIDE=1`): Each thread in a warp accesses a different SMEM bank.
- **Strided Access** (`STRIDE>1`): Simulates bank‑conflict scenarios.

In addition, you can vary the injection level via:

- **Block size**: defines the number of threads per block (set at runtime)
- **ILP factor** (`ILP_FACTOR`): controls instruction‑level parallelism (set at compile‑time)

The benchmark can measure two performance metrics:
- **Overall Bandwidth** (`CALC_BW`): Reports the aggregate cycles taken to complete the memory transfers.
- **Per-Request Latency** (`CALC_LATENCY`): Reports the average latency (in clock cycles) per request.

Other configurable parameters include:
- **ITERATION**: The number of times the read loop is executed.
- **CLUSTER_SIZE**: Number of blocks per cluster (default: 16).

---

## Prerequisites

- **CUDA Toolkit** with `nvcc` (tested with compute capability `sm_90`)
- A Unix-like environment with a **bash** shell

---

## Folder Structure

```
.
├── main.cu          # CUDA kernel and host driver for the SM2SM benchmark
├── Makefile         # Build configuration (uses NVCC_DEFS for compile‑time flags)
├── run.sh           # Script to compile and execute the benchmark for multiple configurations
└── README.md        # Documentation for this project
```

---

## Compilation

The project uses a Makefile controlled by the environment variable `NVCC_DEFS` for compile‑time definitions. The following parameters are set at compile time:

- **CLUSTER_SIZE** is fixed at 16.
- **ILP_FACTOR** determines the unrolling factor (default: 8).
- **ITERATION** sets the number of loop iterations (default: 10000).
- **STRIDE** sets the element stride between threads (default: 1).

Measurement mode is selected by specifying one of:
- `-DCALC_BW` for bandwidth measurement.
- `-DCALC_LATENCY` for latency measurement.

Traffic pattern is selected by specifying exactly one of:
- `-DTRAFFIC_RANDPERM`
- `-DTRAFFIC_ROUNDROBIN`
- `-DTRAFFIC_UNIFORM`
- `-DTRAFFIC_ALL2ALL`

To enable the ALL2ALL pattern to read the full shared memory size, include `-DREAD_ALL2ALL_FULL` alongside `-DTRAFFIC_ALL2ALL`.

For example, to compile with an ILP factor of 8 and bandwidth measurement in the RANDPERM pattern, run:

```bash
make clean
make NVCC_DEFS="-DILP_FACTOR=8 -DCALC_BW -DTRAFFIC_RANDPERM"
```

And for latency measurement in the ALL2ALL (full mode) pattern, run:

```bash
make clean
make NVCC_DEFS="-DILP_FACTOR=8 -DCALC_LATENCY -DTRAFFIC_ALL2ALL -DREAD_ALL2ALL_FULL"
```

The provided **Makefile** compiles the `main.cu` file into the executable **SMEM**.

---

## Running the Experiment

The **run.sh** script automates both the compilation and the execution of the benchmark for multiple configurations. In this experiment, you can select different traffic patterns, measurement modes, and injection rates by looping over a set of block sizes. The executable now accepts **two runtime parameters**:
  
```bash
./SM2SM <numClusters> <blockSize>
```

Where:
- **`numClusters`** (default: 1) sets the number of clusters.
- **`blockSize`** (default: 1024) sets the number of threads per block.

### Explanation of run.sh

The provided **run.sh** script:
- Sets fixed parameters:
  - `CLUSTER_SIZE = 16` (used for the partner map, if needed)
  - `NUM_CLUSTERS = 1`
  - `ILP_FACTOR = 8`
  - `BLOCK_SIZES = (32, 64, 128, 256, 512, 1024)`
- Specifies a set of traffic patterns (valid options: `"randperm"`, `"roundrobin"`, `"uniform"`, `"all2all"`, `"all2all_full"`).
- Loops over each traffic pattern and measurement mode (latency and bandwidth).
- For each configuration, sets the appropriate compile‑time flags via `NVCC_DEFS`, cleans, and compiles the benchmark.
- Creates an output log file named using the measurement mode, traffic pattern, block size, number of clusters, and ILP factor. For example:
  - `bw_randperm_1024_1_8.log`
  - `latency_all2all_full_256_1_8.log`
- Runs the benchmark with the runtime parameters: `<numClusters>` and `<blockSize>`, appending the results to the output log.

To run the benchmark with the default settings, make sure the script is executable, then run:

```bash
chmod +x run.sh
./run.sh
```

---

## Parsing & Plotting (Bandwidth Only)

1. **Extract Raw Bandwidth Values:**  
   To extract the bandwidth data from the log files, run:
   ```bash
   chmod +x parse_logs.sh
   ./parse_logs.sh
   ```
   This script reads the `bw_<BLOCK_SIZE>_<NUM_CLUSTERS>_<ILP_FACTOR>.log` file and writes one bandwidth value per line into a new file (e.g., `bw_1024_1_8_hist.log`).

2. **Plot the Histogram:**  
   To visualize the bandwidth data, run:
   ```bash
   python3 plot.py bw_1024_1_8_hist.log bw_hist.png
   ```
   Here, `plot.py` takes:
   - `<input_file>`: path to the per‑SM bandwidth list (one value per line)
   - `<output_image>`: desired PNG filename for the histogram

---

## Experiment Settings and Customization

- **Measurement Modes:**
  - **Bandwidth Mode** (`-DCALC_BW`): Reports overall bandwidth (GB/s) using a cycle counter (`clock64()`).
  - **Latency Mode** (`-DCALC_LATENCY`): Reports average latency (clock cycles) per memory request using (`clock()`).

- **Traffic Patterns:**
  - **TRAFFIC_RANDPERM** (`-DTRAFFIC_RANDPERM`): Each block reads from a fixed random partner.
  - **TRAFFIC_ROUNDROBIN** (`-DTRAFFIC_ROUNDROBIN`): Each block reads from the next block in the cluster.
  - **TRAFFIC_UNIFORM** (`-DTRAFFIC_UNIFORM`): Each block reads from a different random partner on every iteration.
  - **TRAFFIC_ALL2ALL** (`-DTRAFFIC_ALL2ALL`): Each block reads from all other blocks in the cluster.
    - For full shared memory read in ALL2ALL mode, add `-DREAD_ALL2ALL_FULL`.

- **Injection Levels and Cluster Configuration:**
  - **ILP Unrolling (`ILP_FACTOR`)**: Compile‑time parameter (default: 8).
  - **Block Size (`blockSize`)**: Runtime parameter that affects the injection rate. The **run.sh** script loops over block sizes: 32, 64, 128, 256, 512, and 1024.
  - **Cluster Size (`CLUSTER_SIZE`)**: Fixed at compile time (default: 16).
  - **Number of Clusters (`numClusters`)**: Runtime parameter (default: 1).

To adjust any of these parameters (such as ILP factor, number of iterations, or memory access pattern via `STRIDE`), modify the corresponding compile‑time definitions in **main.cu** or adjust the flags in **run.sh** via `NVCC_DEFS`.

---

## Objectives

- **Traffic Pattern Analysis:**  
  Evaluate the performance differences among various SM-to-SM communication patterns.
  
- **Performance Characterization:**  
  Assess both overall bandwidth and per-request latency under different injection rates and access patterns.
