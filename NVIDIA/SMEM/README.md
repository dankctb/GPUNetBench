# Local SMem Latency and Bandwidth Benchmark

## Overview

This experiment is designed to evaluate the latency and bandwidth of local shared memory (SMEM) on an NVIDIA GPU under various injection levels. The primary goal is to assess how changes in the number of threads per block (injection rate) and the ILP (instruction-level parallelism) unrolling factor affect performance. Two access patterns are provided:

- **Stream Access:** Executed by using a stride value of 1. Each thread in a warp will access a different SMEM bank.
- **Strided Access:** Executed by setting the stride to a value greater than 1. This mode simulates different shared memory bank conflict scenarios—the higher the stride (beyond 1), the more bank conflicts are likely to occur, which negatively impacts performance.

The benchmark supports two measurement modes:

- **Latency Measurement (-DCALC_LATENCY):** Uses `clock()` to measure per-access latency and computes an average latency per block.
- **Bandwidth Measurement (-DCALC_BW):** Uses `clock64()` to measure the total cycles for the entire memory-access loop; bandwidth is then derived by comparing the total bytes moved against the cycle count.

This experiment allocate all the SMEM available for a thread block. The SMEM is read ITERATION times.

## Prerequisites

- **CUDA Toolkit** with the `nvcc` compiler.
- A supported NVIDIA GPU (e.g. V100, A100, or H100).
- **bash** shell.

## File Structure

```
.
├── main.cu         # Unified CUDA source code for the SMEM benchmark
├── Makefile        # Build configuration file (supports various architectures)
├── run.sh          # Bash script to compile and run the benchmark with various settings
└── README.md       # This documentation file
```

## Compilation

Use the provided **Makefile** to compile the experiment. The Makefile supports an `ARCH` variable to choose the GPU architecture (e.g., `v100`, `a100`, or `h100`):

```bash
# General compilation:
make ARCH=<v100|a100|h100>

# Example for V100:
make ARCH=v100
```

Additional compile‑time flags can be passed via the environment variable `NVCC_DEFS`. For example, to compile the benchmark in streaming mode with latency measurement, you can run:
```bash
make ARCH=v100 NVCC_DEFS="-DSTRIDE=1 -DCALC_LATENCY -DILP_FACTOR=8"
```
To switch to bandwidth measurement, replace `-DCALC_LATENCY` with `-DCALC_BW`.

It is possible to modify the number of times the SMEM is read by setting `-DITERATIONS`.

## Run-Time Configuration

The compiled executable (**SMEM**) accepts two runtime parameters:
- `<blockSize>`: Number of threads per block.
- `<numBlocks>`: Number of blocks (typically set to 1 in these experiments).

The provided **run.sh** script automates running the benchmark for various ILP factors, block sizes, and stride values. In this setup, the access pattern is controlled by setting the `STRIDE` compile flag.  
- `STRIDE=1` results in **stream access**.
- A value greater than 1 produces **strided access**.

### Running the Experiment

The provided **run.sh** script iterates over multiple ILP factors, block sizes, and stride values. It performs two types of runs:
- One with latency measurement enabled.
- One with bandwidth measurement enabled.

Each run compiles the application with the appropriate flags and then runs the benchmark using 1 block. Results are saved in output files whose names contain:
- The measurement mode (`latency` or `bw`)
- The access pattern type (determined from the `STRIDE` value)
- The block size  
- The number of blocks

For example, an output file name might be:  
`smem_latency_stride1_1024_1.txt` (stream access with block size 1024 and 1 thread block, latency mode).

## Example Usage

Make the script executable and run it:
```bash
chmod +x run.sh
./run.sh
```
This script will:
- Loop through ILP factors: `1, 2, 4, 8, 16, 32, 64`
- Loop through block sizes: `32, 64, 128, 256, 512, 1024`
- Loop through stride values: `1, 2, 4, 8, 16, 32`
- For each configuration, compile and run the benchmark with 1 block.
- Save the results in files named by the configuration.

## Experiment Objectives

The main objectives of this benchmark are to:
- **Assess SMEM latency:** Measure how average latency per memory access changes with varying injection levels and ILP factors.
- **Evaluate SMEM bandwidth:** Determine the effective bandwidth of SMEM under different thread and ILP configurations.
- **Study Bank Conflicts:** Compare streaming (STRIDE=1) vs. strided access patterns to understand the impact of bank conflicts on performance. A streaming access pattern should show no bank conflicts, whereas increasing stride values will likely increase conflicts, thereby reducing performance.
- **Injection Rate Impact:** By modifying block size (injection rate) and ILP factors (injection burst size), evaluate how performance scales with different levels of memory access contention.

## License

This project is released under the MIT License. You are free to use, modify, and distribute this code.