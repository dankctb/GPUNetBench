# L2 Cache Latency-Throughput and Latency Distribution

## Overview

This experiment is designed to study the latency and throughput of L2 cache accesses under varying injection rates and different access patterns. The traffic pattern is such that all SMs access all L2 slices. The goal is to characterize the latency distribution as the number of threads per CTA is varied and to assess the impact of adding a random delay before each memory access injection. The random delay is introduced with the hypothesis that it will help “schedule” injections to reduce contention in the interconnect network. The expected outcome is a reduction in both the tail and the average of the latency distribution, leading to increased bandwidth and decreased overall execution time compared to runs without delay.

Key features of this experiment:
- **Injection Rate Variation:** By modifying the number of threads per CTA (from as low as 1 up to 1024), the experiment evaluates how different injection rates affect latency and throughput.
- **Access Patterns:** The kernel supports three access patterns:
  - **Stream Access** – Each thread computes its access address on every iteration.
  - **Strided Access** – An initial address is computed once and then incremented by a constant stride.
  - **Random Access** – Starting addresses are randomly generated for each thread and then updated with a fixed stride.
- **Random Delay Injection:** Optionally, a random delay can be added before each memory access. This delay (a simple add instruction) can be controlled at compile time and set either per thread or per warp with a selectable number of delay steps.
- **Latency Measurement:** When enabled at compile time, the kernel uses `clock()` functions to measure latency. This data is then output, allowing the analysis of both the full latency distribution and the average latency.

## Prerequisites

- **CUDA Toolkit** with `nvcc`
- A compatible NVIDIA GPU (V100, A100, or H100)
- **bash**
- **Python 3** with the following packages:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `argparse`
- **ncu** (CUDA profiler)

## Folder Structure

```
├── main.cu         # CUDA/C++ source code for the L2 cache experiment
├── Makefile        # Build configuration (architecture flags for V100, A100, H100)
├── run.sh          # Script to automate compiling and running the experiment
├── hist.py         # Python plotting script to generate latency histograms
├── lat-bw.py  # Python plotting script to generate latency-throughput plots
├── parse_logs.sh   # Script to parse ncu-generated logs and produce two output files (bandwidth and kernel execution time)
└── README.md       # This documentation file
```

## Compilation

Use the provided **Makefile** to compile your experiment. L1 cache is disabled with the `-dlcm=cg` flag to ensure only L2 access is measured. The Makefile accepts an architecture variable to set the gencode appropriately:

```bash
# General form:
make ARCH=<v100|a100|h100>

# Examples:
make ARCH=v100
make ARCH=a100
make ARCH=h100
```

Additional compile-time flags (for example, to choose the access pattern, enable latency measurement, or enable random delay) can be specified via the environment variable `NVCC_DEFS`. For instance:

```bash
# Compile for V100 using stream access with latency measurement enabled:
make ARCH=v100 NVCC_DEFS="-DUSE_STREAM_ACCESS -DENABLE_LATENCY_MEASUREMENT"
```

## Run-Time Configuration

The **run.sh** script automates execution of the experiment. It supports both a normal run and a profiling run using the NVIDIA Compute Utility (`ncu`). The script allows you to choose the access pattern, whether random delay is enabled (and if so, whether it is applied per thread or per warp), and the number of delay steps.

### Usage

```bash
./run.sh <ncu|normal> [access_pattern: stream|strided|random] [random_delay: 0|1] [random_delay_method: thread|warp] [random_delay_steps]
```

**Examples:**

1. **Normal Run (without random delay, using stream access):**

   ```bash
   ./run.sh normal stream 0
   ```

2. **Profiling Run with Random Delay (random access, per warp, 64 delay steps):**

   ```bash
   ./run.sh ncu random 1 warp 64
   ```

In the script:
- The `COMMON_FLAGS` variable is built based on the provided command-line parameters.
- The executable is compiled with these flags using the Makefile.
- The experiment is run for a set of threads-per-CTA values (e.g., 1, 32, 64, …, 1024) and a fixed number of CTAs per SM (e.g., 2).
- When random delay is enabled, the output log file name is appended with a suffix indicating the random delay mode and delay steps.

## Data Collection and Output

For **normal mode** (latency measurement enabled):
- **Latency Distribution Files:**  
  Each file contains individual latency sample values (one per thread). Files are named as:
  ```
  <threads_per_CTA>_<CTAs_per_SM>_<accesspattern>[_rand_delay_<method>_<steps>].log
  ```
  For example, with 1024 threads per CTA, 2 CTAs per SM, using random access with a 64-step random delay per warp, the file is named:
  ```
  1024_2_random_rand_delay_warp_64.log
  ```

- **Average Latency Files:**  
  A single file (per experiment configuration) aggregates the average latency values for increasing injection rates. Its name is based on the access pattern and CTAs per SM (and includes the random delay suffix if enabled), for example:
  ```
  random_2.log
  ```
  Each line in this file corresponds to one injection rate (from 1 thread up to 1024 threads per CTA).

For **ncu (profiling) mode** (latency measurement disabled):
- L2 bandwidth and GPU kernel execution time will be recorded into files with names of the form:
  ```
  ncu_<threads_per_CTA>_<CTAs_per_SM>_<accesspattern>[_rand_delay_<method>_<steps>].log
  ```
  Running `parse_logs.sh` on these files produces two summary files:
  - Bandwidth:  
    ```
    BW_<accesspattern>_<CTAs_per_SM>[_rand_delay_<method>_<steps>].log
    ```
  - Kernel Execution Time:  
    ```
    KET_<accesspattern>_<CTAs_per_SM>[_rand_delay_<method>_<steps>].log
    ```
  Each of these summary files contains one converted numeric value per injection rate (ordered from lowest to highest threads per CTA).

## Plotting Results

### Histograms

A Python script (`hist.py`) is provided to generate histograms from the latency distribution files. Use it to compare different experimental settings. For example:

- **Injection Rate Comparison (without random delay):**

  ```bash
  python hist.py --files 1_all.log 32_all.log 64_all.log --legends "1 Threads" "32 Threads" "64 Threads"
  ```

- **Random Delay Comparison (1024 threads only):**

  ```bash
  python hist.py --files 1024_all.log 1024_all_rand_delay_warp_32.log 1024_all_rand_delay_thread_32.log \
      --legends "1024 no delay" "1024 rand delay warp (32)" "1024 rand delay thread (32)"
  ```

### Latency-Throughput Plots

Another Python script, **lat-bw.py**, is provided to plot L2 throughput versus average latency (or kernel execution time) for multiple data series.  
Each data series is represented by a pair of files:
- A **bandwidth file** (x–axis values, in GB/s).
- A **latency (or execution time) file** (y–axis values; if latency, in clock cycles; if execution time, in microseconds).

#### Usage

```bash
python3 lat-bw.py --bw_files <bw_file1> <bw_file2> ... \
                       --lat_files <lat_file1> <lat_file2> ... \
                       --legends <legend1> <legend2> ... \
                       --yaxis <latency|exec_time> --output <output_plot.png>
```

**Example:**

```bash
python3 lat-bw.py --bw_files BW_stream_2.log BW_stream_2_rand_delay_warp_32.log BW_stream_2_rand_delay_thread_32.log \
                       --lat_files KET_stream_2.log KET_stream_2_rand_delay_warp_32.log KET_stream_2_rand_delay_thread_32.log \
                       --legends "No Delay" "Random Delay (Warp)" "Random Delay (Thread)" \
                       --yaxis exec_time --output latency_throughput.png
```

This command plots the L2 throughput (GB/s) on the x–axis (in log₂ scale) versus average latency (in clock cycles) for each data series. Use `--yaxis exec_time` to plot kernel execution time (in µs) on the y–axis instead.

#### Plot Settings

- The x–axis is plotted on a log₂ scale.  
- Axis limits are set automatically based on the data across all series.
- The script overlays data from each series and includes a legend with user-supplied labels.

## Expected Results

- **Without Random Delay:**  
  Increasing threads per CTA (higher injection rates) is expected to lead to greater latency variance (longer tail) due to interconnect congestion.

- **With Random Delay:**  
  Randomizing the injection schedule is expected to smooth the latency distribution, reduce both tail latency and average latency, increase effective bandwidth, and reduce overall execution time compared to previous case.
