# SM2SM Communication One‑to‑One Benchmark

This benchmark measures the **bandwidth** and **latency** of SM→SM communication within a GPC on an NVIDIA H100 GPU. It leverages two H100‑specific features:

- **Distributed Shared Memory (DSM)**
- **Thread‑Block Clusters** (with a fixed cluster size of 16 that is guaranteed to reside on one GPC)

By sweeping over every source–destination **cluster rank** pair (excluding self‑loops), the experiment can observe inter‑SM performance. In addition, you can vary the injection level via:

- **Block size**: defines the number of threads per block (set at runtime)
- **ILP factor** (`ILP_FACTOR`): controls instruction‑level parallelism (set at compile‑time)

Two access patterns are supported:

- **Stream Access** (`STRIDE=1`): each thread in a warp accesses a different SMEM bank.
- **Strided Access** (`STRIDE>1`): simulates bank‑conflict scenarios.

Each kernel uses all available shared memory per block and repeats its SM→SM transfers **ITERATION** times (set via `-DITERATION`, default 10000).

---

## Prerequisites

- **CUDA Toolkit** with `nvcc` targeting `sm_90`
- **bash** shell
- **Python 3** with the following packages:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `argparse`

---

## Folder Structure

```
.
├── main.cu         # CUDA kernel & host driver for the SM→SM test (modified to accept runtime parameters)
├── Makefile        # Builds the executable using NVCC_DEFS for compile‑time flags
├── run.sh          # Script to compile & run the benchmark over all SM source/destination combinations
├── parse_logs.sh   # Extracts bandwidth values from the log files (used in bandwidth mode)
├── plot.py         # Reads per‑SM bandwidth file and plots a histogram
└── README.md       # This documentation
```

---

## Compilation

The benchmark is built from `main.cu` using several compile‑time definitions. In particular:

- **CLUSTER_SIZE** is fixed at compile time (default: 16).
- **ILP_FACTOR** determines the unrolling factor for memory accesses (default: 8).
- **ITERATION** (default: 100000) sets how many times the SM→SM transfer loop runs.
- **STRIDE** (default: 1) sets the memory access pattern (increase to simulate bank conflicts).

Additionally, you select the measurement mode at compile time by choosing one of:
- `-DCALC_BW` for overall bandwidth measurement (using `clock64()`)
- `-DCALC_LATENCY` for per‑request latency measurement (using `clock()`)

If neither flag is provided, the default measurement mode is bandwidth (`CALC_BW`).

For example, to compile in bandwidth mode with the default ILP factor, run:
```bash
make clean
make NVCC_DEFS="-DILP_FACTOR=8 -DCALC_BW"
```

For latency mode, use:
```bash
make clean
make NVCC_DEFS="-DILP_FACTOR=8 -DCALC_LATENCY"
```

You may modify other compile‑time parameters by editing the Makefile.

---

## Running the Experiment

After compiling, the benchmark produces an executable named **SMEM**. This executable accepts **four runtime parameters** as follows:

```bash
./SMEM <rt_destSM> <rt_srcSM> <numClusters> <blockSize>
```

Where:

- **`rt_destSM`** (default: 1)  
  Specifies the *destination SM rank* within each cluster. This SM performs the warm‑up phase by filling shared memory and recording its SM ID.

- **`rt_srcSM`** (default: 0)  
  Specifies the *source SM rank* within each cluster. This SM reads the shared memory with ILP unrolling and records timing information—either total cycles for bandwidth mode or per-request latency.

- **`numClusters`** (default: 1)  
  Determines the number of clusters to launch. The total number of blocks is computed as:  
  ```
  total_blocks = numClusters * CLUSTER_SIZE
  ```  
  where `CLUSTER_SIZE` is fixed at compile time (typically 16).

- **`blockSize`** (default: 1024)  
  Sets the number of threads per block. Changing this value affects the level of parallelism and the amount of shared memory used per block.

---

### Running with run.sh

The provided `run.sh` script simplifies the process by compiling and running tests for both measurement modes (latency and bandwidth) over all source/destination rank combinations (excluding src == dest).

A brief overview of what `run.sh` does:

1. **Fixed Parameters:**  
   - Sets `CLUSTER_SIZE=16`, `NUM_CLUSTERS=1`, `BLOCK_SIZE=1024`, and `ILP_FACTOR=8`.

2. **Loop over Modes:**  
   - The script compiles and runs tests in both **latency** and **bandwidth** modes.
   - It sets the proper mode flag (`-DCALC_LATENCY` or `-DCALC_BW`) and composes an output log file name as `<mode>_<blockSize>_<NUM_CLUSTERS>_<ILP_FACTOR>.log` (e.g., `latency_1024_1_8.log` or `bw_1024_1_8.log`).

3. **Testing all SM Pairs:**  
   - For each mode, the script loops over all destination (`rt_destSM`) and source (`rt_srcSM`) rank combinations within the cluster, skipping cases where the source and destination are identical.
   - For each combination, it prints a header indicating the current test and then executes the benchmark with the appropriate runtime parameters.

After completion, the script prints messages indicating where the results are stored.

---

## Parsing & Plotting (Bandwidth Only)

1. **Extract Raw Bandwidth Values:**  
   To extract the bandwidth data from the log files, run:
   ```bash
   chmod +x parse_logs.sh
   ./parse_logs.sh
   ```
   This script reads the `bw_<BLOCK_SIZE>_<NUM_CLUSTERS>_<ILP_FACTOR>.log` file and writes a new file (e.g., `bw_1024_1_8_hist.log`) containing one raw bandwidth value per line.

2. **Plot the Histogram:**  
   To visualize the bandwidth data, run:
   ```bash
   python3 plot.py bw_1024_1_8_hist.log bw_hist.png
   ```
   Here, `plot.py` takes:
   - `<input_file>`: the per‑SM bandwidth list (one value per line)
   - `<output_image>`: the desired PNG filename for the histogram

---

## Experiment Settings Summary

- **Measurement Modes:**  
  - **Latency Mode** (`-DCALC_LATENCY`): uses `clock()` to measure per‑request latency (in clock cycles)
  - **Bandwidth Mode** (`-DCALC_BW`): uses `clock64()` to measure overall transfer cycles and computes GB/s

- **Injection Levels:**  
  - **Threads per Block (`blockSize`):** runtime parameter (default: 1024)
  - **ILP Unrolling (`ILP_FACTOR`):** compile‑time parameter (default: 8)

- **Cluster Configuration:**  
  - **Ranks per Cluster (`CLUSTER_SIZE`):** compile‑time constant (fixed at 16)
  - **Number of Clusters (`numClusters`):** runtime parameter (default: 1)

- **Access Pattern:**  
  - **Stream Access (`STRIDE=1`):** each thread accesses a different SMEM bank  
  - **Strided Access (`STRIDE>1`):** simulates bank conflicts by accessing with a stride
