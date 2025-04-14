# SM2SM Communication One‑to‑One Benchmark

## Overview

This experiment measures the **bandwidth** and **latency** of SM→SM communication inside a single GPC on an NVIDIA H100 GPU. We leverage two H100‑specific features:

- **Distributed Shared Memory (DSM)**  
- **Thread‑Block Clusters** (max cluster size 16, guaranteed to reside on one GPC)

By sweeping over every source–destination **cluster rank** pair (excluding self‑loops), we can observe inter‑SM performance.

We also vary injection level via:

- **Block size** (threads per block)  
- **ILP factor** (instruction‑level parallelism unrolling)

Two access patterns are supported:

- **Stream Access** (`STRIDE=1`): each thread in a warp accesses a different SMEM bank.  
- **Strided Access** (`STRIDE>1`): simulates bank‑conflict scenarios.

Each kernel uses all available shared memory per block and repeats its SM→SM transfers **ITERATIONS** times (via `-DITERATIONS`).

## Prerequisites

- **CUDA Toolkit** with `nvcc` targeting `sm_90`  
- **bash** shell  
- **Python 3** with packages:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `argparse`

## File Structure

```
.
├── main.cu          # CUDA kernel & host driver for SM→SM test
├── Makefile         # Builds `DSM` (uses NVCC_DEFS for flags)
├── run.sh           # Compiles & runs both modes over all SM pairs
├── parse_logs.sh    # Extracts bandwidth values from bw_*.log
├── hist.py          # Reads per‑SM bandwidth file and plots a histogram
└── README.md        # This documentation
```

## Compilation

You can compile manually or let `run.sh` handle it:

```bash
# Bandwidth mode
make clean
make NVCC_DEFS="-DILP_FACTOR=8 -DCALC_BW"

# Latency mode
make clean
make NVCC_DEFS="-DILP_FACTOR=8 -DCALC_LATENCY"
```

## Running the Experiment

```bash
chmod +x run.sh
./run.sh
```

By default, `run.sh` sets:

- `CLUSTER_SIZE=16`  
- `NUM_CLUSTERS=1`  
- `BLOCK_SIZE=1024`  
- `ILP_FACTOR=8`

It produces two log files:

```bash
bw_${BLOCK_SIZE}_${NUM_CLUSTERS}_${ILP_FACTOR}.log
latency_${BLOCK_SIZE}_${NUM_CLUSTERS}_${ILP_FACTOR}.log
```

Each line in `bw_*.log` looks like:

```
Cluster 0 destSM <rank> srcSM <rank> Bandwidth <x.y> GB/s
```

## Parsing & Plotting (Bandwidth Only)

1. **Extract** raw bandwidth values:

   ```bash
   chmod +x parse_logs.sh
   ./parse_logs.sh
   ```
   For example it will read from `bw_1024_1_8.log` and write the output in `bw_1024_1_8_hist.log`.

   `parse_logs.sh` reads the `bw_*.log` file and writes one bandwidth value per line.

2. **Plot** the histogram:

   ```bash
   python3 hist.py bw_1024_1_8_hist.log bw_hist.png
   ```

   `hist.py` takes two arguments:
   - `<input_file>`: path to the per‑SM bandwidth list (one value per line)
   - `<output_image>`: desired PNG filename for the histogram

## Experiment Settings

- **Measurement modes**  
  - **Latency** (`-DCALC_LATENCY`) → clock cycles   
  - **Bandwidth** (`-DCALC_BW`) → GB/s  
- **Injection levels**  
  - **Threads/block** (`BLOCK_SIZE`): 1024  
  - **ILP unrolling** (`ILP_FACTOR`): 8  
- **Cluster config**  
  - **Ranks/cluster** (`CLUSTER_SIZE`): 16  
  - **Num clusters** (`NUM_CLUSTERS`): 1  

> To vary `BLOCK_SIZE`, `ILP_FACTOR`, or add `-DSTRIDE=<n>`, edit `run.sh` and adjust `NVCC_DEFS`.

## Objectives

1. **Uniformity Analysis**  
   - Verify whether all SM→SM paths deliver similar bandwidth.  
   - Check for latency variations across rank pairs.  
2. **Injection Impact**  
   - Observe performance changes with different block sizes and ILP factors.  
3. **DSM vs. SMEM**  
   - Compare H100’s DSM performance against local SMEM.

## License

This project is released under the MIT License. Feel free to use, modify, and distribute.