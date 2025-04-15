# BISECTION

## Overview

`BISECTION` is a CUDA benchmark that measures the bisection bandwidth of an L2‑to‑L2 interconnect on NVIDIA A100 and H100 GPUs. It does so by selecting one or more Graphics Processing Clusters (GPCs), mapping their SMs into a contiguous rank space, and issuing global‐memory reads into a single L2 partition. When you target the *remote* partition, the measured bandwidth approximates the L2‑to‑L2 bisection bandwidth. 
Note that the GPCs in same partition should be checked before running this code, and those GPCs might vary for different GPUs (even of the same architecture).

Key features:

- **GPC Selection:** choose any subset of GPCs on the command line  
- **Address Matrix:** loads a per‑SM address matrix from a CSV (`<ARCH>-<PARTITION>.csv`)  
- **Bandwidth Calculation:** uses CUDA events to time the kernel and computes GB/s  
- **Configurable CTAs per SM:** via first command‑line argument  
- **Multi‑arch Support:** compile‑time flags for A100 (`-DUSE_A100`) and H100 (`-DUSE_H100`)  
- **Automation Script:** `run.sh` to build and run with incremental GPC sets  

## Prerequisites

- **CUDA Toolkit** (nvcc)  
- **Make**  
- **bash**  
- A working Linux environment  
- CSV files in the working directory:  
  - `A100-0.csv`, `A100-1.csv` (64 rows × 32 cols)  
  - `H100-0.csv`, `H100-1.csv` (164 rows × 32 cols)  

## Folder Structure

```
├── main.cu           # CUDA/C++ source
├── Makefile          # Build rules
├── run.sh            # Build & run automation script
├── A100-0.csv        # Address matrix for A100 partition 0 (64×32)
├── A100-1.csv        # Address matrix for A100 partition 1 (64×32)
├── H100-0.csv        # Address matrix for H100 partition 0 (164×32)
├── H100-1.csv        # Address matrix for H100 partition 1 (164×32)
└── README.md         # This file
```

## CSV Matrix Format

- **Rows:**  
  - A100 → 64 rows  
  - H100 → 164 rows  
- **Columns:** 32  
- **File names:** `<ARCH>-<PARTITION>.csv` (e.g. `H100-1.csv`)  
- Each entry is the per‑warp index into the global buffer for that SM.

## Compilation

Use the provided Makefile:

```bash
# Default: ARCH=a100, PARTITION=0
make

# To target H100, partition 1:
make ARCH=h100 NVCC_DEFS="-DPARTITION=1"
```

| Variable    | Description                                                                                     |
|-------------|-------------------------------------------------------------------------------------------------|
| `ARCH`      | `a100` (default) or `h100` → sets `-arch=sm_80/-arch=sm_90` and `-DUSE_A100`/`-DUSE_H100`       |
| `NVCC_DEFS` | Extra `-D` flags; use `-DPARTITION=<0|1>` to select the L2 partition                            |

The resulting executable is named `BISECTION`.

## Usage

```bash
./BISECTION <CTAs_per_SM> <GPC0> [<GPC1> ...]
```

| Argument         | Type    | Description                                                       |
|------------------|---------|-------------------------------------------------------------------|
| `CTAs_per_SM`    | int     | Number of thread‑blocks (CTAs) to launch on each SM               |
| `GPC IDs`        | ints    | One or more GPC indices (0–6) to include in the benchmark run     |

**Example:**

```bash
./BISECTION 10 0 2 5
```

Launches 10 CTAs per SM on GPCs 0, 2, 5.

## Automating with run.sh

Make it executable and run:

```bash
chmod +x run.sh
./run.sh
```

At the top of `run.sh` you can configure:

- `ARCH` (`a100`/`h100`)  
- `PARTITION` (`0`/`1`)  
- `CTAS_PER_SM` (defaults to 32)  
- `GPC_LIST` (e.g. `0 1 3`)  

The script will:

1. **Rebuild** `BISECTION` with your `ARCH` and `PARTITION`  
2. **Run** with incremental subsets of `GPC_LIST`:  
   ```bash
   ./BISECTION 32 0
   ./BISECTION 32 0 1
   ./BISECTION 32 0 1 3
   ```

## Output

Each invocation prints:

```
Time = <ms> ms
Active SMs = <N>
L2 read bandwidth = <GB/s>
```

These metrics are meant to compare the L2‑to‑L2 bisection bandwidth across different GPC combinations and partitions.