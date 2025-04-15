# MP

## Overview

`MP` is a CUDA/C++ program that selects one or more Graphics Processing Clusters (GPCs) and runs a kernel on 
specific Streaming Multiprocessors (SMs) within the set of selected GPCs. 
It performs non‑coalesced memory accesses to L2 cache slices defined by rows of an external CSV matrix previously collected.

Key features:
- Reads a matrix of unsigned integers from a CSV file (`L2_slices_4.csv`).
- Selects slices based on user‑specified memory partitions (MPs), where each MP contributes up to 8 slices.
- Allows runtime configuration of:
  - **GPCselectedList**: a comma-separated list of GPC IDs (0..5) to use
  - **SMmax**: maximum SM index to activate within the GPC selected
  - **MPnum**: number of memory partitions (MPs) selected
  - **slicesPerMP**: number of slices to extract per MP (up to 8 slices per MP)
  - **MP IDs**: specific MP identifiers (0..3) || 4 is for choosing slices in interleaved way from different MPs
- Includes a profiling script (`run.sh`) to automate nvprof runs for L2 throughput metrics.

## Prerequisites

- **Python 3** with:
  - `numpy`
  - `matplotlib`
  - `pandas`
- **bash**, **make**
- **CUDA Toolkit** (with `nvcc` compiler)
- **nvprof** (CUDA profiler)
- V100 GPU


## Folder Structure

```
├── main.cu           # Main CUDA/C++ source
├── L2_slices_4.csv   # Input matrix (32 x 32*MULTIPLIER)
├── run.sh            # Profiling script for nvprof
├── plot.py           # 
├── plot2.py          #
└── README.md         # This documentation
```

## CSV Matrix Format

- File: `L2_slices_4.csv`
- Rows: **BLOCK_SIZE** (default 32)
- Columns: **BLOCK_SIZE × MULTIPLIER** (default 32 × 4 = 128)
- Each line: comma-separated unsigned integer values

## Compilation

The code is compiled in the executibile file provided. L1 cache is disabled throught '-dlcm=cg' flag to correctly perform only L2 accesses. It will be compiled for CC7.0 (Volta V100)
-DENABLE_ALL_SM flag is used to skip selection of SMs.

## Usage

```
./MP <GPCselectedList> <SMmax> <MPnum> <slicesPerMP> <MP_id[0]> ... <MP_id[MPnum-1]>
```

| Argument            | Description                                                                                                                                                   |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|                             
| `GPCselectedList`   | Graphics Processing Cluster index (integer 0..5) || 6 has 28 interleaved SMs from different GPCs (with no TPC contention)                                     |
| `SMmax`             | Maximum SM index to activate within the selected GPC                                                                                                          |
| `MPnum`             | Number of memory partitions (MPs); total slices = `MPnum × SlicesPerMp`                                                                                       |
| `SlicesPerMp`       | Number of L2 slices selected per MP                                                                                                                           |
| `MP_id[i]`          | Memory Partition ID for each MP (integer 0..3); list length = `MPnum` || MP 4 has L2 slices selected in a interleved way from all the MPs                     |

**Example:**
```bash
# Select GPC 0 and 1, up to SM 20, use 2 MPs (IDs 0 and 3)
./sameGPC 0,1 20 2 0 3
```

This run will access 16 slices (2 MPs × 8 slices each) on SMs of GPC1 up to index 14.

## Profiling with nvprof

Make the profiling script executable and run it:

```bash
chmod +x run.sh
./run.sh
```

This generates log files named:
```
GPC${GPCselected}_SM${SMmax}_MP${MPnum}.log
```
Each contains metrics:
- `l2_tex_read_throughput`
- `l2_tex_write_throughput`
- `l2_read_throughput`
- `l2_write_throughput`

Another profiling option is provided if you want to check the transations and requests to L2 cache, in a non aggregated way (Each L2 slice providing the metrics). 
This option is commented out by default.

## Kernel Configuration Parameters

Defined at compile time via macros in `main.cu`:
```cpp
#define ITERATION      10000
```
Adjust this value and recompile if needed.
Do not modify the other macros.

## License

This project is released under the MIT License. Feel free to use, modify, and distribute.

