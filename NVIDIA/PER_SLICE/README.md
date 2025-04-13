# SLICE

## Overview

**SLICE** is a CUDA/C++ program that supports two modes of targeting Streaming Multiprocessors (SMs) to perform non‑coalesced access to a target L2 cache slice:

- **GPC Mode (default):** Select one or more Graphics Processing Clusters (GPCs) and activate up to a maximum number of SMs within them, then perform read accesses to a user‑specified L2 slice.
- **Direct SM Mode:** Compile with `-DUSE_DIRECT_SM` to target a single SM (specified on the command line) and a single L2 slice.

Key features:
- Automatically queries device properties (`multiProcessorCount`, `l2CacheSize`) at runtime.
- Supports V100, A100, and H100 architectures via compile‑time flags (`-DUSE_A100`, `-DUSE_H100`).
- Reads L2 slice mappings from an external CSV file (`L2_slices_<ARCH>.csv`).
- Easy configuration via command‑line arguments and Makefile variables.
- Includes a profiling script (`run.sh`) to automate `nvprof` or `ncu` runs.
- Includes a parsing script (`parse_logs.sh`) the outputs generated for plotting.
- Includes a plotting script (`plot.py`) to visualize L2 slice bandwidth distributions.
- `nvprof` only supports V100; use `ncu` for compatibility across all architectures.

## Prerequisites

- **CUDA Toolkit** (with `nvcc`)
- **nvprof** (CUDA profiler)
- **ncu** (CUDA profiler)
- **bash**
- **Python 3** with:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `argparse`
- Compatible NVIDIA GPU (V100, A100, or H100)

## Folder Structure

```
├── main.cu             # CUDA/C++ source code
├── Makefile            # Build configuration
├── run.sh              # Profiling automation script
├── parse_logs.sh       # Parse log files generated to make them ready to be plotted
├── plot.py             # Script to plot L2 slice bandwidth distributions
├── L2_slices_V100.csv  # V100 mapping (32 rows × 32 cols)
├── L2_slices_A100.csv  # A100 mapping (80 rows × 32 cols)
├── L2_slices_H100.csv  # H100 mapping (6 rows × 32 cols)
└── README.md           # This file
```

## CSV Matrix Format

- **Rows:** Number of L2 slices (V100=32, A100=80, H100=6)  
- **Columns:** 32 columns  
- Each row represents the per‑SM indices for that L2 slice.
- A100 slices are sorted in order that every 40 L2 slices, a different L2 partition is accessed. 
- A100 and H100 have a partitioned L2 cache with two L2 partitions containing 40 L2 slices each.
- The two L2 partitions are interconnected, but in H100 each SM can access only the local partition.

## Compilation

Use the Makefile to compile for your architecture and mode:

```bash
# General form:
make ARCH=<v100|a100|h100> MODE=<gpc|direct>

# Examples:
make ARCH=v100 MODE=gpc
make ARCH=a100 MODE=direct
```

| Variable | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| `ARCH`   | Target architecture:                                                        |
|          | • `v100` (default): `-arch=sm_70`                                           |
|          | • `a100`: `-arch=sm_80 -DUSE_A100`                                          |
|          | • `h100`: `-arch=sm_90 -DUSE_H100`                                          |
| `MODE`   | Selection mode:                                                             |
|          | • `gpc` (default): GPC Mode                                                 |
|          | • `direct`: Direct SM Mode (`-DUSE_DIRECT_SM`)                              |

The resulting executable is named `SLICE`.

## Usage

### GPC Mode

```bash
./SLICE <GPCselectedList> <SMmax> <slice_index>
```

| Argument             | Type     | Description                                                                                              |
|----------------------|----------|----------------------------------------------------------------------------------------------------------|
| `GPCselectedList`    | string   | Comma‑separated GPC IDs to use (e.g. `0,1,2`)                                                            |
| `SMmax`              | integer  | Maximum number of SMs to activate within the selected GPCs (index limit)                                 |
| `slice_index`        | integer  | L2 slice row index to access (0–`N-1`, where `N` is number of slices for your architecture)             |

**Example:**

```bash
./SLICE 0,1 14 5
```

Targets GPC 0 and 1, up to SM index 14, using L2 slice row 5.

---

### Direct SM Mode

Compile with:

```bash
make MODE=direct
```

Then run:

```bash
./SLICE <SMid> <slice_index>
```

| Argument      | Type     | Description                                                           |
|---------------|----------|-----------------------------------------------------------------------|
| `SMid`        | integer  | Target SM ID (0–`numSM-1`, where `numSM` is the device’s SM count)   |
| `slice_index` | integer  | L2 slice row index to access (0–`N-1`)                                |

**Example:**

```bash
./SLICE 3 5
```

Targets SM 3 and L2 slice row 5.

## Profiling with run.sh

Make the script executable and run it:

```bash
chmod +x run.sh
./run.sh [nvprof|ncu|normal]
```

It is possible to choose if profiling with nvprof, with ncu or if running normally.

| Argument      | Description                                                           |
|---------------|-----------------------------------------------------------------------|
| `nvprof`        | Profile the code with nvprof (compatible only with V100)   |
| `ncu` | Profile the code with ncu (compatible with all the supported GPUs)                              |
| `normal` | Run the code without profiling (this is useful if no profiler is avilable or if sudo access is not provided)    |

- **Direct SM Mode:** loops over all SM IDs and slices, logs to  
  `<arch>_direct_SM<SM>_slice<slice>.log`
- **GPC Mode:** loops over GPC indices and slices, logs to  
  `<arch>_gpc_GPC<gpc>_slice<slice>.log`

Each invocation uses either:

```bash
nvprof --metrics \
  l2_tex_read_throughput,l2_tex_write_throughput,\
l2_read_throughput,l2_write_throughput \
  --log-file <logFileName> \
  ./SLICE …
```

or, for broader architecture support:

```bash
ncu --metrics \
  l1tex__m_xbar2l1tex_read_bytes.sum.per_second,\
l1tex__m_l1tex2xbar_write_bytes.sum.per_second \
  --log-file <logFileName> \
  ./SLICE …
```

or, if profilers are not available:
```
./SLICE … >> <logFileName>
```

## Plotting Results

Use `plot.py` to visualize L2 slice bandwidth distributions:

```bash
python plot.py --input-log <log-file> --output plot.png
```

| Option         | Description                                                |
|----------------|------------------------------------------------------------|
| `--input-log`  | Path to the log file                                       |
| `--output`     | Filename for the output plot image (default: `figure.png`) |

The log file must contain in every line a different bandwidth value. Blank lines are allowed.
The script will parse the log and automatically adjust axis ranges and bin sizes.
The script directily support the log file generated by non profiled execution. 
Support to parse the profiled generated logs is guaranted by `parse_logs.sh`.

Make the script executable and run it before `plot.py` to generate the input logs:

```bash
chmod +x parse_logs.sh
./run.sh
```


## Kernel Configuration Parameters

Adjust at compile time in `main.cu`:

```cpp
#define ITERATION     10000
```

Recompile after modifying. Do not alter other macros.

## License

This project is released under the MIT License. Feel free to use, modify, and distribute.

