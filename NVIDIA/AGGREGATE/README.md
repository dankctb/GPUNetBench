# BW

## Overview

BW is a benchmark program designed to measure both L2 cache and HBM (High-Bandwidth Memory) read aggregate bandwidth on NVIDIA GPUs. It achieves this by issuing coalesced memory accesses to target either L2 cache or HBM, depending on runtime parameters. By setting the data size multiplier to 1 and using a high loop count, the kernel stresses L2 cache accesses; Otherwise, setting the data size multiplier to a high value while using a single loop targets HBM accesses.

**Key features:**
- Measures both L2 cache and HBM (global memory) read bandwidth.
- Adjustable kernel launch configuration using command-line parameters.
- Uses CUDA’s `__ldcg` intrinsic to force read-only cache loads.
- Supports customization of CTA (thread blocks per SM) and warps per thread block.
- Compatible with modern NVIDIA GPUs including V100, A100, and H100.
- Integrated with three Python plotting scripts:
  - `plot_per_warp.py`: Generates plots on a per-warp basis.
  - `plot_per_cta.py`: Generates plots on a per-CTA (thread block) basis.
  - `plot_2d_colormap.py`: Combines both dimensions in a 2D color map.

## Prerequisites

- **CUDA Toolkit** (with `nvcc` compiler)
- **Python 3** with:
  - `numpy`
  - `matplotlib`
  - `pandas`
- (Optional) **nvprof** or **ncy** for collecting L2 throughput metrics
- Compatible NVIDIA GPU (V100, A100, or H100)

## Folder Structure

```
├── main.cu             # Main CUDA/C++ source file for bandwidth measurement
├── README.md           # Project documentation
├── run.sh              # Profiling script for nvprof (for L2 throughput metrics)
├── plot_per_warp.py    # Python plotting script for per-warp analysis
├── plot_per_cta.py     # Python plotting script for per-CTA analysis
└── plot_2d_colormap.py # Python plotting script for combined 2D visualization
```

## Compilation

Use the provided **Makefile** to compile the experiment. L1 cache is disabled with the `-dlcm=cg` flag to ensure only L2 access or HBM is measured. The Makefile accepts an architecture variable to set the gencode appropriately:

```bash
# General form:
make ARCH=<v100|a100|h100>

# Examples:
make ARCH=v100
make ARCH=a100
make ARCH=h100
```


## Usage

```bash
./MP <CTA> <WARP> <ITERATION> <loopCount> <sizeMultiple>
```

| Argument       | Description                                                                                                                                                          |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CTA`          | Number of thread blocks (CTAs) per SM to launch.                                                                                                                     |
| `WARP`         | Number of warps per thread block (each warp contains 32 threads).                                                                                                    |
| `ITERATION`    | Number of kernel execution iterations for averaging the bandwidth measurement.                                                                                       |
| `loopCount`    | Number of loops executed inside the kernel. Use a high value for L2 cache accesses, or set to 1 for HBM (global memory) accesses.                                     |
| `sizeMultiple` | Multiplier for the L2 cache size used to compute the total data transfer size. Set to 1 for L2 access measurement; use a higher value (>> 1) for HBM benchmarking.  |

**Examples:**

- **L2 Cache Measurement:** (aggregated L2 access with high loop count)
  ```bash
  ./MP 1 1 100 1000 1
  ```

- **HBM Measurement:** (global memory access with a high data size and a single loop)
  ```bash
  ./MP 1 1 100 1 10
  ```

## Automated Benchmark Script (run.sh)

The `run.sh` script automates a comprehensive sweep of CTA and warp configurations for both L2 cache and HBM experiments. It builds the `MP` executable and executes two phases:

1. **L2 Cache Phase**
   - Uses a high inner loop count (`loopCount=L2_LOOPS`, default 1000) and `sizeMultiple=1` to stress L2 cache.
   - Results are appended to `results_L2.log`.

2. **HBM Phase**
   - Uses a single inner loop (`loopCount=1`) and a larger `sizeMultiple` (`HBM_SIZE_MULT`, default 10) to target global HBM bandwidth.
   - Results are appended to `results_HBM.log`.

**Key script parameters (modifiable at the top of `run.sh`):**
- `ITER`: Number of measurement iterations per configuration (default 5).
- `L2_LOOPS`: Number of inner loops for L2-cache stress (default 1000).
- `HBM_SIZE_MULT`: Size multiplier for HBM stress (default 10).
- `CTAS`: Range of CTA values (1–32).
- `WARPS`: Range of warp counts per block (1–32).

**Usage:**
```bash
# Ensure the script is executable:
chmod +x run.sh
# Run the automated benchmark:
./run.sh
```

After completion, you will have two log files:
- `results_L2.log` — bandwidth measurements for L2 cache experiments.
- `results_HBM.log` — bandwidth measurements for HBM global memory experiments.

The output bandwidth are in GB/s. Each bach of lines (separated by blank lines) is a different CTA configuration. Lines inside the same batch, are showing increasing Warp configuration.

## Python Plotting Scripts

Three Python scripts are provided for visualization:
- **plot_per_warp.py:** Plots bandwidth measurements on a per-warp basis.
- **plot_per_cta.py:** Plots bandwidth measurements on a per-CTA (thread block) basis.
- **plot_2d_colormap.py:** Creates a 2D colormap that combines both warp and CTA-level data.

To run the scripts, ensure that the following Python libraries are installed:
```bash
pip install numpy pandas matplotlib
```

## License

This project is released under the MIT License. Feel free to use, modify, and distribute.

