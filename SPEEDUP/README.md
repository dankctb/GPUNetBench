# SPEEDUP

## Overview

**SPEEDUP** is a CUDA/C++ program that selects one or more Graphics Processing Clusters (GPCs) (CPC for H100 is also supported) and runs a microbenchmark on a chosen subset of Streaming Multiprocessors (SMs) within those GPCs (or CPC). It measures aggregate bandwidth (GB/s) to L2 cache by executing either read‑based or write‑based kernels in a streaming way.

**Key features:**
- **Multi‑GPU support**: V100 (default), A100, H100, H100CPC
- **Operation modes**: read (cached loads via `__ldcg`) or write
- **Sort modes**:
  - **GPC**: ascending SM IDs
  - **TPC**: even SM IDs first, then odd
- **Runtime parameters**:
  - **CTA**: CTAs (thread‑blocks) per SM
  - **WARP**: warps per CTA
  - **ITERATION**: number of kernel launches to average over
  - **GPCselectedList**: comma‑separated GPC IDs (e.g. `0,1,3`)
  - **SMmax**: truncate unified SM list to this length

## Prerequisites

- **CUDA Toolkit** (with `nvcc`)
- **nvprof** (CUDA profiler)
- **ncu** (CUDA profiler)
- Compatible NVIDIA GPU (V100, A100, or H100)
- **bash**, **make**

## Folder Structure

```
├── main.cu           # Main CUDA/C++ source
├── Makefile          # Build rules (OP, ARCH, SORT)
├── run.sh            # Compile + run/profile script
└── README.md         # This documentation
```

## Compilation

Use the provided `Makefile`. You can choose:

- **OP**: `write` (default) or `read`
- **ARCH**: `v100` (default), `a100`, `h100` or `h100cpc`
- **SORT**: `tpc` (default) or `gpc`

```bash
# Default (write, v100, gpc)
make

# Read‑mode on V100, TPC‑sort
make OP=read SORT=tpc

# Write‑mode on A100
make ARCH=a100

# Read‑mode on H100, TPC‑sort
make OP=read ARCH=h100 SORT=tpc
```

L1 cache is disabled with the `-dlcm=cg` flag to ensure only L2 access is measured.

## Usage

```
./your_program <CTA> <WARP> <ITERATION> <GPCselectedList> <SMmax>
```

| Argument            | Description                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------|
| `CTA`               | CTAs (thread‑blocks) per SM                                                                  |
| `WARP`              | Warps per CTA                                                                                |
| `ITERATION`         | Number of kernel iterations (to average bandwidth)                                           |
| `GPCselectedList`   | Comma‑separated GPC IDs (e.g. `0,1,3`).  Valid range depends on ARCH:                        |
|                     | • V100: 0–5                                                                                  |
|                     | • A100: 0–6                                                                                  |
|                     | • H100: 0–1                                                                                  |
|                     | • H100CPC: 0–1                                                                               |
| `SMmax`             | Truncate the unified SM list to this many SM IDs                                             |

**Example:**

```bash
# 2 CTAs/SM, 32 warps/CTA, 100 iterations, GPCs 0 and 1, top 10 SMs
./SPEEDUP 2 32 100 0,1 10
```

## Automated Runs & Profiling

`run.sh` wraps both compilation (via `make`) and:

- **Normal execution** → outputs to `out_<ARCH>_<OP>_<SORT>_gpc<GPC>_sm<SM>.txt`
- **nvprof profiling** → logs to `prof_<ARCH>_<OP>_<SORT>_gpc<GPC>_sm<SM>.log`

```bash
chmod +x run.sh

# Default: write, v100, gpc, both run+profile for SMmax=1..14
./run.sh

# Read‑mode on V100, TPC‑sort, run only
./run.sh read v100 tpc run

# Write‑mode on A100, profile only
./run.sh write a100 gpc profile
```

**Metrics collected under profiling:**
- `l2_tex_read_throughput`
- `l2_tex_write_throughput`
- `l2_read_throughput`
- `l2_write_throughput`

## Compile‑Time Flags

In `main.cu`, you can also manually control:

```cpp
// -DUSE_READ    → switch to read‑based kernel (cached loads)
// -DUSE_A100    → use A100 SM mappings
// -DUSE_H100    → use H100 SM mappings
// -DSORT_GPC    → even‑first, then odd SM sorting, to avoid TPC contention and check GPC speedup
```

These are automatically handled by the `Makefile` when you pass `OP=read`, `ARCH=a100|h100`, or `SORT=tpc`.

## License

This project is released under the MIT License. Feel free to use, modify, and distribute.

