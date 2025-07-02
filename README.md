# GPU Microbenchmarks

This repository contains CUDA-based benchmarks designed to evaluate various aspects of GPU memory and interconnection networks on NVIDIA GPUs (V100, A100, H100). Each benchmark focuses on distinct architectural components, using bandwidth, latency, and execution time as key metrics.

---

## Citation

If you use this benchmark suite in your publications or thesis, please cite:

1. Zhixian Jin, Christopher Rocca, Jiho Kim, Hans Kasan, Minsoo Rhu, Ali Bakhoda, Tor M. Aamodt, and John Kim. **Uncovering Real GPU NoC Characteristics: Implications on Interconnect Architecture.** In *Proceedings of the 57th IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 885–898, 2024. [https://doi.org/10.1109/MICRO61859.2024.00070](https://doi.org/10.1109/MICRO61859.2024.00070)

2. Christopher Rocca and John Kim. **A Microbenchmark-Based Characterization of On-Chip Networks Architectures in Modern GPUs.** Master’s Thesis, KAIST, 2025.

### BibTeX

```bibtex
@inproceedings{jin2024uncovering,
  title     = {Uncovering Real GPU NoC Characteristics: Implications on Interconnect Architecture},
  author    = {Jin, Zhixian and Rocca, Christopher and Kim, Jiho and Kasan, Hans and Rhu, Minsoo and Bakhoda, Ali and Aamodt, Tor M. and Kim, John},
  booktitle = {57th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year      = {2024},
  pages     = {885--898},
  doi       = {10.1109/MICRO61859.2024.00070}
}

@mastersthesis{rocca2025microbenchmark,
  title  = {A Microbenchmark-Based Characterization of On-Chip Networks Architectures in Modern GPUs},
  author = {Rocca, Christopher},
  school = {KAIST},
  year   = {2025}
}
```

---

## Repository Structure

```
GPUbench
├── AGGREGATE
├── BISECTION
├── HBM_LAT-BW
├── L2_LAT-BW
├── MP
├── PER_SLICE
├── SM2SM
│   ├── ALL2ALL
│   ├── ONE2ONE
│   └── SM2SM+L2
├── SMEM_LAT-BW
└── SPEEDUP
```

---

## GPUs Specifications

| Feature                  | V100     | A100     | H100 (PCIe) |
| ------------------------ | -------- | -------- | ----------- |
| **SMs**                  | 80       | 108      | 114         |
| **TPCs**                 | 40       | 54       | 57          |
| **GPCs**                 | 6        | 7        | 7           |
| **Max SMs / GPC**        | 14       | 16       | 18          |
| **L2 cache size**        | 6 MB     | 40 MB    | 50 MB       |
| **L2 cache slices**      | 32       | 80       | 80          |
| **L2 Memory Partitions** | 4        | 10       | 10          |
| **GPU memory bandwidth** | 0.9 TB/s | 2 TB/s   | 2 TB/s      |
| **Memory controllers**   | 8        | 10       | 10          |
| **GPU max clock speed**  | 1.38 GHz | 1.41 GHz | 1.755 GHz   |

---

## Benchmarks Overview

This suite enables fine-grained characterization of GPU memory and on-chip network behavior, as detailed in the cited works.

* **AGGREGATE**: STREAM-like aggregate read bandwidth tests for L2 cache and HBM, configurable CTAs/SMs and threads/CTA.
* **HBM\_LAT-BW**: HBM latency and throughput under varied injection rates and patterns, with optional random-delay scheduling.
* **L2\_LAT-BW**: L2 cache latency and throughput under varied injection rates and patterns, with optional random-delay scheduling.
* **MP**: Non-coalesced memory accesses across multiple L2 cache slices; configurable GPC sources and L2 partitions.
* **PER\_SLICE**: Bandwidth uniformity tests targeting single L2 cache slices, across SM/GPC sources, despite zero-load latency nonuniformity.
* **SM2SM**: SM-to-SM network benchmarks using Distributed Shared Memory (DSM) and Thread-Block Clusters:

  * **ALL2ALL**: Traffic patterns across all SMs within a GPC.
  * **ONE2ONE**: Pairwise SM bandwidth/latency measurements.
  * **SM2SM+L2**: Interference analysis between DSM and L2 traffic.
* **SMEM\_LAT-BW**: Local shared memory latency/bandwidth with streaming and strided (bank-conflict) patterns.
* **SPEEDUP**: NoC input speedup by selective SM activation at GPC/CPC/TPC levels, for both read and write.
* **BISECTION**: L2-to-L2 interconnection bisection bandwidth measurements across chip partitions.

---

## General Requirements

* **CUDA Toolkit** with `nvcc`
* NVIDIA GPUs: V100, A100, or H100
* Python 3: `numpy`, `matplotlib`, `scipy`, `pandas`, `argparse`
* Bash shell
* Make
* NVIDIA profiling tools: `nvprof`, `ncu`

### Common Settings

* Disable L1 cache: compile with `-dlcm=cg`.
