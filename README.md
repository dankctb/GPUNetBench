# GPU Microbenchmarks

This repository contains CUDA-based benchmarks designed to evaluate various aspects of GPU memory and interconnection network on NVIDIA GPUs (V100, A100, H100). Each benchmark focuses on distinct architectural components, using as metrics bandwidth, latency, and execution time.

These microbenchmarks were developed and used in:

* **Z.Jin, C.Rocca et al., "Uncovering Real GPU NoC Characteristics: Implications on Interconnect Architecture,"** MICRO 2024 \[[IEEE Reference](https://doi.org/10.1109/MICRO61859.2024.00070)]
* **C.Rocca and J.Kim, "A Microbenchmark-Based Characterization of On-Chip Networks Architectures in Modern GPUs,"** KAIST Master Thesis, 2025

If you use this benchmark suite in your work, please cite these references (see full citations below).

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

For convenience, we report some useful specifications for the GPUs used in these experiments:

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

This suite was designed to enable fine-grained characterization of GPU memory and NoC behavior, as presented in \[Jin et al., MICRO 2024] and \[Rocca & Kim, 2025].

* **AGGREGATE:**

  * Measures aggregate read bandwidth from L2 cache and HBM in a STREAM-like fashion. Configurable CTAs per SM and threads per CTA.

* **HBM\_LAT-BW:**

  * Evaluates latency and throughput of HBM accesses under varying injection rates and access patterns. Allows random delay injection to control NoC scheduling.

* **L2\_LAT-BW:**

  * Evaluates latency and throughput of L2 accesses under varying injection rates and access patterns. Also supports random delay injection.

* **MP:**

  * Performs non-coalesced memory accesses targeting multiple L2 slices. Allows selecting GPCs as sources and L2 memory partitions as destinations.

* **PER\_SLICE:**

  * Targets a single L2 slice to evaluate bandwidth uniformity across SM and GPC sources, despite non-uniform zero-load latency.

* **SM2SM:**

  * Benchmarks the SM-to-SM network using Distributed Shared Memory (DSM) and Thread-Block Cluster mechanisms introduced in Hopper:

    * **ALL2ALL:** Bandwidth and latency under different traffic patterns across all SMs in a GPC.
    * **ONE2ONE:** Bandwidth and latency across each SM pair.
    * **SM2SM+L2:** Evaluates interference when DSM and L2 traffic share the NoC.

* **SMEM\_LAT-BW:**

  * Local shared memory latency and bandwidth microbenchmarks. Includes streaming and strided (bank-conflict) access patterns.

* **SPEEDUP:**

  * Measures NoC input speedup by selectively activating SMs in GPC, CPC, or TPC hierarchies. Supports read and write operations.

* **BISECTION:**

  * Measures L2-to-L2 interconnection bisection bandwidth by having SMs on one side of the chip access remote L2 partitions.

---

## General Requirements

* **CUDA Toolkit** with `nvcc` compiler
* Compatible NVIDIA GPUs (V100, A100, H100)
* Python 3 with:

  * `numpy`
  * `matplotlib`
  * `scipy`
  * `pandas`
  * `argparse`
* Bash shell for execution scripts
* Make
* NVIDIA profiling tools (`nvprof`, `ncu`)

---

## Common Settings

* L1 cache is disabled (`-dlcm=cg`).

---

## References

Please cite the following works if you use this benchmark suite:

* **Conference Paper:**

  ```
  @INPROCEEDINGS{10764573,
    author    = {Jin, Zhixian and Rocca, Christopher and Kim, Jiho and Kasan, Hans and Rhu, Minsoo and Bakhoda, Ali and Aamodt, Tor M. and Kim, John},
    title     = {Uncovering Real GPU NoC Characteristics: Implications on Interconnect Architecture},
    booktitle = {2024 57th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
    year      = {2024},
    pages     = {885-898},
    doi       = {10.1109/MICRO61859.2024.00070}
  }
  ```

* **Thesis:**

  ```
  @mastersthesis{Rocca2025,
    author  = {Christopher Rocca and John Kim},
    title   = {A Microbenchmark-Based Characterization of On-Chip Networks Architectures in Modern GPUs},
    school  = {KAIST},
    year    = {2025}
  }
  ```
