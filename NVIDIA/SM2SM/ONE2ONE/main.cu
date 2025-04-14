#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

// Fixed compile‑time cluster size
#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 16
#endif

#ifndef ILP_FACTOR
#define ILP_FACTOR 8
#endif

#ifndef REPS
#define REPS 100000
#endif

#ifndef STRIDE
#define STRIDE 1
#endif

// Select exactly one of these at compile time:
//   -DCALC_BW      → overall bandwidth (clock64())
//   -DCALC_LATENCY → per‑request latency (clock())
#if !defined(CALC_BW) && !defined(CALC_LATENCY)
#define CALC_BW
#endif

namespace cg = cooperative_groups;

// Inline PTX to get the SM ID
__device__ inline unsigned int get_smid() {
    unsigned int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

/*
  Kernel:
  - Warming block (rank rt_destSM) fills shared memory and writes its SM ID.
  - Reading block (rank rt_srcSM) streams through it with ILP unrolling,
    measures either total cycles or per‑request latency, and writes:
      [destSM, srcSM, timing…]
*/
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void kernel(unsigned long long *out,
            int num_ints,
            int rt_destSM,
            int rt_srcSM) {
  extern __shared__ int sdata[];

  cg::cluster_group cluster = cg::this_cluster();
  unsigned int rank = cluster.block_rank();
  int cluster_id = blockIdx.x / CLUSTER_SIZE;

  // === WARM & record destSM ===
  if (rank == rt_destSM && threadIdx.x == 0) {
    unsigned int my_smid = get_smid();
  #ifdef CALC_BW
    // slot 0 per cluster holds destSM
    out[3 * cluster_id + 0] = my_smid;
  #else
    // [base+0] = destSM
    out[cluster_id * (blockDim.x + 2) + 0] = my_smid;
  #endif
    // fill shared memory
    sdata[0] = my_smid;
    for (int i = 1; i < num_ints; i++)
      sdata[i] = my_smid + i;
  }

  // map the shared buffer for all blocks in the cluster
  int * __restrict__ ws = cluster.map_shared_rank(sdata, rt_destSM);

  int local_sum = 0;
#ifdef CALC_LATENCY
  unsigned long long lat_acc = 0, lat_cnt = 0;
#endif

  // wait until warm is done
  cluster.sync();

  // === READ & record srcSM + timing ===
  if (rank == rt_srcSM) {
#ifdef CALC_BW
    unsigned long long startCycles = clock64();
#endif

    for (int rep = 0; rep < REPS; rep++) {
      for (int off = 0; off < STRIDE; off++) {
        int idx = threadIdx.x * STRIDE + off;
        // ILP‑unrolled main chunk
        for (; idx + (ILP_FACTOR - 1) * blockDim.x * STRIDE < num_ints;
             idx += blockDim.x * ILP_FACTOR * STRIDE) {
#ifdef CALC_LATENCY
          unsigned long long t0 = clock();
#endif
#pragma unroll
          for (int j = 0; j < ILP_FACTOR; j++)
            local_sum += ws[idx + j * blockDim.x * STRIDE];
#ifdef CALC_LATENCY
          unsigned long long t1 = clock();
          lat_acc += (t1 - t0);
          lat_cnt++;
#endif
        }
        // tail
        for (; idx < num_ints; idx += blockDim.x * STRIDE) {
#ifdef CALC_LATENCY
          unsigned long long t0 = clock();
#endif
          local_sum += ws[idx];
#ifdef CALC_LATENCY
          unsigned long long t1 = clock();
          lat_acc += (t1 - t0);
          lat_cnt++;
#endif
        }
      }
    }

    __syncthreads();

#ifdef CALC_BW
    unsigned long long totalCycles = clock64() - startCycles;
    unsigned int my_smid = get_smid();
    // slot 1 per cluster holds srcSM
    out[3 * cluster_id + 1] = my_smid;
    // slot 2 per cluster holds cycle count
    out[3 * cluster_id + 2] = totalCycles;
#else  // CALC_LATENCY
    unsigned long long avgLat = lat_cnt ? (lat_acc / lat_cnt) : 0;
    int base = cluster_id * (blockDim.x + 2);
    // [base+0] = destSM (from warm)
    // [base+1] = srcSM
    out[base + 1] = get_smid();
    // [base+3 ... base+3+blockDim.x-1] = per-thread avgLat
    out[base + 2 + threadIdx.x] = avgLat;
#endif
  }

  cluster.sync();
}

int main(int argc, char **argv) {
  // Defaults
  int rt_destSM   = 1;
  int rt_srcSM    = 0;
  int numClusters = 1;
  int blockSize   = 1024;

  // Parse: ./exe <rt_destSM> <rt_srcSM> <numClusters> <blockSize>
  if (argc > 1) rt_destSM   = atoi(argv[1]);
  if (argc > 2) rt_srcSM    = atoi(argv[2]);
  if (argc > 3) numClusters = atoi(argv[3]);
  if (argc > 4) blockSize   = atoi(argv[4]);

  // Device setup
  int dev = 0;
  cudaDeviceProp prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);

  int maxSMemOptin;
  cudaDeviceGetAttribute(&maxSMemOptin,
    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  size_t shared_bytes = maxSMemOptin;
  int num_ints = shared_bytes / sizeof(int);

  // Grid & output sizing
  int total_blocks = numClusters * CLUSTER_SIZE;
  size_t out_size;
#ifdef CALC_BW
  // 3 values per cluster: destSM, srcSM, cycles
  out_size = numClusters * 3 * sizeof(unsigned long long);
#else
  // (blockSize + 2) values per cluster: destSM, srcSM, avgLat per-thread
  out_size = numClusters * (blockSize + 2) * sizeof(unsigned long long);
#endi

  // Allocate
  unsigned long long *d_out, *h_out;
  h_out = (unsigned long long*)malloc(out_size);
  cudaMalloc(&d_out, out_size);

  // Kernel attributes
  cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  cudaFuncSetAttribute(kernel,
    cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  // Launch
  kernel<<< total_blocks, blockSize, shared_bytes >>>(
    d_out, num_ints, rt_destSM, rt_srcSM);
  cudaDeviceSynchronize();

  // Copy back
  cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);

  double clkHz = prop.clockRate * 1e3;  // kHz→Hz

#ifdef CALC_BW
  for (int i = 0; i < numClusters; i++) {
    unsigned long long destSM = h_out[3*i + 0];
    unsigned long long srcSM  = h_out[3*i + 1];
    unsigned long long cycles = h_out[3*i + 2];
    unsigned long long bytes  = (unsigned long long)num_ints * sizeof(int) * REPS;
    double bpc = (double)bytes / (double)cycles;
    double bw  = bpc * clkHz / 1e9;
    printf("Cluster %d destSM %llu srcSM %llu Bandwidth %.4f GB/s\n",
           i, destSM, srcSM, bw);
  }
#else
  for (int i = 0; i < numClusters; i++) {
    double average = 0;
    int base = i * (blockSize + 2);
    unsigned long long destSM = h_out[base + 0];
    unsigned long long srcSM  = h_out[base + 1];
    for (int j = 0; j < blockSize; j++) {
      unsigned long long avgLat = h_out[base + 2 + j];
      average += avgLat;
    }
    average /= blockSize;
    printf("Cluster %d destSM %llu srcSM %llu Avg Latency %.4f clock cycles\n",
           i, destSM, srcSM, average);
  }
#endif

  cudaFree(d_out);
  free(h_out);
  return 0;
}
