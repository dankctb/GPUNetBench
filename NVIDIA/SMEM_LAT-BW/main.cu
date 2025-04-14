#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

//----------------------------------------------------------------------------
// Compile‑time macros:
//
//   -DSTRIDE=<n>        // stride in ints; default=1 (streaming access)
//   -DCALC_BW           // bandwidth mode (uses clock64())
//   -DCALC_LATENCY      // latency mode (uses clock())
//   -DILP_FACTOR=<n>    // default=8
//   -DITERATION=<n>     // default=10000
//----------------------------------------------------------------------------

#ifndef STRIDE
  #define STRIDE 1
#endif
#ifndef ILP_FACTOR
  #define ILP_FACTOR 8
#endif
#ifndef ITERATION
  #define ITERATION 10000
#endif

// inline PTX to get SM ID
__device__ inline unsigned int get_smid() {
  unsigned int sm;
  asm("mov.u32 %0, %%smid;" : "=r"(sm));
  return sm;
}

__global__ void kernel(unsigned long long *out, int numInts) {
  extern __shared__ int sdata[];

  // Warm‑up: fill shared mem
  for (int i = threadIdx.x; i < numInts; i += blockDim.x)
    sdata[i] = i;
  __syncthreads();

  int local_sum = 0;
#ifdef CALC_LATENCY
  unsigned long long lat_acc = 0, lat_cnt = 0;
#endif

__syncthreads();

#ifdef CALC_BW
  unsigned long long startCycles = clock64();
#endif

  // Access loop (stride=STRIDE; if STRIDE==1 → streaming)
  for (int rep = 0; rep < ITERATION; rep++) {
    int idx = threadIdx.x * STRIDE;
    // bulk via ILP
    for (; idx + (ILP_FACTOR-1)*blockDim.x*STRIDE < numInts;
          idx += blockDim.x*ILP_FACTOR*STRIDE)
    {
#ifdef CALC_LATENCY
      unsigned long long t0 = clock();
#endif
#pragma unroll
      for (int j = 0; j < ILP_FACTOR; j++) {
        local_sum += sdata[idx + j*blockDim.x*STRIDE];
      }
#ifdef CALC_LATENCY
      unsigned long long t1 = clock();
      lat_acc += (t1 - t0);
      lat_cnt++;
#endif
    }
    // remainder
    for (; idx < numInts; idx += blockDim.x*STRIDE) {
#ifdef CALC_LATENCY
      unsigned long long t0 = clock();
#endif
      local_sum += sdata[idx];
#ifdef CALC_LATENCY
      unsigned long long t1 = clock();
      lat_acc += (t1 - t0);
      lat_cnt++;
#endif
    }
  }

  __syncthreads();

#ifdef CALC_BW
  unsigned long long totalCycles = clock64() - startCycles;
#endif
#ifdef CALC_LATENCY
  unsigned long long avgLat = lat_cnt ? (lat_acc/lat_cnt) : 0;
#endif

  // prevent optimization
  sdata[threadIdx.x] = local_sum;

  // write per‑block output
#ifdef CALC_BW
  if (threadIdx.x == 0) {
    int o = blockIdx.x*2;
    out[o    ] = get_smid();
    out[o + 1] = totalCycles;
  }
#else
  int base = blockIdx.x*(blockDim.x+2);
  if (threadIdx.x == 0) {
    out[base    ] = get_smid();
    out[base + 1] = avgLat;
  }
  out[base + 2 + threadIdx.x] = avgLat;
#endif
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s <blockSize> <numBlocks>\n", argv[0]);
    return 1;
  }
  int blockSize = atoi(argv[1]);
  int numBlocks = atoi(argv[2]);

  // Device setup
  int dev = 0;  
  cudaGetDevice(&dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);

  int maxSMemOptin;
  cudaDeviceGetAttribute(&maxSMemOptin,
    cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  size_t sbytes = maxSMemOptin;
  int numInts = sbytes / sizeof(int);

  // allocate output buffer
  unsigned long long *d_out, *h_out;
#ifdef CALC_BW
  size_t cnt = numBlocks*2;
#else
  size_t cnt = numBlocks*(blockSize+2);
#endif
  h_out = (unsigned long long*)malloc(cnt*sizeof(unsigned long long));
  cudaMalloc(&d_out, cnt*sizeof(unsigned long long));

  // set shared‑mem attributes
  cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, sbytes);

  // launch
  kernel<<<numBlocks, blockSize, sbytes>>>(d_out, numInts);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, cnt*sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  double freq = prop.clockRate * 1e3;

#ifdef CALC_BW
  unsigned long long sumC=0; double sumBW=0;
  double sumTimeUs=0; // Sum of execution times in microseconds
  unsigned long long bytes = (unsigned long long)numInts*sizeof(int)*ITERATION;
  printf("\nPer-block Bandwidth:\n");
  for (int b = 0; b < numBlocks; b++) {
    int i = b*2;
    unsigned sm = (unsigned)h_out[i];
    unsigned long long cyc = h_out[i+1];
    double bw = (bytes/(double)cyc)*freq/1e9;
    double execTimeUs = (cyc / freq) * 1e6; // Calculate execution time in microseconds
    printf("Block %d | ILP %u | SM %u | Time %.2f us | BW %.4f GB/s\n",
           b, ILP_FACTOR, sm, execTimeUs, bw);
    sumC += cyc; sumBW += bw;
    sumTimeUs += execTimeUs;
  }
  printf("\nAggregate:\n");
  printf("Avg time=%.2f us  Sum BW=%.2f GB/s\n",
         sumTimeUs / numBlocks, sumBW);
#else
  double sumLat=0;
  printf("\nPer-block Latency:\n");
  for (int b = 0; b < numBlocks; b++) {
    int base = b*(blockSize+2);
    unsigned sm = (unsigned)h_out[base];
    unsigned long long blk=0;
    for (int t = 0; t < blockSize; t++)
      blk += h_out[base+2+t];
    double avg = blk/(double)blockSize;
    printf("Block %d | ILP %u | SM %u | Avg Lat %.4f cycles\n",
           b, ILP_FACTOR, sm, avg);
    sumLat += avg;
  }
  printf("\nAggregate:\n");
  printf("Overall Avg Lat=%.4f cycles\n", sumLat/numBlocks);
#endif

  cudaFree(d_out);
  free(h_out);
  return 0;
}
