#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <time.h>

//----------------------------------------------------------------------------
// User configurable compile options:
//
//   To choose one memory access mode, compile with one of:
//      -DUSE_STREAM_ACCESS  for stream access
//      -DUSE_STRIDED_ACCESS for strided access
//      -DUSE_RANDOM_ACCESS  for random access mode
//
//   To enable random delay functionality, compile with:
//      -DENABLE_RANDOM_DELAY
//   Optionally, compile with -DWARP_RANDOM_DELAY to do the delay per warp 
//   (instead of per thread).
//
//   To choose how many steps of random delay to allow, define RANDOM_DELAY_STEPS.
//      (Example: -DRANDOM_DELAY_STEPS=32)
//
//   To enable latency measurement (the clock() calls and latency output), compile with:
//      -DENABLE_LATENCY_MEASUREMENT
//----------------------------------------------------------------------------
#ifndef RANDOM_DELAY_STEPS
    #define RANDOM_DELAY_STEPS 32
#endif
#define NUM_POSSIBLE_DELAYS RANDOM_DELAY_STEPS

//----------------------------------------------------------------------------
// Experiment parameters.
//----------------------------------------------------------------------------
#define NUM_ITERATIONS 5
#define ITER_START     4

//
// Device function to get SM ID using inline PTX.
//
__device__ unsigned int get_smid(void) {
    unsigned int ret;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

//
// Device function to get a random delay value.
// Returns an integer in the range [0, RANDOM_DELAY_STEPS-1].
//
__device__ unsigned int get_delay(unsigned int x) {
    unsigned int possible_delay[NUM_POSSIBLE_DELAYS];
    #pragma unroll
    for (unsigned int i = 0; i < NUM_POSSIBLE_DELAYS; i++) {
         possible_delay[i] = i;
    }
    return possible_delay[x % NUM_POSSIBLE_DELAYS];
}

//
// Device function to perform a delay operation.
// Increment the accumulator for the given delay number of steps.
//
__device__ unsigned int perform_delay(unsigned int delay, volatile unsigned int accumulator) {
#pragma unroll 1
    for (unsigned int i = 0; i < delay; i++) {
        accumulator++;
    }
    return accumulator;
}

//
// Kernel using shared memory to record latency measurements.
// Supports three memory access modes:
//  - In random access mode (-DUSE_RANDOM_ACCESS), the kernel obtains its starting
//    address from a random-address array (d_randAddrs) and updates it every iteration.
//  - In stream access mode (-DUSE_STREAM_ACCESS), the address is computed each iteration.
//  - In strided access mode (-DUSE_STRIDED_ACCESS), the address is computed once and then
//    incremented by a constant stride.
// Additionally, if ENABLE_RANDOM_DELAY is defined, a delay is inserted.
// Latency measurement (using clock()) is enabled only when ENABLE_LATENCY_MEASUREMENT is defined.
__global__ void kernel(unsigned int* d_data,
#ifdef USE_RANDOM_ACCESS
                         unsigned int* d_randAddrs,
#endif
                         unsigned int* d_latency_out,
                         unsigned int arr_size) {
    unsigned int smid = get_smid();
    unsigned int address = 0;

#ifdef USE_RANDOM_ACCESS
    // Random access: use global thread id to fetch starting address.
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    address = d_randAddrs[tid];
#elif defined(USE_STRIDED_ACCESS)
    // Strided access: compute address based on thread/block indices.
    if (threadIdx.x % 32 < 16)
        address = (threadIdx.x % 32) * 256 + (threadIdx.x / 32) * 8 + blockIdx.x % 8 + smid * 64 * 256;
    else 
        address = (threadIdx.x % 32) * 256 + 16 * 256 + (threadIdx.x / 32) * 8 + (blockIdx.x % 8) + smid * 64 * 256;
#endif

#ifdef ENABLE_LATENCY_MEASUREMENT
    uint32_t start_time = 0;
    uint32_t end_time = 0;
    // Allocate dynamic shared memory (one slot per thread) for latency sample.
    extern __shared__ unsigned int shared_latency[];
#endif

    volatile unsigned int accumulator = 0;
    volatile unsigned int value = 0;
    volatile unsigned int accumulator3 = 0;

    // Determine delay value.
#ifdef ENABLE_RANDOM_DELAY
    #ifdef WARP_RANDOM_DELAY
         unsigned int delay = get_delay(threadIdx.x / 32);
    #else
         unsigned int delay = get_delay(threadIdx.x);
    #endif
#endif

    __syncthreads();

    // Main measurement loop.
#pragma unroll 1
    for (int i = 0; i < NUM_ITERATIONS; i++) {

#ifdef USE_RANDOM_ACCESS
         // For random access: update the address using a stride (i*32).
         address = (address + i * 32) % arr_size;
#elif defined(USE_STREAM_ACCESS)
         // Stream access: compute address based on thread/block indices and iteration.
         address = (threadIdx.x * 8 + blockIdx.x * blockDim.x * 8 + blockDim.x * i) % arr_size;
#elif defined(USE_STRIDED_ACCESS)
         // Strided access: update the pre-computed address.
         address = (address + i * 8) % arr_size;
#endif

#ifdef ENABLE_RANDOM_DELAY
         accumulator3 = perform_delay(delay, accumulator3);
#endif

#ifdef ENABLE_LATENCY_MEASUREMENT
         start_time = clock();
#endif
         accumulator += d_data[address];
#ifdef ENABLE_LATENCY_MEASUREMENT
         end_time = clock();
         if (i == ITER_START)
             shared_latency[threadIdx.x] = (end_time - start_time);
#endif
         value += accumulator;
    }

    __syncthreads();
    value += accumulator3;
    // Dummy write to force retention.
    d_data[address] = value;

#ifdef ENABLE_LATENCY_MEASUREMENT
    d_latency_out[threadIdx.x + blockIdx.x * blockDim.x] = shared_latency[threadIdx.x];
#endif
}

//
// Warm-up kernel.
//
__global__ void k(unsigned int* d, int len, int lps) {
    for (int l = 0; l < lps; l++) {
        for (int i = threadIdx.x + blockDim.x * blockIdx.x; 
             i < len; 
             i += gridDim.x * blockDim.x)
        {
            d[0] += d[i];
        }
    }
}

//
// Host code (main).
// Command-line options:
//   -t <threads_per_CTA>   (default: 1)
//   -c <CTAs_per_SM>        (default: 1)
//   -o <output_mode>        ('d' for full latency distribution, 'a' for average, 'b' for both)
// The grid dimensions are set based on the number of SMs.
// For full latency distribution, the output file is named as:
//    threads_per_CTA_CTAs_per_SM_accesspattern[optional_rand_delay].log
// For average latency, the file name is based on access pattern and CTAs (without threads_per_CTA)
// and new average values are appended (one per different thread count).
//
int main(int argc, char* argv[]) {
    unsigned int threads_per_CTA = 1;
    unsigned int CTAs_per_SM = 1;
    char output_mode = 'b'; // 'd' = full distribution; 'a' = average; 'b' = both
    
    int opt;
    while ((opt = getopt(argc, argv, "t:c:o:")) != -1) {
        switch(opt) {
            case 't': threads_per_CTA = atoi(optarg); break;
            case 'c': CTAs_per_SM = atoi(optarg); break;
            case 'o': output_mode = optarg[0]; break;
            default:
                printf("Usage: %s -t <THREADS_PER_CTA> -c <CTAs_per_SM> -o <output mode: a, d, or b>\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    
    // Query device properties.
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int numSM = devProp.multiProcessorCount;
    
    // Compute usable data size from L2 cache.
    unsigned int L2_total_bytes = devProp.l2CacheSize; // in bytes
    unsigned int reserved_bytes = 1024 * 1024 * sizeof(unsigned int);
    if (L2_total_bytes <= reserved_bytes) {
        fprintf(stderr, "Error: L2 cache size (%u bytes) is insufficient for reserved space (%u bytes).\n", 
                L2_total_bytes, reserved_bytes);
        exit(EXIT_FAILURE);
    }
    unsigned int usable_bytes = L2_total_bytes - reserved_bytes;
    unsigned int data_size = usable_bytes / sizeof(unsigned int);
    
    // Allocate and initialize data array.
    unsigned int* d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, data_size * sizeof(unsigned int));
    if (err != cudaSuccess) {
         fprintf(stderr, "Failed to allocate device memory for data (error: %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }
    unsigned int* h_data = (unsigned int*)malloc(data_size * sizeof(unsigned int));
    for (unsigned int i = 0; i < data_size; i++) {
         h_data[i] = i;
    }
    err = cudaMemcpy(d_data, h_data, data_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
         fprintf(stderr, "Failed to copy data from host to device (error: %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }
    free(h_data);
    
#ifdef USE_RANDOM_ACCESS
    // For random access, set up random addresses.
    unsigned int totalThreads_random = numSM * CTAs_per_SM * threads_per_CTA;
    unsigned int* h_randAddrs = (unsigned int*)malloc(totalThreads_random * sizeof(unsigned int));
    srand((unsigned int)time(NULL));
    for (unsigned int i = 0; i < totalThreads_random; i++) {
        unsigned int r = rand() % (data_size / 8);
        h_randAddrs[i] = r * 8;
    }
    unsigned int* d_randAddrs;
    cudaMalloc((void**)&d_randAddrs, totalThreads_random * sizeof(unsigned int));
    cudaMemcpy(d_randAddrs, h_randAddrs, totalThreads_random * sizeof(unsigned int), cudaMemcpyHostToDevice);
    free(h_randAddrs);
#endif
    
#ifdef ENABLE_LATENCY_MEASUREMENT
    // Allocate latency output array.
    unsigned int totalThreads_global = numSM * CTAs_per_SM * threads_per_CTA;
    unsigned int* d_latency_out;
    err = cudaMalloc((void**)&d_latency_out, totalThreads_global * sizeof(unsigned int));
    if (err != cudaSuccess) {
         fprintf(stderr, "Failed to allocate device memory for latency output (error: %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }
    cudaMemset(d_latency_out, 0, totalThreads_global * sizeof(unsigned int));
#endif

    // Build output filename(s).
#if defined(USE_RANDOM_ACCESS)
    const char* access_pattern = "random";
#elif defined(USE_STREAM_ACCESS)
    const char* access_pattern = "stream";
#elif defined(USE_STRIDED_ACCESS)
    const char* access_pattern = "strided";
#endif
    char rand_delay_str[64] = "";
#ifdef ENABLE_RANDOM_DELAY
    #ifdef WARP_RANDOM_DELAY
         sprintf(rand_delay_str, "_rand_delay_warp_%d", RANDOM_DELAY_STEPS);
    #else
         sprintf(rand_delay_str, "_rand_delay_thread_%d", RANDOM_DELAY_STEPS);
    #endif
#endif
    // Distribution file name: includes threads_per_CTA.
    char dist_filename[64];
    sprintf(dist_filename, "%u_%u_%s%s.log", threads_per_CTA, CTAs_per_SM, access_pattern, rand_delay_str);
    // Average file name: based only on access_pattern and CTAs (aggregated for different thread counts).
    char avg_filename[64];
    sprintf(avg_filename, "%s_%u%s.log", access_pattern, CTAs_per_SM, rand_delay_str);
    
    // Set grid and block dimensions.
    dim3 grid(numSM * CTAs_per_SM);
    dim3 block(threads_per_CTA);
    
    // Warm-up.
    k<<<grid, block>>>(d_data, data_size, 1);
    cudaDeviceSynchronize();
    
    // Allocate dynamic shared memory (one unsigned int per thread).
    size_t sharedMemSize = threads_per_CTA * sizeof(unsigned int);
#ifdef USE_RANDOM_ACCESS
    #ifdef ENABLE_LATENCY_MEASUREMENT
         kernel<<<grid, block, sharedMemSize>>>(d_data, d_randAddrs, d_latency_out, data_size);
    #else
         kernel<<<grid, block, sharedMemSize>>>(d_data, d_randAddrs, NULL, data_size);
    #endif
#elif defined(USE_STREAM_ACCESS) || defined(USE_STRIDED_ACCESS)
    #ifdef ENABLE_LATENCY_MEASUREMENT
         kernel<<<grid, block, sharedMemSize>>>(d_data, d_latency_out, data_size);
    #else
         kernel<<<grid, block, sharedMemSize>>>(d_data, NULL, data_size);
    #endif
#endif
    err = cudaGetLastError();
    if (err != cudaSuccess) {
         fprintf(stderr, "Failed to launch kernel (error: %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    
#ifdef ENABLE_LATENCY_MEASUREMENT
    unsigned int totalThreads = numSM * CTAs_per_SM * threads_per_CTA;
    unsigned int* h_latency_out = (unsigned int*)malloc(totalThreads * sizeof(unsigned int));
    err = cudaMemcpy(h_latency_out, d_latency_out, totalThreads * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
         fprintf(stderr, "Failed to copy latency measurements (error: %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }
    
    // If output_mode == 'a' or 'b', write average latency.
    if (output_mode == 'a' || output_mode == 'b') {
        double sum = 0.0;
        for (unsigned int i = 0; i < totalThreads; i++) {
            sum += h_latency_out[i];
        }
        double average = sum / totalThreads;
        // Open the average file in append mode.
        FILE *fpAvg = fopen(avg_filename, "a");
        if (!fpAvg) {
            fprintf(stderr, "Failed to open average output file %s\n", avg_filename);
            exit(EXIT_FAILURE);
        }
        // Write the thread count and its corresponding average latency.
        fprintf(fpAvg, "%f\n", average);
        fclose(fpAvg);
    }
    // If output_mode == 'd' or 'b', write full distribution.
    if (output_mode == 'd' || output_mode == 'b') {
        FILE *fpDist = fopen(dist_filename, "w");
        if (!fpDist) {
            fprintf(stderr, "Failed to open distribution output file %s\n", dist_filename);
            exit(EXIT_FAILURE);
        }
        for (unsigned int i = 0; i < totalThreads; i++) {
            fprintf(fpDist, "%u\n", h_latency_out[i]);
        }
        fclose(fpDist);
    }
    
    free(h_latency_out);
#endif
    
    // Cleanup.
    cudaFree(d_data);
#ifdef USE_RANDOM_ACCESS
    cudaFree(d_randAddrs);
#endif
#ifdef ENABLE_LATENCY_MEASUREMENT
    cudaFree(d_latency_out);
#endif
    return 0;
}
