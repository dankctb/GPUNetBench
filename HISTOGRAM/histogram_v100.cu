#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BINS 2048
#define DATA_SIZE (1024 * 1024)  // 1M elements, ~4MB fits in V100 L2 cache

__global__ void histogram_v100_kernel(const int* data, int* histogram, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Shared memory histogram for each block
    __shared__ int shared_hist[BINS];
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < BINS; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Process data elements
    for (int i = tid; i < n; i += stride) {
        int bin = data[i] % BINS;
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();
    
    // Merge to global histogram
    for (int i = threadIdx.x; i < BINS; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

extern "C" void launch_histogram_v100(const int* d_data, int* d_histogram, int n) {
    dim3 block(256);
    dim3 grid(min(65535, (n + block.x - 1) / block.x));
    
    histogram_v100_kernel<<<grid, block>>>(d_data, d_histogram, n);
    cudaDeviceSynchronize();
} 