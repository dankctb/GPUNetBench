#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/barrier>
#include <cooperative_groups.h>

#define DATA_SIZE (1024 * 1024)

namespace cg = cooperative_groups;

// DSM kernel with variable bin size
__global__ void histogram_h100_dsm_kernel(const int* data, int* histogram, int n, int bins, int cluster_size) {
    auto cta = cg::this_thread_block();
    auto cluster = cg::this_cluster();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    extern __shared__ int shared_hist[];
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    cta.sync();
    
    // Process data with cluster cooperation
    for (int i = tid; i < n; i += stride) {
        int bin = data[i] % bins;
        atomicAdd(&shared_hist[bin], 1);
    }
    cta.sync();
    
    // Each block contributes its histogram to global memory
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

// Non-DSM kernel with variable bin size
__global__ void histogram_h100_kernel(const int* data, int* histogram, int n, int bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    extern __shared__ int shared_hist[];
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Process data
    for (int i = tid; i < n; i += stride) {
        int bin = data[i] % bins;
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();
    
    // Each block contributes its histogram to global memory
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

extern "C" void launch_histogram_h100_dsm(const int* d_data, int* d_histogram, int n, int bins, int cluster_size) {
    dim3 block(256);
    dim3 grid(cluster_size);
    
    void* args[] = {(void*)&d_data, (void*)&d_histogram, (void*)&n, (void*)&bins, (void*)&cluster_size};
    
    cudaLaunchCooperativeKernel(
        (void*)histogram_h100_dsm_kernel,
        grid, block, args,
        bins * sizeof(int)
    );
    cudaDeviceSynchronize();
}

extern "C" void launch_histogram_h100(const int* d_data, int* d_histogram, int n, int bins) {
    dim3 block(256);
    dim3 grid(32);  // Standard grid size
    
    histogram_h100_kernel<<<grid, block, bins * sizeof(int)>>>(d_data, d_histogram, n, bins);
    cudaDeviceSynchronize();
} 