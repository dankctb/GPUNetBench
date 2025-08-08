#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/barrier>
#include <cooperative_groups.h>

#define BINS 2048
#define DATA_SIZE (1024 * 1024)  // Same data size for fair comparison

namespace cg = cooperative_groups;

__global__ void histogram_h100_kernel(const int* data, int* histogram, int n, int cluster_size) {
    auto cta = cg::this_thread_block();
    auto cluster = cg::this_cluster();
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Distributed shared memory across cluster
    extern __shared__ int shared_hist[];
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < BINS; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    cta.sync();
    
    // Process data with cluster cooperation
    for (int i = tid; i < n; i += stride) {
        int bin = data[i] % BINS;
        atomicAdd(&shared_hist[bin], 1);
    }
    cta.sync();
    
    // Cluster-wide reduction using DSM
    if (cluster.thread_rank() < BINS) {
        int local_count = shared_hist[cluster.thread_rank()];
        int global_count = cg::reduce(cluster, local_count, cg::plus<int>());
        
        if (cluster.block_rank() == 0) {
            atomicAdd(&histogram[cluster.thread_rank()], global_count);
        }
    }
}

extern "C" void launch_histogram_h100(const int* d_data, int* d_histogram, int n, int cluster_size) {
    dim3 block(256);
    dim3 grid(cluster_size);  // Single GPC, varied cluster size
    
    // Launch with cluster cooperation
    void* args[] = {(void*)&d_data, (void*)&d_histogram, (void*)&n, (void*)&cluster_size};
    
    cudaLaunchCooperativeKernel(
        (void*)histogram_h100_kernel,
        grid, block, args,
        BINS * sizeof(int)  // Shared memory size
    );
    cudaDeviceSynchronize();
} 