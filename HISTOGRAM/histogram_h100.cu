#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#define DATA_SIZE (1024 * 1024)

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input,
                                   size_t array_size)
{
  extern __shared__ int smem[];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();

  // Cluster initialization, size and calculating local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0; //Initialize shared memory histogram to zeros
  }

  // cluster synchronization ensures that shared memory is initialized to zero in
  // all thread blocks in the cluster. It also ensures that all thread blocks
  // have started executing and they exist concurrently.
  cluster.sync();

  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    int ldata = input[i];

    //Find the right histogram bin.
    int binid = ldata;
    if (ldata < 0)
      binid = 0;
    else if (ldata >= nbins)
      binid = nbins - 1;

    //Find destination block rank and offset for computing
    //distributed shared memory histogram
    int dst_block_rank = (int)(binid / bins_per_block);
    int dst_offset = binid % bins_per_block;

    //Pointer to target block shared memory
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    //Perform atomic update of the histogram bin
    atomicAdd(dst_smem + dst_offset, 1);
  }

  // cluster synchronization is required to ensure all distributed shared
  // memory operations are completed and no thread block exits while
  // other thread blocks are still accessing distributed shared memory
  cluster.sync();

  // Perform global memory histogram, using the local distributed memory histogram
  int *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
}

// Non-DSM histogram kernel
__global__ void histogram_kernel(int *bins, const int nbins, const int *__restrict__ input, size_t array_size)
{
    extern __shared__ int smem[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        smem[i] = 0;
    }
    __syncthreads();
    
    // Process data
    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
        int ldata = input[i];
        int binid = (ldata < 0) ? 0 : ((ldata >= nbins) ? nbins - 1 : ldata);
        atomicAdd(&smem[binid], 1);
    }
    __syncthreads();
    
    // Write to global memory 
    for (int i = threadIdx.x; i < nbins; i += blockDim.x) {
        atomicAdd(&bins[i], smem[i]);
    }
}

void generate_data(std::vector<int>& data, int bins) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, bins * 4);
    for (int& val : data) {
        val = dis(gen);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <bin_size> <cluster_size>" << std::endl;
        return 1;
    }
    
    int bins = std::atoi(argv[1]);
    int cluster_size = std::atoi(argv[2]);
    
    // Generate test data
    std::vector<int> h_data(DATA_SIZE);
    generate_data(h_data, bins);
    
    // Allocate GPU memory
    int *d_data, *d_histogram;
    cudaMalloc(&d_data, DATA_SIZE * sizeof(int));
    cudaMalloc(&d_histogram, bins * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, bins * sizeof(int));
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch kernel based on cluster size
    cudaEventRecord(start);
    
    if (cluster_size == 0) {
        // Non-DSM case
        int threads = 256;
        int blocks = std::min(32, (DATA_SIZE + threads - 1) / threads);
        size_t smem_size = bins * sizeof(int);
        size_t array_size = DATA_SIZE;
        histogram_kernel<<<blocks, threads, smem_size>>>(d_histogram, bins, d_data, array_size);
    } else {
        // DSM usage
        int threads = 256;
        int bins_per_block = (bins + cluster_size - 1) / cluster_size;
        size_t smem_size = bins_per_block * sizeof(int);
        size_t array_size = DATA_SIZE;
        
        cudaLaunchConfig_t config = {0};
        config.gridDim = dim3(cluster_size, 1, 1);
        config.blockDim = dim3(threads, 1, 1);
        config.dynamicSmemBytes = smem_size;
        
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = cluster_size;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;
        
        void* args[] = {&d_histogram, &bins, &bins_per_block, &d_data, &array_size};
        cudaLaunchKernelEx(&config, (void*)clusterHist_kernel, args, 5, &attr, 1);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    std::cout << bins << " " << cluster_size << " " << ms << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_histogram);
    
    return 0;
}