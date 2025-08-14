#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <assert.h>

#define DATA_SIZE (1024 * 1024)
#define HISTOGRAM256_BIN_COUNT 256
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)
#define WARP_COUNT 6
#define HISTOGRAM256_THREADBLOCK_SIZE (WARP_COUNT * WARP_SIZE)
#define HISTOGRAM256_THREADBLOCK_MEMORY (WARP_COUNT * HISTOGRAM256_BIN_COUNT)
#define TAG_MASK 0xFFFFFFFFU
#define UMUL(a, b) ((a) * (b))
#define UMAD(a, b, c) (UMUL((a), (b)) + (c))
#define MERGE_THREADBLOCK_SIZE 256
#define CLUSTER_SIZE 8
#define NUM_CLUSTERS 16
#define PARTIAL_HISTOGRAM256_COUNT (NUM_CLUSTERS * CLUSTER_SIZE)

namespace cg = cooperative_groups;
typedef unsigned int uint;

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Optimized histogram functions
static uint *d_PartialHistograms;

inline __device__ void addByte(uint *s_WarpHist, uint data, uint threadTag) { 
    atomicAdd(s_WarpHist + data, 1); 
}

inline __device__ void addWord(uint *s_WarpHist, uint data, uint tag) {
    addByte(s_WarpHist, (data >> 0) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 8) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void histogram256DSMEMKernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block cta = cg::this_thread_block();
    
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint *s_WarpHist = s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    // Initialize shared memory
    #pragma unroll
    for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++) {
        s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
    }

    const uint tag = threadIdx.x << (32 - LOG2_WARP_SIZE);
    cg::sync(cta);

    // Process data - each warp computes its histogram
    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x)) {
        uint data = d_Data[pos];
        addWord(s_WarpHist, data, tag);
    }
    cg::sync(cta);

    // Block 0 in cluster directly reads all warp histograms via DSMEM
    if (cluster.block_rank() == 0) {
        cluster.sync(); // Ensure all blocks finished warp-level computation
        
        // Direct merge from all warp histograms across all blocks in cluster
        for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE) {
            uint sum = 0;
            
            // Sum across all blocks in cluster
            #pragma unroll
            for (uint block_id = 0; block_id < CLUSTER_SIZE; block_id++) {
                uint *remote_hist = cluster.map_shared_rank(s_Hist, block_id);
                
                // Sum across all warps in this block
                #pragma unroll
                for (uint i = 0; i < WARP_COUNT; i++) {
                    sum += remote_hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;
                }
            }
            
            // Global atomic add from cluster representative
            uint cluster_id = blockIdx.x / CLUSTER_SIZE;
            atomicAdd(&d_PartialHistograms[cluster_id * HISTOGRAM256_BIN_COUNT + bin], sum);
        }
    }
    cluster.sync();
}

__global__ void mergeHistogram256Kernel(uint *d_Histogram, uint *d_PartialHistograms, uint histogramCount) {
    cg::thread_block cta = cg::this_thread_block();
    uint sum = 0;

    for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE) {
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];
    }

    __shared__ uint data[MERGE_THREADBLOCK_SIZE];
    data[threadIdx.x] = sum;

    for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        cg::sync(cta);
        if (threadIdx.x < stride) {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0) {
        d_Histogram[blockIdx.x] = data[0];
    }
}

void initHistogram256() {
    CUDA_CHECK(cudaMalloc((void **)&d_PartialHistograms, 
                         NUM_CLUSTERS * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
}

void closeHistogram256() {
    CUDA_CHECK(cudaFree(d_PartialHistograms));
}

void histogram256(uint *d_Histogram, void *d_Data, uint byteCount) {
    assert(byteCount % sizeof(uint) == 0);
    
    // Clear partial histograms
    CUDA_CHECK(cudaMemset(d_PartialHistograms, 0, NUM_CLUSTERS * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    
    histogram256DSMEMKernel<<<PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_THREADBLOCK_SIZE>>>(
        d_PartialHistograms, (uint *)d_Data, byteCount / sizeof(uint));
    CUDA_CHECK(cudaGetLastError());

    mergeHistogram256Kernel<<<HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(
        d_Histogram, d_PartialHistograms, NUM_CLUSTERS);
    CUDA_CHECK(cudaGetLastError());
}

void generate_data(std::vector<uint>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint> dis(0, HISTOGRAM256_BIN_COUNT - 1);
    for (uint& val : data) {
        val = dis(gen);
    }
}

int main() {
    // Generate test data
    std::vector<uint> h_data(DATA_SIZE);
    generate_data(h_data);
    
    // Allocate GPU memory
    uint *d_data, *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_data, DATA_SIZE * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    
    // Copy data to device and initialize histogram
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), DATA_SIZE * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    
    // Initialize histogram
    initHistogram256();
    
    // Enable cluster launch
    CUDA_CHECK(cudaFuncSetAttribute((const void*)histogram256DSMEMKernel, 
                                   cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)histogram256DSMEMKernel, 
                                   cudaFuncAttributeRequiredClusterWidth, CLUSTER_SIZE));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)histogram256DSMEMKernel, 
                                   cudaFuncAttributeRequiredClusterHeight, 1));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)histogram256DSMEMKernel, 
                                   cudaFuncAttributeRequiredClusterDepth, 1));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start event
    CUDA_CHECK(cudaEventRecord(start));
    
    // Execute histogram kernel
    histogram256(d_histogram, d_data, DATA_SIZE * sizeof(uint));
    
    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    std::cout << "DSMEM Histogram execution time: " << milliseconds << " ms" << std::endl;
    
    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Cleanup
    closeHistogram256();
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histogram));
    
    return 0;
} 