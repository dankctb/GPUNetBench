#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#define DATA_SIZE (1024 * 1024)
#define HISTOGRAM256_BIN_COUNT 256
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)
#define WARP_COUNT 6
#define HISTOGRAM256_THREADBLOCK_SIZE (WARP_COUNT * WARP_SIZE)
#define HISTOGRAM256_THREADBLOCK_MEMORY (WARP_COUNT * HISTOGRAM256_BIN_COUNT)
#define CLUSTER_SIZE 8
#define NUM_CLUSTERS 16
#define PARTIAL_HISTOGRAM256_COUNT (NUM_CLUSTERS * CLUSTER_SIZE)

namespace cg = cooperative_groups;
typedef unsigned int uint;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

static uint *d_PartialHistograms;

inline __device__ void addByte(uint *s_WarpHist, uint data) { 
    atomicAdd(s_WarpHist + data, 1); 
}

inline __device__ void addWord(uint *s_WarpHist, uint data) {
    addByte(s_WarpHist, (data >> 0) & 0xFFU);
    addByte(s_WarpHist, (data >> 8) & 0xFFU);
    addByte(s_WarpHist, (data >> 16) & 0xFFU);
    addByte(s_WarpHist, (data >> 24) & 0xFFU);
}

__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void histogram256OptimizedDSMEMKernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block cta = cg::this_thread_block();
    
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint *s_WarpHist = s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;
    
    // Map shared memory to owner SM (rank 0 in cluster)
    uint *owner_hist = cluster.map_shared_rank(s_Hist, 0);
    uint *owner_warp_hist = owner_hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    // Initialize histograms in owner SM
    if (cluster.block_rank() == 0) {
        #pragma unroll
        for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++) {
            s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
        }
    }
    cluster.sync();

    // Process data using optimized word-level approach, writing to owner SM
    for (uint pos = (blockIdx.x * blockDim.x + threadIdx.x); pos < dataCount; pos += blockDim.x * gridDim.x) {
        uint data = d_Data[pos];
        addWord(owner_warp_hist, data);
    }
    cluster.sync();

    // Merge results in owner SM and write to global memory
    if (cluster.block_rank() == 0) {
        for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE) {
            uint sum = 0;
            for (uint i = 0; i < WARP_COUNT; i++) {
                sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT];
            }
            d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
        }
    }
}

__global__ void mergeHistogram256Kernel(uint *d_Histogram, uint *d_PartialHistograms, uint histogramCount) {
    cg::thread_block cta = cg::this_thread_block();
    uint sum = 0;

    for (uint i = threadIdx.x; i < histogramCount; i += blockDim.x) {
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];
    }

    __shared__ uint data[256];
    data[threadIdx.x] = sum;

    for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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
                         PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
}

void closeHistogram256() {
    CUDA_CHECK(cudaFree(d_PartialHistograms));
}

void histogram256(uint *d_Histogram, void *d_Data, uint byteCount) {
    histogram256OptimizedDSMEMKernel<<<PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_THREADBLOCK_SIZE>>>(
        d_PartialHistograms, (uint *)d_Data, byteCount / sizeof(uint));
    CUDA_CHECK(cudaGetLastError());

    mergeHistogram256Kernel<<<HISTOGRAM256_BIN_COUNT, 256>>>(
        d_Histogram, d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT);
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
    std::vector<uint> h_data(DATA_SIZE);
    generate_data(h_data);
    
    uint *d_data, *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_data, DATA_SIZE * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), DATA_SIZE * sizeof(uint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    
    initHistogram256();
    
    // Enable cluster launch
    CUDA_CHECK(cudaFuncSetAttribute((const void*)histogram256OptimizedDSMEMKernel, 
                                   cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    histogram256(d_histogram, d_data, DATA_SIZE * sizeof(uint));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    std::cout << "Optimized DSMEM Histogram execution time: " << milliseconds << " ms" << std::endl;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    closeHistogram256();
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histogram));
    
    return 0;
} 