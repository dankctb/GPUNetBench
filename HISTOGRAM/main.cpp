#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define DATA_SIZE (1024 * 1024)

extern "C" void launch_histogram_h100_dsm(const int* d_data, int* d_histogram, int n, int bins, int cluster_size);
extern "C" void launch_histogram_h100(const int* d_data, int* d_histogram, int n, int bins);

void generate_data(std::vector<int>& data, int bins) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, bins * 4);
    
    for (int& val : data) {
        val = dis(gen);
    }
}

float measure_time(void (*launch_func)(const int*, int*, int, int), const int* d_data, int* d_histogram, int bins) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMemset(d_histogram, 0, bins * sizeof(int));
    
    cudaEventRecord(start);
    launch_func(d_data, d_histogram, DATA_SIZE, bins);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

float measure_time_dsm(const int* d_data, int* d_histogram, int bins, int cluster_size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMemset(d_histogram, 0, bins * sizeof(int));
    
    cudaEventRecord(start);
    launch_histogram_h100_dsm(d_data, d_histogram, DATA_SIZE, bins, cluster_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    // Test different bin sizes that exceed SM shared memory H100 (256KB = ~64K ints)
    std::vector<int> bin_sizes = {16384, 32768, 65536, 131072, 262144, 524288}; // 16K, 32K, 64K, 128K, 256K, 512K
    std::vector<int> cluster_sizes = {2, 4, 8, 16};
    
    std::cout << "=== H100 Histogram Performance Test ===" << std::endl;
    std::cout << "Data size: " << DATA_SIZE << " elements" << std::endl << std::endl;
    
    for (int bins : bin_sizes) {
        std::cout << "Bin size: " << bins << " (Shared memory: " << bins * 4 / 1024 << " KB)" << std::endl;
        
        // Generate test data
        std::vector<int> h_data(DATA_SIZE);
        generate_data(h_data, bins);
        
        // Allocate GPU memory
        int *d_data, *d_histogram;
        cudaMalloc(&d_data, DATA_SIZE * sizeof(int));
        cudaMalloc(&d_histogram, bins * sizeof(int));
        cudaMemcpy(d_data, h_data.data(), DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
        
        // Test non-DSM
        float time_no_dsm = measure_time(launch_histogram_h100, d_data, d_histogram, bins);
        std::cout << "  No DSM:     " << time_no_dsm << " ms" << std::endl;
        
        // Test different cluster sizes with DSM
        for (int cluster_size : cluster_sizes) {
            float time_dsm = measure_time_dsm(d_data, d_histogram, bins, cluster_size);
            float speedup = time_no_dsm / time_dsm;
            std::cout << "  DSM C" << cluster_size << ":     " << time_dsm << " ms (speedup: " << speedup << "x)" << std::endl;
        }
        
        cudaFree(d_data);
        cudaFree(d_histogram);
        std::cout << std::endl;
    }
    
    return 0;
} 