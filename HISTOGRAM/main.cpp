#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define BINS 2048
#define DATA_SIZE (1024 * 1024)

// Conditional function declarations based on compilation target
#ifdef COMPILE_V100
extern "C" void launch_histogram_v100(const int* d_data, int* d_histogram, int n);
#endif

#ifdef COMPILE_H100
extern "C" void launch_histogram_h100(const int* d_data, int* d_histogram, int n, int cluster_size);
#endif

void generate_data(std::vector<int>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, BINS * 4);
    
    for (int& val : data) {
        val = dis(gen);
    }
}

bool verify_histogram(const std::vector<int>& histogram, const std::vector<int>& data) {
    std::vector<int> cpu_hist(BINS, 0);
    for (int val : data) {
        cpu_hist[val % BINS]++;
    }
    
    for (int i = 0; i < BINS; i++) {
        if (histogram[i] != cpu_hist[i]) {
            std::cout << "Mismatch at bin " << i << ": GPU=" << histogram[i] 
                      << " CPU=" << cpu_hist[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <mode>" << std::endl;
        std::cout << "mode: v100 | h100_c{cluster_size}" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    
    // Generate test data
    std::vector<int> h_data(DATA_SIZE);
    generate_data(h_data);
    
    // Allocate GPU memory
    int *d_data, *d_histogram;
    cudaMalloc(&d_data, DATA_SIZE * sizeof(int));
    cudaMalloc(&d_histogram, BINS * sizeof(int));
    
    cudaMemcpy(d_data, h_data.data(), DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, BINS * sizeof(int));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    if (mode == "v100") {
#ifdef COMPILE_V100
        std::cout << "Running V100 histogram..." << std::endl;
        launch_histogram_v100(d_data, d_histogram, DATA_SIZE);
#else
        std::cout << "V100 support not compiled in this build!" << std::endl;
        return 1;
#endif
    } else if (mode.substr(0, 4) == "h100") {
#ifdef COMPILE_H100
        int cluster_size = std::stoi(mode.substr(6));
        std::cout << "Running H100 histogram with cluster size " << cluster_size << "..." << std::endl;
        launch_histogram_h100(d_data, d_histogram, DATA_SIZE, cluster_size);
#else
        std::cout << "H100 support not compiled in this build!" << std::endl;
        return 1;
#endif
    } else {
        std::cout << "Invalid mode: " << mode << std::endl;
        return 1;
    }
    
    // Record stop event and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back and verify
    std::vector<int> h_histogram(BINS);
    cudaMemcpy(h_histogram.data(), d_histogram, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    if (verify_histogram(h_histogram, h_data)) {
        std::cout << "✓ Histogram computation successful!" << std::endl;
        std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
        
        // Print some statistics
        int total = 0;
        for (int count : h_histogram) total += count;
        std::cout << "Total elements processed: " << total << std::endl;
        
        // Calculate throughput
        float throughput = (DATA_SIZE / 1e6) / (milliseconds / 1000);
        std::cout << "Throughput: " << throughput << " Million elements/sec" << std::endl;
    } else {
        std::cout << "✗ Histogram verification failed!" << std::endl;
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_histogram);
    
    return 0;
} 