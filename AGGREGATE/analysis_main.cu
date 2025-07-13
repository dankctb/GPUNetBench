#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <iomanip> 
#include <cstdlib>
#include <ctime>


// find largest power of 2
unsigned flp2(unsigned x) {
    x = x| (x>>1);
    x = x| (x>>2);
    x = x| (x>>4);
    x = x| (x>>8);
    x = x| (x>>16);
    return x - (x>>1);
}

float mean(float arr[], int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}

using mt = unsigned long long;

__global__ void bandwidthKernel(mt *data, int data_len, int loopCount) {
    volatile int sum = 0;
    volatile int value = 0;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Loop over the number of iterations to measure bandwidth.
    for (int l = 0; l < loopCount; l++) {
        if (stride < data_len) {
            for (int i = idx; i < data_len; i += stride) {
                //int wrapped_idx = i % len;  // Wrap around the index.
                sum += __ldcg(&data[i]);  // Read using __ldcg intrinsic.
                value += sum;
            }
        }
        else {
            // if number of threads is larger than data_len
            // Each thread accesses one element with index wrapping
            int access_idx = idx % data_len;
            sum += __ldcg(&data[access_idx]);  // Read using __ldcg intrinsic.
            value += sum;
        }
    }
    
    // Write back the sum to global memory to avoid optimization removal.
    int access_idx = idx % data_len;
    data[access_idx] = value;
}

int main(int argc, char** argv) {
    // Check for required command-line arguments.
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <CTA> <WARP> <ITERATION> <loopCount> <sizeMultiple>" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Parse command-line parameters.
    int CTA         = std::atoi(argv[1]);  // Number of thread blocks per SM.
    int WARP        = std::atoi(argv[2]);  // Number of warps per thread block.
    int ITERATION   = std::atoi(argv[3]);  // Number of iterations for measurement.
    int loopCount   = std::atoi(argv[4]);  // Number of loops executed inside the kernel.
    int sizeMultiple = std::atoi(argv[5]); // Multiplier for L2 cache size to set data transfer size.

    // Set up CUDA device.
    (cudaSetDevice(0));
    cudaDeviceProp prop;
    (cudaGetDeviceProperties(&prop, 0));

    // Set a function attribute for the kernel (preferred shared memory carveout).
    (cudaFuncSetAttribute(bandwidthKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    // Determine the number of SMs and the L2 cache size.
    const int numSM = prop.multiProcessorCount;
    const unsigned l2CacheSize = prop.l2CacheSize;  // In bytes.

    // Determine target data size: sizeMultiple times the L2 cache size.
    const unsigned targetSize = sizeMultiple * l2CacheSize;
    // Compute total bytes rounded up to the next power of 2.
    unsigned long long totalBytes = flp2(targetSize);
    // Determine number of elements for type mt.
    unsigned long long numElements = totalBytes / sizeof(mt);
    std::cout << "numElements: " << numElements << std::endl;
    std::cout << "size of each element: " << sizeof(mt) << std::endl;

    // Configure kernel launch parameters:
    //   - Each block has (32 * WARP) threads.
    //   - Total number of blocks is (numSM * CTA).
    int threadsPerBlock = 32 * WARP; 
    int blocks = numSM * CTA; 

    // Print kernel configuration information
    int threadsPerSM = threadsPerBlock * CTA;
    std::cout << "block per SM: " << CTA << std::endl;
    std::cout << "WARP per block: " << WARP << std::endl;
    std::cout << "----" << std::endl;
    std::cout << "Threads per SM: " << threadsPerSM << std::endl;
    std::cout << "Threads per block: " << threadsPerBlock << std::endl;
    std::cout << "total Threads: " << threadsPerSM * numSM << std::endl;

    // Allocate device memory.
    mt *d_data;
    (cudaMalloc(&d_data, numElements * sizeof(mt)));


    // Allocate array to store bandwidth measurements.
    float* bandwidthMeasurements = new float[ITERATION];
    
    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    (cudaEventCreate(&start));
    (cudaEventCreate(&stop));

    // Run the kernel for the specified number of iterations.
    for (int iter = 0; iter < ITERATION; iter++) {

        (cudaEventRecord(start));
        bandwidthKernel<<<blocks, threadsPerBlock>>>(d_data, numElements, loopCount);

        (cudaEventRecord(stop));

        // Ensure the kernel has completed.
        (cudaDeviceSynchronize());
        (cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        (cudaEventElapsedTime(&milliseconds, start, stop));
        std::cout << "Iteration " << iter << " time: " << milliseconds << " ms" << std::endl;
        // Calculate effective bandwidth in GB/s.
        // Total bytes read = numElements * sizeof(mt) * loopCount.
        // Divide by (milliseconds * 1e6) to convert time and bytes into GB/s.
        float bandwidth = ((numElements) * sizeof(mt) * loopCount) / (milliseconds * 1e6f);
        bandwidthMeasurements[iter] = bandwidth;
    }

    // Compute and output the average bandwidth.
    float avgBandwidth = mean(bandwidthMeasurements, ITERATION);
    std::cout << std::fixed << std::setprecision(2)
              << "Average bandwidth: " << avgBandwidth << " GB/s" << std::endl;
    std::cout << "" << std::endl;
    // Cleanup resources.
    (cudaEventDestroy(start));
    (cudaEventDestroy(stop));
    (cudaFree(d_data));
    delete[] bandwidthMeasurements;

    return EXIT_SUCCESS;
}