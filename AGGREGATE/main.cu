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

__global__ void bandwidthKernel(mt *data, int data_len, int L2_access_len, int loopCount) {
    volatile int sum = 0;
    volatile int value = 0;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x; // total number of threads in the grid

    // Loop over the number of loopCount to measure bandwidth.
    // 2nd, 3rd, … access of the same line come from L2, but the very 1st access from HBM.
    for (int l = 0; l < loopCount; l++) {
        for (int i = idx; i < L2_access_len; i += stride) {
            int access_idx = i;
            if (i >= data_len){
                access_idx = i % data_len;
            }
            sum += __ldcg(&data[access_idx]);  // Read using __ldcg intrinsic.
            value += sum;
        }
    }
    
    // Write back the sum to global memory to avoid optimization removal.
    data[idx % data_len] = value;
}

int main(int argc, char** argv) {
    // Check for required command-line arguments.
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <CTA> <WARP> <ITERATION> <loopCount> <sizeMultiple>" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Parse command-line parameters.
    int CTA         = std::atoi(argv[1]);  // Number of thread blocks per SM. 1-32
    int WARP        = std::atoi(argv[2]);  // Number of warps per thread block. 1-32
    int ITERATION   = std::atoi(argv[3]);  // Number of iterations for measurement.
    int loopCount   = std::atoi(argv[4]);  // Number of loops executed inside the kernel.
    int sizeMultiple = std::atoi(argv[5]); // Multiplier for L2 cache size to set data transfer size.
    int numL2Access = std::atoi(argv[6]); // Number of L2 accesses.
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
    
    // Configure kernel launch parameters:
    //   - Each block has (32 * WARP) threads.
    //   - Total number of blocks is (numSM * CTA).
    int threadsPerBlock = 32 * WARP; 
    int blocks = numSM * CTA; 

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
        bandwidthKernel<<<blocks, threadsPerBlock>>>(d_data, numElements, numL2Access, loopCount);
        cudaError_t kernelError = cudaGetLastError();
        if (kernelError != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelError) << std::endl;
            return EXIT_FAILURE;
        }
        (cudaEventRecord(stop));

        // Ensure the kernel has completed.
        (cudaDeviceSynchronize());
        (cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        (cudaEventElapsedTime(&milliseconds, start, stop));

        // Calculate effective bandwidth in GB/s.
        // Total bytes read = numElements * sizeof(mt) * loopCount.
        // Divide by (milliseconds * 1e6) to convert time and bytes into GB/s.
        float bandwidth = ((numL2Access) * sizeof(mt) * loopCount) / (milliseconds * 1e6f);
        bandwidthMeasurements[iter] = bandwidth;
    }

    // Compute and output the average bandwidth.
    float avgBandwidth = mean(bandwidthMeasurements, ITERATION);
    std::cout << std::fixed << std::setprecision(2)
              <<  avgBandwidth << std::endl;

    // Cleanup resources.
    (cudaEventDestroy(start));
    (cudaEventDestroy(stop));
    (cudaFree(d_data));
    delete[] bandwidthMeasurements;

    return EXIT_SUCCESS;
}