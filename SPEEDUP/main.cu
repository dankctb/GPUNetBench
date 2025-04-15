/*
    OVERVIEW:
    ----------

    This experiment selects one or more Graphics Processing Clusters (GPCs)
    and runs a CUDA kernel on a unified list of Streaming Multiprocessors (SMs)
    derived from the selected GPCs. The program measures performance by
    executing a kernel that either reads from or writes to a global data buffer.
    The goal of this experiment is to evaluate the bandwidth speedup of TPC and GPC
    hierarchies in NVIDIA GPUs (V100, A100, H100).
    
    HOW IT WORKS:
    - The program accepts the following command line arguments:
         <CTA> <WARP> <ITERATION> <GPCselectedList> <SMmax>
      where:
         • CTA             : Number of thread blocks per SM (CTAs per SM).
         • WARP            : Number of warps per thread block.
         • ITERATION       : Number of kernel iterations for performance measurement.
         • GPCselectedList : Comma-separated list of GPC IDs (e.g., "0,1,3"). (Allowed values 0 to 6)
         • SMmax           : Maximum number of SMs to be activated from the unified SM list.
                           If SMmax is less than the total number of SM IDs, the list is truncated.
    
    - The unified SM list is constructed by copying pre-defined SM ID mappings from the selected GPCs.
    - The SM IDs are then sorted in ascending order (or with a specialized even/odd sort if SORT_TPC is defined).
    - A global data buffer is allocated based on the device's L2 cache size.
    - The kernel (function "k") is launched with CTA block per SM and performs either read-based
      operations (using __ldcg) when USE_READ is defined, or write-based operations otherwise.
    - After all iterations, the program computes an average bandwidth value and prints the result.
    

    Note:
         - Define USE_READ at compile time to enable read-based operations.
         - Define USE_A100 or USE_H100 to select the appropriate GPC mappings.
         - Define SORT_GPC to sort the SM IDs to avoid TPC contention.
         - Adjust CTA and WARP according to the desired workload per SM.
*/

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

// Define the type for our data elements.
using mt = unsigned long long;

//------------------------------------------------------------------------------
// Utility Functions
//------------------------------------------------------------------------------

// Returns the largest power of 2 less than or equal to x.
unsigned flp2(unsigned x) {
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x - (x >> 1);
}

// Computes the arithmetic mean of an array.
float mean(const float arr[], int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}

//------------------------------------------------------------------------------
// Device Function: Retrieve SM ID
//------------------------------------------------------------------------------

__device__ unsigned int get_smid(void) {
    unsigned int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

//------------------------------------------------------------------------------
// Kernel Function
//------------------------------------------------------------------------------
//
// Performs read or write operations on a global data buffer based on
// compile-time flag USE_READ.
__global__ void kernel(mt *d, int len, int loops_per_sm,
                  const unsigned int *sm_ids,
                  const unsigned int active_sms,
                  const unsigned int GPCsize) {
    const unsigned int sm_id = get_smid();
    volatile unsigned int accumulation = 0;

    // Each SM in the unified list performs its loops.
    for (unsigned int j = 0; j < active_sms; j++) {
        if (sm_id == sm_ids[j]) {
            for (int l = 0; l < loops_per_sm; l++) {
                // Stride across the buffer by GPCsize * blockDim.x
                for (int i = threadIdx.x + blockDim.x * j; i < len; i += GPCsize * blockDim.x) {
#ifdef USE_READ
                    // Read mode: two cached reads per element
                    for (int it = 0; it < 2; it++) {
                        accumulation += __ldcg(&d[i]);
                    }
#else
                    // Write mode
                    d[i] = threadIdx.x + l + sm_id;
#endif
                }
            }
#ifdef USE_READ
            // Store accumulation to prevent optimization
            d[sm_id * 128] = accumulation;
#endif
        }
    }

    __syncthreads();
}

//------------------------------------------------------------------------------
// GPC Mapping Definitions
//------------------------------------------------------------------------------
//
// Select one of the following mapping sets at compile time:
//   - USE_A100 → A100 mappings (7 GPCs: sizes 16,16,16,16,16,14,14)
//   - USE_H100 → H100 mappings (2 GPCs: sizes 16,18)
//   - USE_H100CPC → H100 mappings (2 CPCs: sizes 6,6)
//   - default  → V100 mappings (6 GPCs: sizes 14,14,14,14,12,12)
//

struct GPCMapping {
    int size;
    const unsigned int *sm_ids;
};

#ifdef USE_A100
// --------------------- A100 MAPPINGS ---------------------
static const unsigned int a100_gpc0[] = { 0, 1,14,15,28,29,42,43,56,57,70,71,84,85,98,99 };
static const unsigned int a100_gpc1[] = { 2, 3,30,31,17,16,73,72,59,58,99,98,45,44,87,86 };
static const unsigned int a100_gpc2[] = { 4, 5,33,32,19,18,75,74,61,60,101,100,47,46,89,88 };
static const unsigned int a100_gpc3[] = { 6, 7,20,21,35,34,63,62,77,76,103,102,49,48,91,90 };
static const unsigned int a100_gpc4[] = { 8, 9,22,23,37,36,79,78,65,64,105,104,51,50,93,92 };
static const unsigned int a100_gpc5[] = {10,11,39,38,25,24,81,80,95,94,52,53,67,66        }; 
static const unsigned int a100_gpc6[] = {12,13,40,41,27,26,82,83,55,54,97,96,69,68        }; 

static const GPCMapping gpcMappings[] = {
    {16, a100_gpc0}, {16, a100_gpc1}, {16, a100_gpc2},
    {16, a100_gpc3}, {16, a100_gpc4}, {14, a100_gpc5},
    {14, a100_gpc6}
};

#elif defined(USE_H100)
// --------------------- H100 PCIe MAPPINGS ---------------------

static const unsigned int h100_gpc0[] = {0, 1, 14, 15, 28, 29, 42, 43, 56, 57, 70, 71, 84, 85, 110, 111, 112, 113}; //part 0
static const unsigned int h100_gpc1[] = {2, 3, 16, 17, 30, 31, 44, 45, 58, 59, 72, 73, 86, 87, 98, 99}; // part 1
static const unsigned int h100_gpc2[] = {4, 5, 18, 19, 32, 33, 46, 47, 60, 61, 74, 75, 88, 89, 100, 101}; // part 0
static const unsigned int h100_gpc3[] = {6, 7, 20, 21, 34, 35, 48, 49, 62, 63, 76, 77, 90, 91, 102, 103}; //part 0
static const unsigned int h100_gpc4[] = {8, 9, 22, 23, 36, 37, 50, 51, 64, 65, 78, 79, 92, 93, 104, 105}; //part 0
static const unsigned int h100_gpc5[] = {10, 11, 24, 25, 38, 39, 52, 53, 66, 67, 80, 81, 94, 95, 106, 107}; //part 1
static const unsigned int h100_gpc6[] = {12, 13, 26, 27, 40, 41, 54, 55, 68, 69, 82, 83, 96, 97, 108, 109}; // part 1

static const GPCMapping gpcMappings[] = {
    {18, h100_gpc0}, {16, h100_gpc1}, {16, h100_gpc2},
    {16, h100_gpc3}, {16, h100_gpc4}, {16, h100_gpc5},
    {16, h100_gpc6}
};

#elif defined(USE_H100cpc)
// --------------------- H100cpc MAPPINGS ---------------------
// Only two of the three CPCs available in GPC1 of H100 PCIe
static const unsigned int h100_cpc0[] = {
    2,3,44,45,86,87 
};
static const unsigned int h100_cpc1[] = {
   16, 17, 58, 59, 98, 99
};

static const GPCMapping gpcMappings[] = {
    {6, h100_cpc0}, {6, h100_cpc1}
};


#else
// --------------------- V100 MAPPINGS (default) ---------------------
static const unsigned int gpc0[] = {0,12,24,36,48,60,70,1,13,25,37,49,61,71};
static const unsigned int gpc1[] = {2,3,14,15,26,27,38,39,50,51,62,63,72,73};
static const unsigned int gpc2[] = {4,5,16,17,28,29,40,41,52,53,64,65,74,75};
static const unsigned int gpc3[] = {6,7,18,19,30,31,42,43,54,55,66,67,76,77};
static const unsigned int gpc4[] = {8,9,20,21,32,33,44,45,56,57,68,69};
static const unsigned int gpc5[] = {10,11,22,23,34,35,46,47,58,59,78,79};

static const GPCMapping gpcMappings[] = {
    {14, gpc0}, {14, gpc1}, {14, gpc2},
    {14, gpc3}, {12, gpc4}, {12, gpc5}
};
#endif

//------------------------------------------------------------------------------
// Main Function
//------------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <CTA> <WARP> <ITERATION> <GPCselectedList> <SMmax>\n"
                     "Example: " << argv[0] << " 1 2 100 0,1,3 10\n";
        return EXIT_FAILURE;
    }

    // Parse arguments
    int CTA       = std::atoi(argv[1]);
    int WARP      = std::atoi(argv[2]);
    int ITERATION = std::atoi(argv[3]);
    std::vector<int> selectedGPCs;
    {
        std::istringstream ss(argv[4]);
        std::string token;
        while (std::getline(ss, token, ',')) {
            int gpc = std::stoi(token);
#ifdef USE_A100
            if (gpc < 0 || gpc > 6) {
                std::cerr << "Error: For A100, GPC must be 0..6.\n";
                return EXIT_FAILURE;
            }
#elif defined(USE_H100)
            if (gpc < 0 || gpc > 6) {
                std::cerr << "Error: For H100, GPC must be 0 or 1.\n";
                return EXIT_FAILURE;
            }
#else
            if (gpc < 0 || gpc > 5) {
                std::cerr << "Error: For V100, GPC must be 0..5.\n";
                return EXIT_FAILURE;
            }
#endif
            selectedGPCs.push_back(gpc);
        }
    }
    if (selectedGPCs.empty()) {
        std::cerr << "No valid GPC selected.\n";
        return EXIT_FAILURE;
    }
    int SM_max = std::atoi(argv[5]);

    // Initialize device
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int NUM_SM = prop.multiProcessorCount;

    // Build unified SM list from selected GPCs
    std::vector<unsigned int> unifiedSMIDs;
    for (int gpc : selectedGPCs) {
        const GPCMapping &m = gpcMappings[gpc];
        unifiedSMIDs.insert(unifiedSMIDs.end(), m.sm_ids, m.sm_ids + m.size);
    }

#ifdef SORT_GPC
        // Sort by even-odd first, then by ascending value
        std::sort(unifiedSMIDs.begin(), unifiedSMIDs.end(), [](unsigned int a, unsigned int b) {
            bool a_even = (a % 2 == 0);
            bool b_even = (b % 2 == 0);
            if (a_even != b_even) {
                // Put even first
                return a_even;
            }
            // Both even or both odd → compare numerically
            return a < b;
        });
#else
        // Sort in ascending order only
        std::sort(unifiedSMIDs.begin(), unifiedSMIDs.end());
#endif

    // Keep original mapping size for stride calculation
    int GPCsize = static_cast<int>(unifiedSMIDs.size());
    // Truncate if needed
    if (SM_max < GPCsize) {
        unifiedSMIDs.resize(SM_max);
    }
    unsigned int active_sms = static_cast<unsigned int>(unifiedSMIDs.size());

    // Copy SM list to device
    unsigned int *d_SM_ids = nullptr;
    size_t sm_array_bytes = unifiedSMIDs.size() * sizeof(unsigned int);
    cudaMalloc(&d_SM_ids, sm_array_bytes);
    cudaMemcpy(d_SM_ids, unifiedSMIDs.data(), sm_array_bytes, cudaMemcpyHostToDevice);

    // Allocate data buffer based on L2 size
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    unsigned int l2Bytes = prop.l2CacheSize;
    int elementCount = flp2(l2Bytes) / sizeof(mt);
    mt *d_data = nullptr;
    cudaMalloc(&d_data, elementCount * sizeof(mt));

    // Prepare timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> bwMeasurements(ITERATION);

    // Launch kernel ITERATION times
    const int kernelLoops = 2000;
    int threadsPerBlock = 32 * WARP;
    dim3 gridDim(NUM_SM, CTA);
    for (int iter = 0; iter < ITERATION; iter++) {
        cudaEventRecord(start, 0);
        kernel<<<gridDim, threadsPerBlock>>>(d_data,
                                        elementCount,
                                        kernelLoops,
                                        d_SM_ids,
                                        active_sms,
                                        GPCsize);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float dt_ms = 0.0f;
        cudaEventElapsedTime(&dt_ms, start, stop);
        cudaDeviceSynchronize();

        // Bandwidth in GB/s
        bwMeasurements[iter] =
            (elementCount * sizeof(mt) * kernelLoops * active_sms * CTA)
            / (dt_ms * 1e6f * GPCsize);
    }

    float avgBW = mean(bwMeasurements.data(), ITERATION);
    std::cout << std::fixed << std::setprecision(1)
              << std::setw(13) << avgBW 
              << std::endl;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_SM_ids);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}