/*
    OVERVIEW:
    ----------
    This experiment demonstrates two modes for targeting a specific L2 slice,
    by selecting the Streaming Multiprocessors (SMs) directicly, or by choosing 
    a subset of SMs based on the GPU's GPC (Graphics Processing Cluster) mapping.
    The experiment uses a CSV file to define the L2 cache slices for different
    architectures (A100, H100, V100). The CSV file contains 32 columns and a number
    of rows that varies by architecture. Each row in the CSV file corresponds to a slice of the L2 cache.

      1. GPC Mode (default):
         - The application accepts:
              <GPCselectedList> <SMmax> <slice_index>
         - GPCselectedList: Comma-separated GPC IDs (e.g., "0,1,2")
         - SMmax: Maximum SM index (within the selected GPCs) that will be active.
         - slice_index: The row index from the external CSV file to select the L2 cache slice.
         - The kernel will run on a subset of SMs selected via pre-defined GPC mappings and a user-chosen L2 slice.

      2. Direct SM Mode:
         - Compiled with the flag ‑DUSE_DIRECT_SM.
         - The application accepts:
              <SMid> <slice_index>
         - SMid: Directly specifies the SM id to execute the kernel.
         - slice_index: The row index from the CSV file defining the L2 slice.
         - In this mode, the GPC mapping and SM range selection are bypassed so that only the specified SM is used.

    ARCHITECTURE CONFIGURATION:
    ---------------------------
    - The correct GPC mapping and CSV file are chosen at compile time according to GPU type:
          • -DUSE_A100 → A100 mapping & "L2_slices_A100.csv" (80 rows)
          • -DUSE_H100 → H100 mapping & "L2_slices_H100.csv" (6 rows)
          • default   → V100 mapping & "L2_slices_V100.csv" (32 rows)
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_runtime.h>

//----------------------------------------------------------------------------
// Definitions and macros
//----------------------------------------------------------------------------
#define BLOCK_SIZE    32
#define ADDRESS_BLOCK 256

#ifndef ITERATION
#define ITERATION 10000
#endif

//------------------------------------------------------------------------------
// GPC Mapping Definitions (only used in GPC mode)
//------------------------------------------------------------------------------
// When compiling for A100 or H100, add -DUSE_A100 or -DUSE_H100 respectively.
// Otherwise, the code defaults to V100.
#ifndef USE_DIRECT_SM  // Only needed when using GPC mode

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
    static const unsigned int a100_gpc5[] = {10,11,39,38,25,24,81,80,95,94,52,53,67,66};
    static const unsigned int a100_gpc6[] = {12,13,40,41,27,26,82,83,55,54,97,96,69,68};

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

#endif  // USE_DIRECT_SM

//------------------------------------------------------------------------------
// Utility: Check CUDA errors
//------------------------------------------------------------------------------
#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if(e != cudaSuccess) {                                          \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,    \
               cudaGetErrorString(e));                              \
        exit(0);                                                    \
    }                                                               \
}

//------------------------------------------------------------------------------
// Device function: Retrieve the current SM (Streaming Multiprocessor) ID.
//------------------------------------------------------------------------------
__device__ unsigned int get_smid(void) {
    unsigned int sm_id;
    asm("mov.u32 %0, %smid;" : "=r"(sm_id));
    return sm_id;
}

//------------------------------------------------------------------------------
// Kernel:
//   In Direct SM Mode, only threads on the specified SM (targetSM) execute the computation.
//   In GPC Mode (default), threads are filtered based on the selected GPC mappings and SMmax.
//------------------------------------------------------------------------------
#ifdef USE_DIRECT_SM

__global__ void kernel(unsigned int *d, unsigned int *slice, unsigned int targetSM) {
    volatile unsigned int k = 0;
    unsigned int tid   = threadIdx.x;
    unsigned int warp  = tid / 32;
    unsigned int tx    = tid % 32;
    unsigned int sm_id = get_smid();

    if (sm_id == targetSM) {
        for (int i = 0; i < ITERATION; i++) {
            unsigned int idx = warp * 8 + slice[tx] * ADDRESS_BLOCK;
            for (int j = 0; j < 2; j++) {
                k += d[idx];
            }
            d[sm_id * ADDRESS_BLOCK] = k;
        }
    }
}

#else

__global__ void kernel(unsigned int *a0, unsigned int *slice,
                         unsigned int *sm_ids, unsigned int num_sm_ids, unsigned int SMmax) {
    volatile unsigned int k = 0;
    unsigned int tid   = threadIdx.x;
    unsigned int warp  = tid / 32;
    unsigned int tx    = tid % 32;
    unsigned int sm_id = get_smid();

    for (int h = 0; h < num_sm_ids; h++) {
        if (sm_id == sm_ids[h] && h < SMmax) {
            for (int i = 0; i < ITERATION; i++) {
                unsigned int idx = warp * 8 + slice[tx] * ADDRESS_BLOCK;
                for (int j = 0; j < 2; j++) {
                    k += a0[idx];
                }
                a0[sm_id * ADDRESS_BLOCK] = k;
            }
        }
    }
}

#endif

//------------------------------------------------------------------------------
// Utility: Read matrix from CSV file
//------------------------------------------------------------------------------
static bool readMatrixFromCSVFunc(const std::string &filename, std::vector<unsigned int>& matrix,
                                  size_t expectedRows, size_t expectedCols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    std::string line;
    size_t row = 0;
    while (std::getline(file, line) && row < expectedRows) {
        std::stringstream linestream(line);
        std::string cell;
        while (std::getline(linestream, cell, ',')) {
            if (cell.empty()) continue;
            try {
                unsigned int val = std::stoul(cell);
                matrix.push_back(val);
            } catch (const std::exception &e) {
                std::cerr << "Error parsing value in CSV: " << e.what() << std::endl;
                return false;
            }
        }
        row++;
    }
    file.close();
    if (matrix.size() != expectedRows * expectedCols) {
        std::cerr << "Error: Matrix dimensions do not match expected "
                  << expectedRows << "x" << expectedCols << ". Got "
                  << matrix.size() << " elements." << std::endl;
        return false;
    }
    return true;
}

//------------------------------------------------------------------------------
// Host Code (main)
//------------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // Query device properties.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int numSM = deviceProp.multiProcessorCount;
    // L2 cache size (in bytes) is converted to unsigned ints.
    unsigned int L2_size = deviceProp.l2CacheSize / sizeof(unsigned int);

    // Determine expected rows based on architecture.
    int expectedRows = 32; // default for V100
#ifdef USE_A100
    expectedRows = 80;
#elif defined(USE_H100)
    expectedRows = 6;
#endif
    int expectedCols = 32; // All CSV files have 32 columns.

    int slice_index = 0;

#ifdef USE_DIRECT_SM
    // ---------------- DIRECT SM MODE ----------------
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <SMid> <slice_index>" << std::endl;
        return EXIT_FAILURE;
    }
    unsigned int targetSM = static_cast<unsigned int>(atoi(argv[1]));
    slice_index = atoi(argv[2]);
    unsigned int SMmax = 1;
#else
    // ---------------- GPC MODE (default) ----------------
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <GPCselectedList> <SMmax> <slice_index>" 
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Parse comma-separated list of GPC indices.
    std::vector<int> selectedGPCs;
    std::string gpcArg(argv[1]);
    std::istringstream gpcStream(gpcArg);
    std::string token;
    while (std::getline(gpcStream, token, ',')) {
        try {
            int gpc = std::stoi(token);
            selectedGPCs.push_back(gpc);
        } catch (const std::exception &e) {
            std::cerr << "Error parsing GPC list: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }
    if (selectedGPCs.empty()) {
        std::cerr << "Error: No valid GPC specified." << std::endl;
        return EXIT_FAILURE;
    }

    unsigned int SMmax = static_cast<unsigned int>(atoi(argv[2]));
    slice_index = atoi(argv[3]);
#endif

    // Check that slice_index is within the valid range.
    if (slice_index < 0 || slice_index >= expectedRows) {
        std::cerr << "Error: slice_index must be between 0 and " << (expectedRows - 1)
                  << " for the selected architecture." << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate and initialize the global array using L2_size.
    unsigned int *h_globalArray = (unsigned int *)malloc(sizeof(unsigned int) * L2_size);
    for (unsigned int i = 0; i < L2_size; i++) {
        h_globalArray[i] = i;
    }

    // Select the appropriate CSV file based on architecture.
    std::string csvFile;
#ifdef USE_A100
    csvFile = "L2_slices_A100.csv";
#elif defined(USE_H100)
    csvFile = "L2_slices_H100.csv";
#else
    csvFile = "L2_slices_V100.csv";
#endif

    // Read the CSV matrix.
    std::vector<unsigned int> matrixCSV;
    if (!readMatrixFromCSVFunc(csvFile, matrixCSV, expectedRows, expectedCols)) {
        std::cerr << "Failed to read matrix from CSV file: " << csvFile << std::endl;
        return EXIT_FAILURE;
    }

    // Extract the chosen slice (one row) from the CSV data.
    unsigned int *h_slice = (unsigned int *)malloc(sizeof(unsigned int) * expectedCols);
    for (int col = 0; col < expectedCols; col++) {
        h_slice[col] = matrixCSV[slice_index * expectedCols + col];
    }

#ifndef USE_DIRECT_SM
    // ----- GPC MODE: Build unified SM IDs array from the selected GPC mappings -----
    int totalGPCSize = 0;
    int availableMappings = sizeof(gpcMappings) / sizeof(GPCMapping);
    for (int gpc : selectedGPCs) {
        if (gpc < 0 || gpc >= availableMappings) {
            std::cerr << "Error: Invalid GPC index: " << gpc << std::endl;
            return EXIT_FAILURE;
        }
        totalGPCSize += gpcMappings[gpc].size;
    }
    unsigned int *h_GPCSelectedSMs = (unsigned int *)malloc(sizeof(unsigned int) * totalGPCSize);
    unsigned int offset = 0;
    for (int gpc : selectedGPCs) {
        int size = gpcMappings[gpc].size;
        std::memcpy(h_GPCSelectedSMs + offset, gpcMappings[gpc].sm_ids, sizeof(unsigned int) * size);
        offset += size;
    }
#endif

    // Allocate device memory.
    unsigned int *d_globalArray = nullptr;
    unsigned int *d_slice       = nullptr;
    cudaMalloc((void **)&d_globalArray, sizeof(unsigned int) * L2_size);
    cudaMalloc((void **)&d_slice, sizeof(unsigned int) * expectedCols);
    cudaCheckError();

#ifndef USE_DIRECT_SM
    unsigned int *d_GPCSelectedSMs = nullptr;
    cudaMalloc((void **)&d_GPCSelectedSMs, sizeof(unsigned int) * (totalGPCSize));
    cudaCheckError();
#endif

    cudaMemcpy(d_globalArray, h_globalArray, sizeof(unsigned int) * L2_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_slice, h_slice, sizeof(unsigned int) * expectedCols, cudaMemcpyHostToDevice);
    cudaCheckError();

#ifndef USE_DIRECT_SM
    cudaMemcpy(d_GPCSelectedSMs, h_GPCSelectedSMs, sizeof(unsigned int) * totalGPCSize, cudaMemcpyHostToDevice);
    cudaCheckError();
#endif

    // Set the kernel attribute for preferred shared memory carveout.
    int carveout = 100; // Use 100% of available shared memory.
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaCheckError();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int CTAperSM = 2;

    // Configure the kernel launch dimensions.
    // Block dimension: 1024 threads per block.
    // Grid dimensions: (numSM, 32), 32 CTAs per SM.
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE); // 1024 threads per block
    dim3 gridDim(numSM, CTAperSM);

    cudaEventRecord(start, 0);
#ifdef USE_DIRECT_SM
    // Launch the kernel in Direct SM Mode.
    kernel<<<gridDim, blockDim>>>(d_globalArray, d_slice, targetSM);
#else
    // Launch the kernel in GPC Mode.
    kernel<<<gridDim, blockDim>>>(d_globalArray, d_slice,
                                  d_GPCSelectedSMs, totalGPCSize, SMmax);
#endif
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    float bw = (float)(1024 * sizeof(unsigned int) * 8 * ITERATION * SMmax * CTAperSM) / (elapsedTime * 1e6);

    std::cout << bw 
    << std::endl;

    // Free allocated host and device memory.
    free(h_globalArray);
    free(h_slice);
#ifndef USE_DIRECT_SM
    free(h_GPCSelectedSMs);
    cudaFree(d_GPCSelectedSMs);
#endif
    cudaFree(d_globalArray);
    cudaFree(d_slice);

    return 0;
}
