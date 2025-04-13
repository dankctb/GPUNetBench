/*
    This experiment selects one or more Graphics Processing Clusters (GPCs) and runs a kernel on
    specific Streaming Multiprocessors (SMs) within the set of selected GPCs. It accesses a user‐specified
    set of slices from L2 cache by reading slice data rows from an external CSV file.

    HOW IT WORKS:
    - The program accepts the following command line arguments:
         <GPCselectedList> <SMmax> <MPnum> <slicesPerMP> <MP_id[0]> ... <MP_id[MPnum-1]>
      where:
         • GPCselectedList: a comma-separated list of Graphics Processing Cluster IDs (0..5) || 6 has 28 interleaved SMs from different GPCs (with no TPC contention) 
         • SMmax: the maximum SM index to be active within the selected GPCs
         • MPnum: the number of memory partitions (MPs) selected
         • slicesPerMP: number of slices to extract per MP (up to 8 based on current mapping)
         • MP_id[i]: the MP IDs (from 0 to 3) for each memory partition.

    - Each selected MP contributes the specified number of slices, so the total number
      of slices equals MPnum * slicesPerMP.
      
    - The CSV file ("L2_slices_4.csv") is expected to hold a matrix with
      32 rows and (32 * MULTIPLIER) columns. Each row corresponds to a different L2 slice. 
      The slices corresponding to the selected MPs are extracted based on a pre-defined mapping.
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

// Definitions and macros
#define BLOCK_SIZE     32                   // Number of threads per block dimension or similar indexing basis
#define ADDRESS_BLOCK  256                  // For memory addressing offset
#define S_SIZE         (((6*1024)*1024)/4)  // Size of the primary data array (must be smaller than L2 cache)

#ifndef ITERATION
#define ITERATION 10000
#endif

#define MULTIPLIER     4                    // Determines number of columns: BLOCK_SIZE * MULTIPLIER

// Check CUDA errors.
#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if(e != cudaSuccess) {                                          \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(0);                                                  \
    }                                                               \
}

// Device function: Retrieve the current SM (Streaming Multiprocessor) ID.
__device__ unsigned int get_smid(void) {
    unsigned int sm_id;
    asm("mov.u32 %0, %smid;" : "=r"(sm_id));
    return sm_id;
}

/*
    Kernel Function:
    This kernel uses shared memory and performs non‐coalesced accesses to a set of
    L2 slices provided from a reordered matrix. The slices come from the MPs selected,
    and the number of slices is determined as (slicesPerMP) times the number of MPs.

    Parameters:
      - globalArray: Primary global data array.
      - selectedSlices: Array containing the slice addresses selected from the CSV matrix.
      - numSlicesPerMP: Number of slices per MP (provided via command line).
      - gpcSelectedSMs: Array holding the unified list of SM IDs for the selected GPCs.
      - selectedGPCSize: Total number of SMs in the unified selection.
      - maxSM: Maximum SM index to run.
      - MPnum: Number of memory partitions (MPs) selected.
*/

__global__ void kernel(unsigned int *a0, unsigned int *value, unsigned int max_slice, unsigned int *sm_ids1, unsigned int GPC_size, unsigned int SMmax, unsigned int MPnum){
    volatile unsigned int k;
    unsigned int idx;
    unsigned int tid = threadIdx.x;
    unsigned int warp = tid/32;
    unsigned int tx = tid%32;
    unsigned int sm_id = get_smid();
    unsigned int bid = blockIdx.y%MPnum;

    __syncthreads();
#ifdef ENABLE_ALL_SMS
    // If all SMs are enabled, skip the GPC-based loops
    for (int i = 0; i < ITERATION; i++) {
        for (int b = 0; b < MPnum; b++) {
            for (int n = 0; n < max_slice; n++) {
                idx = warp * 8 + tx % 8 
                    + value[tx + n * BLOCK_SIZE * MULTIPLIER 
                    + b * BLOCK_SIZE * MULTIPLIER * 8] * ADDRESS_BLOCK;
                for (int j = 0; j < 2; j++) k += a0[idx];
            }
        }
    }
    a0[sm_id * ADDRESS_BLOCK] = k;
#else
    for (int h = 0; h < GPC_size; h++) {
        if (sm_id == sm_ids1[h] && h < SMmax) {
            for (int i = 0; i < ITERATION; i++) {
                for (int b = 0; b < MPnum; b++) {
                    for (int n = 0; n < max_slice; n++) {
                        idx = warp * 8 + tx % 8
                            + value[tx + n * BLOCK_SIZE * MULTIPLIER
                            + b * BLOCK_SIZE * MULTIPLIER * 8] * ADDRESS_BLOCK;
                        for (int j = 0; j < 2; j++) k += a0[idx];
                    }
                }
            }
            a0[sm_id * ADDRESS_BLOCK] = k;
        }
    }
#endif


}



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

//
// Host Code
//
int main(int argc, char *argv[]) {
    // usage: <GPCselectedList> <SMmax> <MPnum> <slicesPerMP> <MP_id[0]> ... <MP_id[MPnum-1]>
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] 
                  << " <GPCselectedList> <SMmax> <MPnum> <slicesPerMP> <MP_id[0]> ... <MP_id[MPnum-1]>" 
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Parse the comma-separated list of GPCs.
    std::vector<int> selectedGPCs;
    std::string gpcArg(argv[1]);
    std::istringstream gpcStream(gpcArg);
    std::string token;
    while (std::getline(gpcStream, token, ',')) {
        try {
            int gpc = std::stoi(token);
            if (gpc < 0 || gpc > 6) {
                std::cerr << "Error: Each GPC must be between 0 and 5." << std::endl;
                return EXIT_FAILURE;
            }
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

    // Get device properties to determine the number of SMs.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int NUM_SM = deviceProp.multiProcessorCount;
    cudaSetDevice(0);

    // Parse SMmax, MPnum, and slicesPerMP.
    unsigned int maxSM    = atoi(argv[2]);    // Maximum SM index to be active within the selected GPCs
    unsigned int MPnum    = atoi(argv[3]);    // Number of memory partitions (MPs) selected
    unsigned int slicesPerMP = atoi(argv[4]);  // Number of slices per MP to select

    // Ensure that the number of slices per MP does not exceed the mapping size.
    if (slicesPerMP > 8) {
        std::cerr << "Error: Maximum slices per MP supported is 8." << std::endl;
        return EXIT_FAILURE;
    }

    // Check if the appropriate number of MP IDs have been provided.
    if (argc != 5 + MPnum) {
        std::cerr << "Error: Incorrect number of MP IDs provided. Expected " << MPnum << " MP IDs." << std::endl;
        return EXIT_FAILURE;
    }

    // Each MP contributes 'slicesPerMP' slices.
    const unsigned int totalSlices  = MPnum * slicesPerMP;

    // Allocate and initialize the primary global array.
    unsigned int *h_globalArray = (unsigned int *)malloc(sizeof(unsigned int) * S_SIZE);
    for (unsigned int i = 0; i < S_SIZE; i++) {
        h_globalArray[i] = i;
    }
    unsigned int *d_globalArray = nullptr;

    // Parse MP IDs from command line into an array.
    unsigned int *mpSelection = (unsigned int *)malloc(sizeof(unsigned int) * MPnum);
    for (unsigned int i = 0; i < MPnum; i++) {
        mpSelection[i] = atoi(argv[5 + i]);
    }

    // Allocate and compute slice indices.
    // For each selected MP, use a predefined mapping (MP IDs 0-3 are valid, MP ID 4 is interleaved).
    // that selects up to 8 rows (slices). Only the first 'slicesPerMP' indices are used.
    unsigned int *h_sliceIndices = (unsigned int *)malloc(sizeof(unsigned int) * totalSlices);
    static const int mpMapping[5][8] = {
        {7,  3, 31, 27,  5,  1, 29, 25}, // MP 0 mapping: yields 8 slice (row) indices
        {8, 12, 16, 20, 10, 14, 18, 22}, // MP 1 mapping
        {0,  4, 24, 28,  2,  6, 26, 30}, // MP 2 mapping
        {17, 21,  9, 13, 19, 23, 11, 15},  // MP 3 mapping
        {7, 8, 0, 17, 3, 12, 4, 21}   // Interleaved MPs mapping
    };

    // For each MP, copy its designated slice indices (only the first 'slicesPerMP' values).
    for (unsigned int mp = 0; mp < MPnum; mp++) {
        int mpID = mpSelection[mp];  // MP identifier (should be between 0 and 3)
        if (mpID < 0 || mpID > 4) {
            std::cerr << "Error: MP ID must be between 0 and 4." << std::endl;
            return EXIT_FAILURE;
        }
        for (unsigned int slice = 0; slice < slicesPerMP; slice++) {
            h_sliceIndices[slice + mp * slicesPerMP] = mpMapping[mpID][slice];
        }
    }
    
    // Define the GPC (Graphics Processing Cluster) mappings for V100.
    static const unsigned int gpc0[] = { 0, 12, 24, 36, 48, 60, 70,  1, 13, 25, 37, 49, 61, 71 };
    static const unsigned int gpc1[] = { 2,  3, 14, 15, 26, 27, 38, 39, 50, 51, 62, 63, 72, 73 };
    static const unsigned int gpc2[] = { 4,  5, 16, 17, 28, 29, 40, 41, 52, 53, 64, 65, 74, 75 };
    static const unsigned int gpc3[] = { 6,  7, 18, 19, 30, 31, 42, 43, 54, 55, 66, 67, 76, 77 };
    static const unsigned int gpc4[] = { 8,  9, 20, 21, 32, 33, 44, 45, 56, 57, 68, 69 };
    static const unsigned int gpc5[] = { 10, 11, 22, 23, 34, 35, 46, 47, 58, 59, 78, 79 };
    static unsigned int interleaved[28];

    for (int i = 0; i < 28; i++) {
        interleaved[i] = i*2;
    }

    // Define a structure to hold the GPC mappings.
    struct GPCMapping {
        int size;
        const unsigned int* sm_ids;
    };

    static const GPCMapping gpcMappings[] = {
        {14, gpc0}, {14, gpc1}, {14, gpc2},
        {14, gpc3}, {12, gpc4}, {12, gpc5},
        {28, interleaved}
    };

    // Build a unified SM IDs array from all the selected GPCs.
    unsigned int totalGPCSize = 0;
    for (int gpc : selectedGPCs) {
        totalGPCSize += gpcMappings[gpc].size;
    }
    unsigned int *h_GPCSelectedSMs = (unsigned int*)malloc(sizeof(unsigned int) * totalGPCSize);
    unsigned int offset = 0;
    for (int gpc : selectedGPCs) {
        int size = gpcMappings[gpc].size;
        std::memcpy(h_GPCSelectedSMs + offset, gpcMappings[gpc].sm_ids, sizeof(unsigned int) * size);
        offset += size;
    }
    unsigned int *d_GPCSelectedSMs = nullptr;
    cudaMalloc((void**)&d_GPCSelectedSMs, sizeof(unsigned int) * totalGPCSize);


    std::vector<unsigned int> matrixCSV;
    const size_t expectedRows = BLOCK_SIZE;
    const size_t expectedCols = BLOCK_SIZE * MULTIPLIER;
    if (!readMatrixFromCSVFunc("L2_slices_4.csv", matrixCSV, expectedRows, expectedCols)) {
        std::cerr << "Failed to read matrix from CSV file." << std::endl;
        return EXIT_FAILURE;
    }


    // Build the selected slices array from the CSV matrix.
    // For each selected MP, extract its designated slice rows (as defined in h_sliceIndices)
    // and store them into the selection array.
    unsigned int *h_selectedSlices = (unsigned int *)malloc(sizeof(unsigned int) * totalSlices * BLOCK_SIZE * MULTIPLIER);
    for (unsigned int mp = 0; mp < MPnum; mp++) {
        for (unsigned int slice = 0; slice < slicesPerMP; slice++) {
            for (unsigned int col = 0; col < BLOCK_SIZE * MULTIPLIER; col++) {
                h_selectedSlices[col +
                    (BLOCK_SIZE * MULTIPLIER) * slice +
                    mp * (slicesPerMP * BLOCK_SIZE * MULTIPLIER)] =
                    matrixCSV[h_sliceIndices[slice + mp * slicesPerMP] * BLOCK_SIZE * MULTIPLIER + col];
            }
        }
    }
    // This construction allows you to choose the number of slices to access by selecting
    // the number of MPs (each MP contributing 'slicesPerMP' slices).

    // Allocate device memory for the global array and the selected slices array.
    unsigned int *d_selectedSlices = nullptr;
    cudaMalloc((void**)&d_globalArray, sizeof(unsigned int) * S_SIZE);
    cudaMalloc((void**)&d_selectedSlices, sizeof(unsigned int) * totalSlices * BLOCK_SIZE * MULTIPLIER);
    cudaCheckError();

    cudaMemcpy(d_globalArray, h_globalArray, sizeof(unsigned int) * S_SIZE, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_selectedSlices, h_selectedSlices, sizeof(unsigned int) * totalSlices * BLOCK_SIZE * MULTIPLIER, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_GPCSelectedSMs, h_GPCSelectedSMs, sizeof(unsigned int) * totalGPCSize, cudaMemcpyHostToDevice);
    cudaCheckError();

    // Set the kernel attribute for preferred shared memory carveout.
    int carveout = 100; // Use 100% of available shared memory.
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaCheckError();

    // Configure the kernel launch dimensions.
    // Block dimension: 1024 threads per block.
    // Grid dimensions: NUM_SM x 8 x 4. (32 CTAs per SM)
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(NUM_SM, 8, 4);
    
    // Launch the kernel.
    kernel<<<gridDim, blockDim>>>(d_globalArray, d_selectedSlices,
                                  slicesPerMP, d_GPCSelectedSMs, totalGPCSize,
                                  maxSM, MPnum);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    // Free allocated host and device memory.
    free(h_globalArray);
    free(h_selectedSlices);
    free(h_sliceIndices);
    free(mpSelection);
    free(h_GPCSelectedSMs);
    cudaFree(d_globalArray);
    cudaFree(d_selectedSlices);
    cudaFree(d_GPCSelectedSMs);

    return 0;
}
