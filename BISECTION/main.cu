/*
    OVERVIEW:
    ----------
    This experiment selects one or more Graphics Processing Clusters (GPCs) and
    runs a CUDA kernel on a unified list of Streaming Multiprocessors (SMs)
    derived from the selected GPCs. The program allocates a global data buffer
    based on the device’s L2 cache size and reads an address matrix from a CSV
    file (the file chosen depends on the target GPU type and a compile‑time
    partition option). The kernel then uses this address matrix to perform read 
    operations only on one partition of the L2 cache. When targeting the remote
    partition, the bandwidth measured will be the bisection bandwidth of the
    L2-to-L2 interconnect. The kernel is designed to run on NVIDIA A100 and H100 GPUs.
    
    HOW IT WORKS:
    - Command-line arguments are used as follows:
         <CTAs per SM> <list_of_GPCs> 
      where:
         • CTAs per SM      : Number of concurrent thread blocks per SM.
         • list_of_GPCs     : Comma-separated list of GPC IDs (e.g., "0,1,3").
           (Allowed GPC values depend on the target – for A100: 0..6; for H100: 0..1.)
    - The address matrix is loaded from a CSV file. The file name is chosen based on:
         • GPU type: A100 or H100 (via compile‑time flag USE_A100 or USE_H100)
         • Partition: (defined by compile‑time macro PARTITION, e.g. 0 or 1)
         For example, the file "A100-0.csv" contains a 64‑row×32‑column matrix, and
         "H100-1.csv" contains a 164‑row×32‑column matrix.
    - A unified SM list is built from pre‑defined GPC mappings. The SM IDs are sorted
      and, if necessary, truncated based on a maximum SM count.
    - The kernel uses the per‑SM mapping and the address matrix to generate read indices,
      performs many iterations of global memory reads, and writes back a feedback value.
    - CUDA events are used to measure kernel execution time, and approximate L2-read
      bandwidth is computed from the data volume and elapsed time.

    - NOTE: The GPCs in the same partition might change between different GPUs also 
            of the same architecure. Please check the GPC mapping for the target GPU.
    - NOTE: The kernel is designed to run on NVIDIA A100 and H100 GPUs only.
*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>

// -----------------------------------------------------------------------------
// CONFIG CONSTANTS & MACROS
// -----------------------------------------------------------------------------
#define BLOCK_SIZE     32     // Number of columns (32 columns in CSV)
#ifdef ITERATION
    #define ITERATION  10000  // Number of iterations in kernel
#endif
#define ADDRESS_BLOCK  256    // Stride block in 32-bit words

// -----------------------------------------------------------------------------
// CSV File Selection via Compile-Time Options
// -----------------------------------------------------------------------------
#ifndef PARTITION
  #define PARTITION 0  // Default partition value if not provided
#endif

// -----------------------------------------------------------------------------
// GPC MAPPING (unchanged from original kernel code style)
// -----------------------------------------------------------------------------
#ifdef USE_H100
// H100 mappings
static const unsigned int GPC0[] = {0, 1,14,15,28,29,42,43,56,57,70,71,84,85,110,111,112,113};
static const unsigned int GPC1[] = {2, 3,16,17,30,31,44,45,58,59,72,73,86,87,98,99};
static const unsigned int GPC2[] = {4, 5,18,19,32,33,46,47,60,61,74,75,88,89,100,101};
static const unsigned int GPC3[] = {6, 7,20,21,34,35,48,49,62,63,76,77,90,91,102,103};
static const unsigned int GPC4[] = {8, 9,22,23,36,37,50,51,64,65,78,79,92,93,104,105};
static const unsigned int GPC5[] = {10,11,24,25,38,39,52,53,66,67,80,81,94,95,106,107};
static const unsigned int GPC6[] = {12,13,26,27,40,41,54,55,68,69,82,83,96,97,108,109};
#else
// A100 mappings (default)
static const unsigned int GPC0[] = {0, 1,14,15,28,29,42,43,56,57,70,71,84,85,98,99};
static const unsigned int GPC1[] = {2, 3,30,31,17,16,73,72,59,58,99,98,45,44,87,86};
static const unsigned int GPC2[] = {4, 5,33,32,19,18,75,74,61,60,101,100,47,46,89,88};
static const unsigned int GPC3[] = {6, 7,20,21,35,34,63,62,77,76,103,102,49,48,91,90};
static const unsigned int GPC4[] = {8, 9,22,23,37,36,79,78,65,64,105,104,51,50,93,92};
static const unsigned int GPC5[] = {10,11,39,38,25,24,81,80,95,94,52,53,67,66};
static const unsigned int GPC6[] = {12,13,40,41,27,26,82,83,55,54,97,96,69,68};
#endif

struct GPCInfo {
    const unsigned int *smIDs;
    int count;
};

static const GPCInfo gpcTable[] = {
    { GPC0, (int)(sizeof(GPC0)/sizeof(GPC0[0])) },
    { GPC1, (int)(sizeof(GPC1)/sizeof(GPC1[0])) },
    { GPC2, (int)(sizeof(GPC2)/sizeof(GPC2[0])) },
    { GPC3, (int)(sizeof(GPC3)/sizeof(GPC3[0])) },
    { GPC4, (int)(sizeof(GPC4)/sizeof(GPC4[0])) },
    { GPC5, (int)(sizeof(GPC5)/sizeof(GPC5[0])) },
    { GPC6, (int)(sizeof(GPC6)/sizeof(GPC6[0])) }
};

// -----------------------------------------------------------------------------
// DEVICE FUNCTION: Get SM ID using inline assembly
// -----------------------------------------------------------------------------
__device__ __forceinline__ unsigned int get_smid(void) {
    unsigned int ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}

// -----------------------------------------------------------------------------
// KERNEL
// -----------------------------------------------------------------------------
__global__ void kernel(unsigned int *d_a0, 
                       unsigned int *d_value, 
                       unsigned char *dSelectedSM,
                       unsigned int *dSMMapping,
                       int matrixRows)  // number of rows in the CSV-derived matrix
{
    // 1D block of 1024 threads (32 warps)
    unsigned int tid   = threadIdx.x;
    unsigned int warp  = tid / 32;
    unsigned int lane  = tid % 32;
    unsigned int sm_id = get_smid();

    // Only work if this SM is selected.
    if(dSelectedSM[sm_id]) {
        // Use precomputed contiguous mapping for this SM.
        unsigned int myRank = dSMMapping[sm_id];

        // Compute the address from the address matrix.
        // The matrix is stored in column‑major order.
        unsigned int addr = d_value[warp + ((myRank % matrixRows) * BLOCK_SIZE)];
        unsigned int index = lane * 8 + addr * ADDRESS_BLOCK;
        volatile unsigned int kVal = 0;

        __syncthreads();

        for (int i = 0; i < ITERATION; i++) {
            for (int j = 0; j < 2; j++) {
                kVal += d_a0[index];  // each iteration: read 4 bytes
            }
        }
        d_a0[sm_id * ADDRESS_BLOCK] = kVal;

    }
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // New command-line usage:
    //    <CTAs per SM> <list_of_GPCs>
    if(argc < 3) {
        printf("Usage: %s <CTAs per SM> <list_of_GPCs>\n", argv[0]);
        printf("Example: %s 10 0 2 5\n", argv[0]);
        exit(1);
    }

    // 1) Parse CTAs per SM (first argument)
    int ctaspersm = atoi(argv[1]);
    if(ctaspersm < 1) {
        printf("Invalid number of CTAs per SM: %d\n", ctaspersm);
        return 1;
    }

    // 2) Parse chosen GPCs from remaining arguments.
    std::vector<int> chosenGPCs;
    for(int i = 2; i < argc; i++){
        int g = atoi(argv[i]);
        if(g < 0 || g > 6) {
            printf("Invalid GPC index: %d (must be between 0 and 6)\n", g);
            return 1;
        }
        chosenGPCs.push_back(g);
    }
    if(chosenGPCs.empty()){
        printf("No GPC specified.\n");
        return 1;
    }

    // 3) Initialize CUDA and get device properties.
    cudaSetDevice(0);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    int NUM_SM = device_prop.multiProcessorCount;

    // 4) Allocate global data buffer d_a0 based on L2 cache size.
    size_t l2Bytes = device_prop.l2CacheSize;
    size_t numElements = l2Bytes / sizeof(unsigned int); // use full L2 capacity (in 32-bit words)
    unsigned int *d_a0 = NULL;
    cudaMalloc((void**)&d_a0, sizeof(unsigned int) * numElements);

    // Initialize d_a0 with sequential numbers.
    unsigned int *h_a0 = (unsigned int*)malloc(sizeof(unsigned int) * numElements);
    for(size_t i = 0; i < numElements; i++){
        h_a0[i] = i;
    }
    cudaMemcpy(d_a0, h_a0, sizeof(unsigned int) * numElements, cudaMemcpyHostToDevice);
    free(h_a0);

    // 5) Determine CSV file for address matrix.
    //    The file name is chosen by the target (A100 vs. H100) and a compile-time PARTITION option.
#ifdef USE_H100
    const int matrixRows = 164;      // For H100, the CSV is expected to have 164 rows.
    std::string deviceName = "H100";
#else
    const int matrixRows = 64;       // For A100, the CSV is expected to have 64 rows.
    std::string deviceName = "A100";
#endif
    std::stringstream ssFilename;
    ssFilename << deviceName << "-" << PARTITION << ".csv";  // e.g., "A100-0.csv"
    std::string csvFileName = ssFilename.str();

    // 6) Read the address matrix from the CSV file.
    //    Expecting a file with 'matrixRows' rows and 32 comma-separated values per row.
    const int matrixCols = BLOCK_SIZE;  // always 32 columns.
    std::vector<unsigned int> h_matrixCSV(matrixRows * matrixCols);
    std::ifstream infile(csvFileName);
    if(!infile.is_open()){
        std::cerr << "Error opening CSV file: " << csvFileName << std::endl;
        return 1;
    }
    std::string line;
    int row = 0;
    while(std::getline(infile, line) && row < matrixRows) {
        std::istringstream linestream(line);
        std::string token;
        int col = 0;
        while(std::getline(linestream, token, ',') && col < matrixCols){
            h_matrixCSV[row * matrixCols + col] = static_cast<unsigned int>(std::stoul(token));
            col++;
        }
        if(col != matrixCols) {
            std::cerr << "Error: row " << row 
                      << " in CSV file does not have " << matrixCols << " columns." 
                      << std::endl;
            return 1;
        }
        row++;
    }
    if(row != matrixRows) {
        std::cerr << "Error: CSV file " << csvFileName 
                  << " does not have " << matrixRows << " rows." << std::endl;
        return 1;
    }
    infile.close();

    // 7) Allocate and copy the address matrix to device memory (d_value).
    unsigned int *d_value = NULL;
    cudaMalloc((void**)&d_value, sizeof(unsigned int) * matrixRows * matrixCols);
    cudaMemcpy(d_value, h_matrixCSV.data(), sizeof(unsigned int) * matrixRows * matrixCols,
               cudaMemcpyHostToDevice);

    // 8) Build a boolean array marking selected SMs based on chosen GPCs.
    unsigned char *hSelectedSM = (unsigned char*)calloc(NUM_SM, sizeof(unsigned char));
    for (int g : chosenGPCs) {
        for(int i = 0; i < gpcTable[g].count; i++){
            unsigned int sm = gpcTable[g].smIDs[i];
            if(sm < (unsigned int)NUM_SM)
                hSelectedSM[sm] = 1;
        }
    }
    unsigned char *dSelectedSM = NULL;
    cudaMalloc((void**)&dSelectedSM, NUM_SM * sizeof(unsigned char));
    cudaMemcpy(dSelectedSM, hSelectedSM, NUM_SM * sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    free(hSelectedSM);

    // 9) Precompute contiguous mapping for selected SMs.
    unsigned int *hSMMapping = (unsigned int*)malloc(NUM_SM * sizeof(unsigned int));
    unsigned int currentRank = 0;
    for(int i = 0; i < NUM_SM; i++){
        if(hSelectedSM[i])
            hSMMapping[i] = currentRank++;
        else
            hSMMapping[i] = 0xFFFFFFFF; // invalid value
    }
    unsigned int *dSMMapping = NULL;
    cudaMalloc((void**)&dSMMapping, NUM_SM * sizeof(unsigned int));
    cudaMemcpy(dSMMapping, hSMMapping, NUM_SM * sizeof(unsigned int),
               cudaMemcpyHostToDevice);
    free(hSMMapping);

    // 10) Configure kernel launch parameters.
    //     The grid dimensions now reflect: one block per physical SM (in first dim)
    //     and 'ctaspersm' blocks in the second dimension (i.e. CTAs per SM)
    dim3 gridDim(NUM_SM, ctaspersm);
    dim3 blockDim(32 * 32);  // 1024 threads per block (32 warps)

    // Set a shared-memory carveout if desired.
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    // 11) Time the kernel execution using CUDA events.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    // Launch kernel; pass matrixRows from the CSV.
    kernel<<<gridDim, blockDim>>>(d_a0, d_value, dSelectedSM, dSMMapping, matrixRows);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaDeviceSynchronize();

    // 12) Compute approximate L2-read bandwidth.
    int activeSM = 0;
    for (int g : chosenGPCs) {
        for (int i = 0; i < gpcTable[g].count; i++) {
            if(gpcTable[g].smIDs[i] < (unsigned int)NUM_SM)
                activeSM++;
        }
    }
    double timeSec  = elapsed_ms * 1e-3;
    // Total reads: activeSM * 32 warps * 32 threads * CTAperSM* ITERATION.
    long long totalReads = (long long)activeSM * 32 * 32 * ctaspersm *ITERATION;
    long long totalBytes = totalReads * sizeof(unsigned int);
    double bandwidthGBs  = (double)totalBytes / (timeSec * 1e9);

    printf("Time = %.3f ms\n", elapsed_ms);
    printf("Active SMs = %d\n", activeSM);
    printf("L2 read bandwidth = %.2f GB/s\n", bandwidthGBs);

    // 13) Cleanup.
    cudaFree(d_a0);
    cudaFree(d_value);
    cudaFree(dSelectedSM);
    cudaFree(dSMMapping);

    return 0;
}
