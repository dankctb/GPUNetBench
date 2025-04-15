/*
    OVERVIEW:
    ---------
    Mixed‑Traffic Benchmark (L2 vs. SM‑to‑SM)

    This benchmark runs two traffic branches inside a GPC (using Thread Block Cluster):
        • L2 Traffic   : Each block stream‑reads a warmed global buffer that 
                         is resident in the L2 cache.
        • DSM Traffic  : Each block reads from another block’s shared memory 
                         through the SM‑to‑SM interconnect.
    
    CLI Arguments:
        • Role mapping per rank determines the branch for each block:
            - 0 = idle, 1 = L2, 2 = DSM.
          Example: "0,1,0,0,0,0,2,1,0,0,0,0,2,1,0,0" for one 16‑block cluster.
    
    Compile‑time Knobs (set using -D<FLAG>=value):
        • CALC_LATENCY       : Measure per‑thread latency (default is CALC_BW).
        • DSM_DEST_OTHER_CPC : Choose DSM destinations from a different CPC 
                               (if unset, the destination stays in the same CPC).
        • READ_FULL  : In DSM, read the full shared memory size per destination 
                               (if unset, each destination reads SMEM/numDest).
    
    CPC / RANK MAP (H100 16‑rank cluster):
        - CPC0 : ranks { 4, 5, 10, 11 }
        - CPC1 : ranks { 2, 3, 8, 9, 14, 15 }
        - CPC2 : ranks { 0, 1, 6, 7, 12, 13 }
    
    METRICS:
        • CALC_BW      : 2 output values per block (source SM, cycles).
        • CALC_LATENCY : (blockDim.x+1) values per block (source SM, average latency per thread).
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <iostream>
#include <algorithm>

namespace cg = cooperative_groups;

//------------------------------------------------------------------------------
// CONFIGURATION: Compile‑Time Defaults
//------------------------------------------------------------------------------
#ifndef CLUSTER_SIZE
    #define CLUSTER_SIZE 16              // Default: Blocks per cluster (H100 default)
#endif

#ifndef ITERATIONS
    #define ITERATIONS 10000             // Outer repetitions for DSM, ×4 for L2 branch.
#endif

#ifndef L2_ITERATIONS
    #define L2_ITERATIONS 4              // additional multiplier repetitions for L2 branch.
#endif

#ifndef ILP_L2
    #define ILP_L2 8                     // Unrolling factor for L2 loop.
#endif
#ifndef ILP_DSM
    #define ILP_DSM 8                    // Unrolling factor for DSM loop.
#endif

#ifndef ENABLE_L2
    #define ENABLE_L2 1                  // Enable L2 branch.
#endif
#ifndef ENABLE_SM2SM
    #define ENABLE_SM2SM 1               // Enable DSM branch.
#endif

#if !defined(CALC_BW) && !defined(CALC_LATENCY)
    #define CALC_BW                    // Default metric: overall bandwidth.
#endif

//------------------------------------------------------------------------------
// DEVICE HELPERS:
//    Retrieves the current Streaming Multiprocessor (SM) ID via inline assembly.
__device__ __forceinline__ unsigned get_smid() {
    unsigned smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

//------------------------------------------------------------------------------
// DEVICE HELPERS:
//    Determines the CPC ID based on the rank.
__host__ __device__ __forceinline__ int cpc_id(int r) {
    return (r==0 || r==1 || r==6 || r==7 || r==12 || r==13) ? 2 :
           (r==2 || r==3 || r==8 || r==9 || r==14 || r==15) ? 1 : 0;
}

//------------------------------------------------------------------------------
// KERNEL OVERVIEW:
//    Executes the mixed traffic benchmark using two branches (L2 and DSM).
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void kernel(unsigned long long *d_l2,
            unsigned long long *d_dsm,
            unsigned int       *d_smid,
            int                *gmem,
            int                 gmem_ints,
            const int          *role,
            int                 smem_ints,
            int                *d_numDest)
{
    cg::cluster_group cluster = cg::this_cluster();
    const int rank   = cluster.block_rank();
    const int cl_id  = blockIdx.x / CLUSTER_SIZE;
    const int gblk_id= cl_id * CLUSTER_SIZE + rank;
    const int myRole = role[rank];

    //------------------------------------------------------------------------------
    // RECORD SM ID:
    //    Each block stores its SM ID in global memory.
    if (threadIdx.x == 0)
        d_smid[gblk_id] = get_smid();

    //------------------------------------------------------------------------------
    // DYNAMIC SHARED MEMORY:
    //    Obtain pointer to the block’s dynamic shared memory.
    extern __shared__ int sdata[];

#ifdef CALC_LATENCY
    unsigned long long latAcc = 0, latCnt = 0;
#endif

    //------------------------------------------------------------------------------
    // SHARED MEMORY INITIALIZATION:
    //    Initialize shared memory with unique values per block.
    for (int i = threadIdx.x; i < smem_ints; i += blockDim.x)
        sdata[i] = i + rank;
    __syncthreads();

    //------------------------------------------------------------------------------
    // L2 WARM‑UP PHASE:
    //    Warm up the global memory using stream‑reads.
    int acc = 0, tmp = 0;
    for (int i = 0; i < gmem_ints; i++) {
        int a = (threadIdx.x + i * blockDim.x) % gmem_ints;
        acc += gmem[a];
        tmp += acc;
    }
    gmem[threadIdx.x % gmem_ints] = tmp;
    __syncthreads();

    //------------------------------------------------------------------------------
    // DSM DESTINATION LIST & GROUP ASSIGNMENT:
    //    Build a list of eligible destination block ranks for DSM traffic.
    int destRank[CLUSTER_SIZE];
    int destCnt = 0;
    // Declare a pointer to the remote block’s shared memory.
    int *__restrict__ remote = nullptr;
    if (myRole == 2) {
        const int myCPC = cpc_id(rank);
        for (int r = 0; r < CLUSTER_SIZE; r++) {
            if (r == rank || role[r] != 2)
                continue;
#ifdef DSM_DEST_OTHER_CPC
            if (cpc_id(r) == myCPC)
                continue;
#else
            if (cpc_id(r) != myCPC)
                continue;
#endif
            destRank[destCnt++] = r;
        }
    }

    d_numDest[0] = destCnt;

#ifdef READ_FULL
    // In this mode, each block reads the entire shared memory of all destination blocks.
    int read_per_dest = smem_ints;
#else
    // In this mode, each block reads a portion of the shared memory of each destination block.
    int read_per_dest = smem_ints / destCnt;
#endif

    const int groupSize = (destCnt > 0) ? blockDim.x / destCnt : blockDim.x;
    const int groupId   = (destCnt > 0) ? threadIdx.x / groupSize   : 0;
    const int localTid  = (destCnt > 0) ? threadIdx.x % groupSize   : threadIdx.x;
    if (myRole == 2) {
        remote = (groupId < destCnt) ? cluster.map_shared_rank(sdata, destRank[groupId]) : nullptr;
    }

    //------------------------------------------------------------------------------
    // CLUSTER SYNCHRONIZATION:
    //    Ensure that all blocks are synchronized before beginning the benchmark.
    cluster.sync();

    //------------------------------------------------------------------------------
    // L2 TRAFFIC BRANCH:
    //    Run the L2 branch to measure bandwidth or latency by reading from global memory.
    if (myRole == 1 && ENABLE_L2) {
#ifdef CALC_BW
        unsigned long long tStart = clock64();
#endif
        for (int rep = 0; rep < ITERATIONS * L2_ITERATIONS; ++rep) {
            int i = threadIdx.x;
            for (; i + (ILP_L2 - 1) * blockDim.x < smem_ints; i += blockDim.x * ILP_L2) {
#ifdef CALC_LATENCY
                unsigned long long t0 = clock();
#endif
#pragma unroll
                for (int j = 0; j < ILP_L2; j++)
                    acc += gmem[i + j * blockDim.x];
#ifdef CALC_LATENCY
                unsigned long long t1 = clock();
                latAcc += (t1 - t0);
                latCnt++;
#endif
            }
#ifdef CALC_LATENCY
            unsigned long long t0 = clock();
#endif
            for (; i < smem_ints; i += blockDim.x)
                acc += gmem[i];
#ifdef CALC_LATENCY
            unsigned long long t1 = clock();
            latAcc += (t1 - t0);
            latCnt++;
#endif
        }

        __syncthreads();
#ifdef CALC_BW
        unsigned long long tStop = clock64();
        d_l2[gblk_id] = tStop - tStart;
#else
        int base = gblk_id * (blockDim.x + 1);
        d_l2[base] = get_smid();
        d_l2[base + 1 + threadIdx.x] = latCnt ? latAcc / latCnt : 0ULL;
#endif

        gmem[threadIdx.x] = acc;
    }
    //------------------------------------------------------------------------------
    // DSM TRAFFIC BRANCH:
    //    Run the DSM branch to measure bandwidth or latency using SM‑to‑SM communication.
    else if (myRole == 2 && ENABLE_SM2SM) {
#ifdef CALC_BW
        unsigned long long tStart = clock64();
#endif
        for (int rep = 0; rep < ITERATIONS; ++rep) {
            if (groupId < destCnt) {
                int idx = localTid;
                for (; idx + (ILP_DSM - 1) * groupSize < read_per_dest; idx += groupSize * ILP_DSM) {
#ifdef CALC_LATENCY
                    unsigned long long t0 = clock();
#endif
#pragma unroll
                    for (int j = 0; j < ILP_DSM; j++)
                        acc += remote[idx + j * groupSize];
#ifdef CALC_LATENCY
                    unsigned long long t1 = clock();
                    latAcc += (t1 - t0);
                    latCnt++;
#endif
                }
                for (; idx < read_per_dest; idx += groupSize) {
#ifdef CALC_LATENCY
                    unsigned long long t0 = clock();
#endif
                    acc += remote[idx];
#ifdef CALC_LATENCY
                    unsigned long long t1 = clock();
                    latAcc += (t1 - t0);
                    latCnt++;
#endif
                }
            }
        }
        __syncthreads();
#ifdef CALC_BW
        unsigned long long tStop = clock64();
        d_dsm[gblk_id] = tStop - tStart;

#else
        int base = gblk_id * (blockDim.x + 1);
        d_dsm[base] = get_smid();
        d_dsm[base + 1 + threadIdx.x] = latCnt ? latAcc / latCnt : 0ULL;
#endif

        sdata[threadIdx.x] = acc;
    }

    cluster.sync();
}

//------------------------------------------------------------------------------
// HOST CODE OVERVIEW:
//    Initializes the device, allocates memory, configures the kernel, launches it, 
//    and prints the resulting measurements.
int main(int argc, char **argv)
{
    /*
        CLI ARGUMENTS:
        --------------
        argv[1] : numClusters (number of clusters; default = 1)
        argv[2] : blockSize   (threads per block; default = 1024)
        argv[3] : role mapping for each block in a comma‑separated list
                  Example: "0,1,0,0,0,0,2,1,0,0,0,0,2,1,0,0" 
                  (Roles: 0 = idle, 1 = L2, 2 = DSM)
    */
    const int numClusters = (argc > 1) ? atoi(argv[1]) : 1;
    const int blockSize   = (argc > 2) ? atoi(argv[2]) : 1024;

    int role[CLUSTER_SIZE] = {0};
    if (argc > 3) {
        char *tok = strtok(argv[3], ",");
        int i = 0;
        while (tok && i < CLUSTER_SIZE) {
            role[i++] = atoi(tok);
            tok = strtok(nullptr, ",");
        }
    } else {
        // Default mapping for CPC2.
        for (int r = 0; r < CLUSTER_SIZE; r++) {
            if (cpc_id(r) == 2)
                role[r] = (r == 1 || r == 7 || r == 13) ? 1 :
                          (r == 0 || r == 6 || r == 12) ? 2 : 0;
        }
    }

    //------------------------------------------------------------------------------
    // DEVICE INITIALIZATION:
    //    Set the CUDA device, retrieve its properties, and calculate memory sizes.
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int maxSMemOptin;
    cudaDeviceGetAttribute(&maxSMemOptin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    const size_t smemBytes = maxSMemOptin;
    const int smemInts = smemBytes / sizeof(int);
    size_t l2Bytes = (prop.l2CacheSize > (1 << 20)) ? prop.l2CacheSize - (1 << 20) : prop.l2CacheSize;
    const int gmemInts = l2Bytes / sizeof(int);
    int *numDest = (int *)malloc(sizeof(int));

    //------------------------------------------------------------------------------
    // MEMORY ALLOCATION:
    //    Allocate global buffers, initialize device memory, and copy the role mapping.
    const int totalBlocks = numClusters * CLUSTER_SIZE;
#ifdef CALC_BW
    const size_t outBytes = totalBlocks * sizeof(unsigned long long);
#else
    const size_t outBytes = totalBlocks * (blockSize + 1) * sizeof(unsigned long long);
#endif
    unsigned long long *d_l2, *d_dsm;
    cudaMalloc(&d_l2, outBytes);
    cudaMemset(d_l2, 0, outBytes);
    cudaMalloc(&d_dsm, outBytes);
    cudaMemset(d_dsm, 0, outBytes);
    unsigned int *d_smid;
    cudaMalloc(&d_smid, totalBlocks * sizeof(unsigned int));
    int *d_gmem;
    cudaMalloc(&d_gmem, l2Bytes);
    cudaMemset(d_gmem, 0, l2Bytes);
    int *d_role;
    cudaMalloc(&d_role, CLUSTER_SIZE * sizeof(int));
    cudaMemcpy(d_role, role, CLUSTER_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    int *d_numDest;
    cudaMalloc(&d_numDest, sizeof(int));

    //------------------------------------------------------------------------------
    // KERNEL ATTRIBUTES:
    //    Set dynamic shared memory size and allow a nonportable cluster size.
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    //------------------------------------------------------------------------------
    // KERNEL LAUNCH:
    //    Launch the kernel with the computed grid and block dimensions.
    dim3 grid(totalBlocks);
    dim3 blk(blockSize);
    kernel<<<grid, blk, smemBytes>>>(d_l2, d_dsm, d_smid, d_gmem, gmemInts, d_role, smemInts, d_numDest);
    cudaDeviceSynchronize();

    //------------------------------------------------------------------------------
    // RESULTS COPY & PRINT:
    //    Copy back the measurements from device memory and print the per‑block metrics.
#ifdef CALC_BW
        std::vector<unsigned long long> h_l2(totalBlocks), h_dsm(totalBlocks);
#else
        std::vector<unsigned long long> h_l2(totalBlocks * (blockSize + 1)), h_dsm(totalBlocks * (blockSize + 1));
#endif
    std::vector<unsigned int> h_smid(totalBlocks);
    cudaMemcpy(h_l2.data(), d_l2, outBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dsm.data(), d_dsm, outBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_smid.data(), d_smid, totalBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numDest, d_numDest, sizeof(int), cudaMemcpyDeviceToHost);

    const double clkHz = prop.clockRate * 1000.0;
#ifdef CALC_BW
    const double bytesPerBlockL2 = (double)smemBytes * ITERATIONS * L2_ITERATIONS;
#ifdef READ_FULL
    const double bytesPerBlockDSM = (double)smemBytes * ITERATIONS * numDest[0];
#else
    const double bytesPerBlockDSM = (double)smemBytes * ITERATIONS;
#endif
    printf("\n=== BW ===\n");
    for (int c = 0; c < numClusters; c++) {
        printf("Cluster %d\n", c);
        for (int r = 0; r < CLUSTER_SIZE; r++) {
            if (role[r]) {
                int gid = c * CLUSTER_SIZE + r;
                double secs = (role[r] == 1) ? h_l2[gid] / clkHz : h_dsm[gid] / clkHz;
                double bw   = (role[r] == 1) ? (bytesPerBlockL2 / secs) / 1e9 : (bytesPerBlockDSM / secs) / 1e9;
                printf("  Rank%2d SM%3u  %s BW %.2f GB/s\n",
                       r, h_smid[gid], (role[r] == 1 ? "L2 " : "DSM"), bw);
            }
        }
    }
#else
    printf("\n=== LATENCY (avg cycles) ===\n");
    for (int c = 0; c < numClusters; c++) {
        for (int r = 0; r < CLUSTER_SIZE; r++) {
            if (role[r]) {
                int gid = c * CLUSTER_SIZE + r;
                int base = gid * (blockSize + 1);
                printf("Cluster%d Rank%2d SM%3u %s Average Latency", c, r, (unsigned)h_l2[base],
                       (role[r] == 1) ? "L2" : "DSM");
                const unsigned long long *ptr = (role[r] == 1) ? &h_l2[base + 1] : &h_dsm[base + 1];
                unsigned long long sum = 0;
                for (int t = 0; t < blockSize; t++)
                    sum += ptr[t];
                double avg = (double)sum / blockSize;
                printf(" %.2f clock cycles\n", avg);
            }
        }
    }
#endif
    return 0;
}
