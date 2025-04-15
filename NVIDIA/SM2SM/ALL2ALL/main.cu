/*
    OVERVIEW:
    ---------
    SM2SM Benchmark for various Traffic Patterns

    This benchmark target the DSM (Distributed Shared Memory) feature of the NVIDIA
    Hopper architecture. It measures the performance of different traffic patterns 
    between SMs in a GPC (Graphics Processing Cluster) using the SM-to-SM network.
    
    Each source block reads from one or more destination blocks in the same cluster.
    The benchmark supports the following traffic patterns:
        • TRAFFIC_RANDPERM   : Each block reads from a fixed random partner.
        • TRAFFIC_ROUNDROBIN : Each block reads from the next block in the cluster.
        • TRAFFIC_UNIFORM    : Each block reads from a different random partner on each iteration.
        • TRAFFIC_ALL2ALL    : Each block reads from all other blocks in the cluster.

    ALL2ALL traffic pattern can be configured to read the full shared memory size
    of each destination block or a reduced fraction (num_ints divided by (CLUSTER_SIZE-1)).
    The reduced fraction is used to ensure that each source block reads the same amount
    of data in total of the other traffic patterns.alignas
    It is possible to select this option at compile time using the READ_ALL2ALL_FULL parameter.

    The benchmark can measure either:
        • CALC_BW      : Overall bandwidth (total cycles).
        • CALC_LATENCY : Per-request latency.

    The access pattern is controlled by the following parameter:alignas
        • STRIDE       : Number of elements between threads in a block.
                         The default is 1, for streaming access.
                         A value greater that 1 results in bank conflicts.

    The benchmark can be configured to run with different cluster sizes and block sizes.
    The default is 16 blocks per cluster and 1024 threads per block.

    The benchmark support different ILP (Instruction Level Parallelism) levels.
    It is possible to select it at compile time using the ILP_FACTOR parameter.
    The benchmark can be run with different number of iterations.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// CONFIGURATION: Compile‑Time Knobs
//------------------------------------------------------------------------------
#ifndef CLUSTER_SIZE
    #define CLUSTER_SIZE 16    // Default: blocks per cluster.
#endif

#ifndef ILP_FACTOR
    #define ILP_FACTOR 8       // Default: unrolling factor.
#endif

#ifndef ITERATION
    #define ITERATION 10000   // Default: number of read loop iterations.
#endif

#ifndef STRIDE
    #define STRIDE 1           // Default: element stride between threads.
#endif

//------------------------------------------------------------------------------
// CONFIGURATION: Metric Choice (default is CALC_BW)
//------------------------------------------------------------------------------
#if !defined(CALC_BW) && !defined(CALC_LATENCY)
    #define CALC_BW
#endif

//------------------------------------------------------------------------------
// CONFIGURATION: Traffic Pattern Selection
//------------------------------------------------------------------------------
#if !defined(TRAFFIC_RANDPERM) && !defined(TRAFFIC_ROUNDROBIN) && \
    !defined(TRAFFIC_UNIFORM) && !defined(TRAFFIC_ALL2ALL)
    #define TRAFFIC_RANDPERM   // Default traffic pattern.
#endif

#if defined(TRAFFIC_RANDPERM) + defined(TRAFFIC_ROUNDROBIN) + \
    defined(TRAFFIC_UNIFORM) + defined(TRAFFIC_ALL2ALL) != 1
    #error "Define exactly one of: TRAFFIC_RANDPERM, TRAFFIC_ROUNDROBIN, TRAFFIC_UNIFORM, TRAFFIC_ALL2ALL"
#endif

namespace cg = cooperative_groups;

//------------------------------------------------------------------------------
// SM HELPER FUNCTION:
//    Retrieves the current Streaming Multiprocessor (SM) ID via inline assembly.
//------------------------------------------------------------------------------
__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

//------------------------------------------------------------------------------
// PARTNER MAP STORAGE:
//------------------------------------------------------------------------------
#if defined(TRAFFIC_RANDPERM) || defined(TRAFFIC_ROUNDROBIN)
    __device__ __constant__ int d_partner_map[CLUSTER_SIZE];
#endif
// For TRAFFIC_UNIFORM, the partner map pointer is passed from host.
// TRAFFIC_ALL2ALL does not require a partner map pointer.

/*
    KERNEL OVERVIEW:
    ----------------
    1. Warm Phase:
       • Each block writes its local shared memory.
    2. Read Phase (traffic pattern dependent):
       • TRAFFIC_RANDPERM    : Uses a fixed random partner per block.
       • TRAFFIC_ROUNDROBIN  : Uses partner = (rank + 1) mod CLUSTER_SIZE.
       • TRAFFIC_UNIFORM     : Selects a new uniform‑random partner on each iteration.
       • TRAFFIC_ALL2ALL     : Each block’s threads are partitioned so that, together,
                               they read from all other blocks in the cluster.
           -> For ALL2ALL, a compile‑time option (READ_ALL2ALL_FULL) chooses whether
              the read limit per partner is the full SMem size or a reduced portion.
    3. Metrics:
       • CALC_BW      : Measures overall bandwidth (total cycles).
       • CALC_LATENCY : Measures per‑request latency.
    4. Result Output:
       • The kernel writes the destination SM (or a marker), the source SM, and the metric.
         For TRAFFIC_ALL2ALL, the destination SM is set to 0xFFFFFFFF.
         For TRAFFIC_UNIFORM and fixed patterns, the destination SM is obtained by reading the partner block's shared memory.
*/
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void kernel(unsigned long long *out,
            int                num_ints,
            const int         *all_partner_maps) // Only used for TRAFFIC_UNIFORM.
{
    extern __shared__ int sdata[];  // Dynamic shared memory buffer.

    cg::cluster_group cluster = cg::this_cluster();
    const unsigned int rank    = cluster.block_rank();
    const int          cl_id   = blockIdx.x / CLUSTER_SIZE;
    const int          gblk_id = cl_id * CLUSTER_SIZE + rank;

    //------------------------------------------------------------------------------
    // WARM PHASE:
    //    Each block writes its local shared memory.
    //------------------------------------------------------------------------------
    if (threadIdx.x == 0) {
        unsigned int my_smid = get_smid();
        sdata[0] = my_smid;  // Mark the owner SM.
        for (int i = 1; i < num_ints; i++)
            sdata[i] = my_smid + i;
    }

#if defined(TRAFFIC_ALL2ALL)
    //------------------------------------------------------------------------------
    // ALL2ALL Traffic Pattern Setup:
    //    Partition threads so that each group serves a different partner block.
    //------------------------------------------------------------------------------
    const int partners   = CLUSTER_SIZE - 1;
    const int group_size = blockDim.x / partners;
    const int group_id   = threadIdx.x / group_size; // Identify which partner group.
    const int local_tid  = threadIdx.x % group_size;

    //--------------------------------------------------------------------------
    // ALL2ALL Read Limit Selection.
    //   If READ_ALL2ALL_FULL is defined, read the full shared memory 
    //   size of each destination;
    //   otherwise, read only a fraction (num_ints divided by (CLUSTER_SIZE-1))
    //   so that each source reads in total the full shared memory size.
    //--------------------------------------------------------------------------
    #ifdef READ_ALL2ALL_FULL
            int partner_read_limit = num_ints;
    #else
            int partner_read_limit = num_ints / (CLUSTER_SIZE - 1);
    #endif
#endif

    int partner_rank;  // Will hold the partner block’s rank.
#if defined(TRAFFIC_ROUNDROBIN)
    partner_rank = (rank + 1) % CLUSTER_SIZE;
#elif defined(TRAFFIC_RANDPERM)
    partner_rank = d_partner_map[rank];
#endif

    //------------------------------------------------------------------------------
    // TIMED LOOP:
    //    Execute the read phase based on the selected traffic pattern.
    //------------------------------------------------------------------------------
    int local_sum = 0;
#ifdef CALC_LATENCY
    unsigned long long lat_acc = 0, lat_cnt = 0;
#endif

    cluster.sync();

#ifdef CALC_BW
    unsigned long long startCycles = clock64();
#endif

#if defined(TRAFFIC_ALL2ALL)
    if (group_id < partners) {
        int partner_rank = (group_id < (int)rank) ? group_id : (group_id + 1);
        int *__restrict__ ws = cluster.map_shared_rank(sdata, partner_rank);
        for (int rep = 0; rep < ITERATION; rep++) {
            int idx = local_tid * STRIDE;
            for (; idx + (ILP_FACTOR - 1) * group_size * STRIDE < partner_read_limit;
                 idx += group_size * ILP_FACTOR * STRIDE) {
#pragma unroll
                for (int j = 0; j < ILP_FACTOR; j++)
                    local_sum += ws[idx + j * group_size * STRIDE];
            }
            for (; idx < partner_read_limit; idx += group_size * STRIDE)
                local_sum += ws[idx];
        }
    }
#else   // Single‑partner traffic patterns.
    for (int rep = 0; rep < ITERATION; rep++) {
#if defined(TRAFFIC_UNIFORM)
        // For Uniform Random, update partner_rank every iteration.
        partner_rank = all_partner_maps[rep * CLUSTER_SIZE + rank];
#endif
        int *__restrict__ ws = cluster.map_shared_rank(sdata, partner_rank);
        int idx = threadIdx.x * STRIDE;
        for (; idx + (ILP_FACTOR - 1) * blockDim.x * STRIDE < num_ints;
             idx += blockDim.x * ILP_FACTOR * STRIDE) {
#pragma unroll
            for (int j = 0; j < ILP_FACTOR; j++)
                local_sum += ws[idx + j * blockDim.x * STRIDE];
        }
        for (; idx < num_ints; idx += blockDim.x * STRIDE)
            local_sum += ws[idx];
    }
#endif  // End of traffic-pattern‑specific read loops.

    __syncthreads();

    //------------------------------------------------------------------------------
    // RESULTS COLLECTION:
    //    Store source and destination SM IDs along with the timing metric.
    //    For ALL2ALL, destination is set to 0xFFFFFFFF.
    //    For Uniform Random and fixed partner patterns, the destination SM is obtained
    //    by mapping the partner block's shared memory and reading its first element.
    //------------------------------------------------------------------------------
#ifdef CALC_BW
    unsigned long long totalCycles = clock64() - startCycles;
    unsigned int src_smid = get_smid();
    unsigned int dest_smid = 0u;
#if defined(TRAFFIC_ALL2ALL)
    dest_smid = 0xFFFFFFFF; // Marker indicating not applicable.
#else
    {
        // Obtain the partner block's shared memory pointer and read its SM id.
        int *__restrict__ partner_sdata = cluster.map_shared_rank(sdata, partner_rank);
        dest_smid = partner_sdata[0];
    }
#endif
    out[3 * gblk_id + 0] = dest_smid;   // Destination SM.
    out[3 * gblk_id + 1] = src_smid;      // Source SM.
    out[3 * gblk_id + 2] = totalCycles;   // Timing metric (cycles).
#else  // CALC_LATENCY branch.
    unsigned long long avgLat = lat_cnt ? (lat_acc / lat_cnt) : 0ULL;
    unsigned int src_smid = get_smid();
    unsigned int dest_smid = 0u;
#if defined(TRAFFIC_ALL2ALL)
    dest_smid = 0xFFFFFFFF;
#else
    {
        int *__restrict__ partner_sdata = cluster.map_shared_rank(sdata, partner_rank);
        dest_smid = partner_sdata[0];
    }
#endif
    int base = gblk_id * (blockDim.x + 2);
    out[base + 0] = dest_smid;
    out[base + 1] = src_smid;
    out[base + 2 + threadIdx.x] = avgLat;
#endif

    cluster.sync();
}

//------------------------------------------------------------------------------
// HOST CODE:
//    Sets up the device, builds partner maps, launches the kernel, and prints
//    detailed per‑block and per‑cluster results.
//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    /*
        CLI ARGUMENTS:
        --------------
        argv[1] : numClusters (number of clusters; default = 1)
        argv[2] : blockSize   (threads per block; default = 1024)
    */
    int numClusters = 1;
    int blockSize   = 1024;
    if (argc > 1) numClusters = atoi(argv[1]);
    if (argc > 2) blockSize   = atoi(argv[2]);

    //------------------------------------------------------------------------------
    // DEVICE INITIALIZATION:
    //    Select CUDA device and obtain its properties.
    //------------------------------------------------------------------------------
    int dev = 0; 
    cudaGetDevice(&dev);
    cudaDeviceProp prop; 
    cudaGetDeviceProperties(&prop, dev);

    int maxSMemOptin;
    cudaDeviceGetAttribute(&maxSMemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    size_t shared_bytes = maxSMemOptin;
    int num_ints = static_cast<int>(shared_bytes / sizeof(int));

    //------------------------------------------------------------------------------
    // BUILD PARTNER MAPS:
    //    For TRAFFIC_RANDPERM or TRAFFIC_ROUNDROBIN, build the partner map.
    //    For TRAFFIC_UNIFORM, generate a partner map matrix ensuring no self‑loops.
    //------------------------------------------------------------------------------
#if defined(TRAFFIC_RANDPERM) || defined(TRAFFIC_ROUNDROBIN)
    int h_partner_map[CLUSTER_SIZE];
#if defined(TRAFFIC_ROUNDROBIN)
    for (int i = 0; i < CLUSTER_SIZE; i++) 
        h_partner_map[i] = (i + 1) % CLUSTER_SIZE;
#else // TRAFFIC_RANDPERM
    for (int i = 0; i < CLUSTER_SIZE; i++) 
        h_partner_map[i] = i;
    srand((unsigned)time(NULL));
    for (int i = CLUSTER_SIZE - 1; i >= 1; i--) {
        int j = rand() % (i + 1);
        int tmp = h_partner_map[i];
        h_partner_map[i] = h_partner_map[j];
        h_partner_map[j] = tmp;
    }
    // Ensure no block is paired with itself.
    for (int i = 0; i < CLUSTER_SIZE; i++) {
        if (h_partner_map[i] == i) {
            int swap = (i + 1) % CLUSTER_SIZE;
            int tmp = h_partner_map[i];
            h_partner_map[i] = h_partner_map[swap];
            h_partner_map[swap] = tmp;
        }
    }
#endif
    cudaMemcpyToSymbol(d_partner_map, h_partner_map, CLUSTER_SIZE * sizeof(int));
    const int *d_all_partner_maps = nullptr;
#elif defined(TRAFFIC_UNIFORM)
    size_t partner_bytes = (size_t)ITERATION * CLUSTER_SIZE * sizeof(int);
    int *h_all_partner_maps = (int*)malloc(partner_bytes);
    srand((unsigned)time(NULL));
    for (int rep = 0; rep < ITERATION; rep++) {
        for (int i = 0; i < CLUSTER_SIZE; i++) {
            int r;
            do { r = rand() % CLUSTER_SIZE; } while (r == i);
            h_all_partner_maps[rep * CLUSTER_SIZE + i] = r;
        }
    }
    int *d_tmp; 
    cudaMalloc(&d_tmp, partner_bytes);
    cudaMemcpy(d_tmp, h_all_partner_maps, partner_bytes, cudaMemcpyHostToDevice);
    const int *d_all_partner_maps = d_tmp;
#else // TRAFFIC_ALL2ALL
    const int *d_all_partner_maps = nullptr;
#endif

    //------------------------------------------------------------------------------
    // OUTPUT ALLOCATION:
    //    Allocate memory for storing the kernel results.
    //------------------------------------------------------------------------------
#ifdef CALC_BW
    int total_blocks = numClusters * CLUSTER_SIZE;
    size_t out_size = total_blocks * 3 * sizeof(unsigned long long);
#else
    int total_blocks = numClusters * CLUSTER_SIZE;
    size_t out_size = total_blocks * (blockSize + 2) * sizeof(unsigned long long);
#endif
    unsigned long long *d_out, *h_out;
    h_out = (unsigned long long*)malloc(out_size);
    cudaMalloc(&d_out, out_size);

    //------------------------------------------------------------------------------
    // KERNEL ATTRIBUTES:
    //    Set preferred shared memory carveout and dynamic shared memory size.
    //------------------------------------------------------------------------------
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    //------------------------------------------------------------------------------
    // KERNEL LAUNCH:
    //    Launch the kernel with specified grid and block dimensions.
    //------------------------------------------------------------------------------
    int totalGridBlocks = numClusters * CLUSTER_SIZE;
    kernel<<< totalGridBlocks, blockSize, shared_bytes >>>(d_out, num_ints, d_all_partner_maps);
    cudaDeviceSynchronize();

    //------------------------------------------------------------------------------
    // RESULTS COPY & FORMATTED PRINTING:
    //    For each cluster, print one line per block showing:
    //      (Cluster id, Block id, Source SM, Destination SM, Metric)
    //    Then print an aggregate line per cluster.
    //------------------------------------------------------------------------------
    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);
    double clkHz = prop.clockRate * 1e3;   // kHz → Hz

#ifdef CALC_BW
    unsigned long long bytes_per_iter = (unsigned long long)num_ints * sizeof(int);
#if defined(TRAFFIC_ALL2ALL)
    bytes_per_iter *= (CLUSTER_SIZE - 1);     // Each partner contributes.
#endif
    unsigned long long total_bytes = bytes_per_iter * ITERATION;

    for (int cluster = 0; cluster < numClusters; cluster++) {
        printf("===== Cluster %d =====\n", cluster);
        double cluster_bw_sum = 0.0;
        double cluster_time_sum = 0.0;
        for (int block = 0; block < CLUSTER_SIZE; block++) {
            int global_id = cluster * CLUSTER_SIZE + block;
            unsigned long long cycles = h_out[3*global_id + 2];
            double bpc = (double)total_bytes / (double)cycles;
            double bw  = bpc * clkHz / 1e9;
            double exec_time_us = (double)cycles / clkHz * 1e6;
            cluster_bw_sum += bw;
            cluster_time_sum += exec_time_us;
            unsigned int dest_smid = (unsigned int) h_out[3*global_id + 0];
            unsigned int src_smid  = (unsigned int) h_out[3*global_id + 1];
            printf("Cluster %d, Block %2d | Src SM: %3u | Dest SM: ", cluster, block, src_smid);
            if (dest_smid == 0xFFFFFFFFu)
                printf("   N/A  ");
            else
                printf("%3u", dest_smid);
            printf(" | Cycles: %10llu | BW: %.4f GB/s | Exec Time: %.2f us\n",
                   cycles, bw, exec_time_us);
        }
        double cluster_avg_bw = cluster_bw_sum / CLUSTER_SIZE;
        double cluster_avg_time = cluster_time_sum / CLUSTER_SIZE;
        printf("Aggregate for Cluster %d: Avg BW: %.4f GB/s, Avg Exec Time: %.2f us\n\n",
               cluster, cluster_avg_bw, cluster_avg_time);
    }
#else  // CALC_LATENCY
    for (int cluster = 0; cluster < numClusters; cluster++) {
        printf("===== Cluster %d =====\n", cluster);
        double cluster_lat_sum = 0.0;
        for (int block = 0; block < CLUSTER_SIZE; block++) {
            int global_id = cluster * CLUSTER_SIZE + block;
            int base = global_id * (blockSize + 2);
            unsigned int dest_smid = (unsigned int) h_out[base + 0];
            unsigned int src_smid  = (unsigned int) h_out[base + 1];
            unsigned long long sum_lat = 0;
            for (int t = 0; t < blockSize; t++) {
                sum_lat += h_out[base + 2 + t];
            }
            double avg_lat = (double)sum_lat / blockSize;
            cluster_lat_sum += avg_lat;
            printf("Cluster %d, Block %2d | Src SM: %3u | Dest SM: ", cluster, block, src_smid);
            if (dest_smid == 0xFFFFFFFFu)
                printf("   N/A  ");
            else
                printf("%3u", dest_smid);
            printf(" | Avg Latency: %.2f cycles\n", avg_lat);
        }
        double cluster_avg_lat = cluster_lat_sum / CLUSTER_SIZE;
        printf("Aggregate for Cluster %d: Avg Latency: %.2f cycles\n\n", cluster, cluster_avg_lat);
    }
#endif

    //------------------------------------------------------------------------------
    // CLEANUP:
    //    Free allocated memory.
    //------------------------------------------------------------------------------
    cudaFree(d_out);
#if defined(TRAFFIC_UNIFORM)
    cudaFree((void*)d_all_partner_maps); 
    free(h_all_partner_maps);
#endif
    free(h_out);
    return 0;
}
