#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

#ifndef TILE
#define TILE 32
#endif

// Baseline tiled matmul (no DSM)
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < N; t += TILE) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    C[row * N + col] = acc;
}

// DSMEM variant: blocks in a cluster share As (A tile), each block loads its own Bs
__global__ void matmul_dsmem(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             int N)
{
    extern __shared__ float smem[];
    float* As = smem;                      // TILE x TILE
    float* Bs = smem + TILE * TILE;        // TILE x TILE (private per block)

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int cRank = cluster.block_rank();

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < N; t += TILE) {
        // Block 0 in the cluster loads the As tile for this block-row and k-tile
        if (cRank == 0) {
            As[threadIdx.y * TILE + threadIdx.x] = A[row * N + (t + threadIdx.x)];
        }
        // Each block loads its own Bs tile
        Bs[threadIdx.y * TILE + threadIdx.x] = B[(t + threadIdx.y) * N + col];

        // Ensure As is ready across the cluster
        cluster.sync();

        // Map As from block 0 into every block in the cluster
        float* As_shared = cluster.map_shared_rank(As, 0);

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a = As_shared[threadIdx.y * TILE + k];
            float b = Bs[k * TILE + threadIdx.x];
            acc += a * b;
        }

        // Ensure all DSMEM users finished before next iteration overwrites tiles
        cluster.sync();
    }

    C[row * N + col] = acc;
}

static void checkCuda(cudaError_t e, const char* m) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e));
        exit(1);
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <cluster_size>\n", argv[0]);
        return 1;
    }
    const int N = atoi(argv[1]);
    const int cluster_size = atoi(argv[2]); // 0 => baseline

    // Allocate and init
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    for (int i = 0; i < N * N; ++i) { hA[i] = (i % 3) * 0.5f; hB[i] = (i % 5) * 0.25f; }

    float *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, bytes), "cudaMalloc A");
    checkCuda(cudaMalloc(&dB, bytes), "cudaMalloc B");
    checkCuda(cudaMalloc(&dC, bytes), "cudaMalloc C");
    checkCuda(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice), "H2D B");

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    if (cluster_size > 0 && (grid.x % cluster_size) != 0) {
        fprintf(stderr, "Error: grid.x=%u not divisible by cluster_size=%d\n", grid.x, cluster_size);
        return 2;
    }

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    if (cluster_size == 0) {
        cudaEventRecord(start);
        matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
        cudaEventRecord(stop);
    } else {
        // Configure cluster width along x so blocks with same y share As
        cudaFuncSetAttribute(matmul_dsmem, cudaFuncAttributeRequiredClusterWidth, cluster_size);
        cudaFuncSetAttribute(matmul_dsmem, cudaFuncAttributeRequiredClusterHeight, 1);
        cudaFuncSetAttribute(matmul_dsmem, cudaFuncAttributeRequiredClusterDepth, 1);

        cudaEventRecord(start);
        matmul_dsmem<<<grid, block, (size_t)(2 * TILE * TILE * sizeof(float))>>>(dA, dB, dC, N);
        cudaEventRecord(stop);
    }

    cudaEventSynchronize(stop);
    float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);

    // CSV: N,cluster_size,ms
    printf("%d,%d,%.6f\n", N, cluster_size, ms);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    return 0;
} 