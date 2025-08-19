// DSM-Optimized Matrix Multiplication with Configurable Thread Block Clusters
#define BLOCK_SIZE 16

#include <iostream>
#include <cstdlib>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

typedef struct {
    int width;
    int height;
    int stride;
    int* elements;
} Matrix;

__device__ int GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, int value) {
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// DSM-Optimized Kernel with larger effective tiles per cluster
__global__ void MatMulKernelDSM(Matrix A, Matrix B, Matrix C, int clusterSize) {
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int myRank = cluster.block_rank();
    
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    int Cvalue = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Shared memory for current block's tiles
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        
        // DSM optimization: All blocks participate in loading
        // Each block loads a portion of both A and B tiles
        int rowsPerBlock = BLOCK_SIZE / clusterSize;
        int myRowStart = myRank * rowsPerBlock;
        int myRowEnd = (myRank + 1) * rowsPerBlock;
        
        // Load assigned rows of both A and B tiles
        if (row >= myRowStart && row < myRowEnd) {
            As[row][col] = GetElement(Asub, row, col);
            Bs[row][col] = GetElement(Bsub, row, col);
        }
        
        // Cluster-wide synchronization
        cluster.sync();
        
        // Compute using DSM for remote data access
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            int a_val, b_val;
            
            // Determine which block owns As[row][e]
            int a_owner = row / rowsPerBlock;
            if (a_owner == myRank) {
                a_val = As[row][e];
            } else {
                int (*As_remote)[BLOCK_SIZE] = cluster.map_shared_rank(As, a_owner);
                a_val = As_remote[row][e];
            }
            
            // Determine which block owns Bs[e][col]  
            int b_owner = e / rowsPerBlock;
            if (b_owner == myRank) {
                b_val = Bs[e][col];
            } else {
                int (*Bs_remote)[BLOCK_SIZE] = cluster.map_shared_rank(Bs, b_owner);
                b_val = Bs_remote[e][col];
            }
            
            Cvalue += a_val * b_val;
        }
        
        __syncthreads();
        cluster.sync();
    }
    
    SetElement(Csub, row, col, Cvalue);
}

void MatMul(const Matrix A, const Matrix B, Matrix C, int clusterSize) {
    // Device memory allocation
    Matrix d_A, d_B, d_C;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    
    size_t sizeA = A.width * A.height * sizeof(int);
    size_t sizeB = B.width * B.height * sizeof(int);
    size_t sizeC = C.width * C.height * sizeof(int);
    
    cudaMalloc(&d_A.elements, sizeA);
    cudaMalloc(&d_B.elements, sizeB);
    cudaMalloc(&d_C.elements, sizeC);
    
    cudaMemcpy(d_A.elements, A.elements, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, B.elements, sizeB, cudaMemcpyHostToDevice);
    
    // Kernel launch configuration
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Configure for cluster launch
    cudaFuncSetAttribute(MatMulKernelDSM, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    
    cudaLaunchConfig_t config = {0};
    config.gridDim = dimGrid;
    config.blockDim = dimBlock;
    config.dynamicSmemBytes = 0;
    config.stream = 0;
    
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = clusterSize;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
    
    cudaEventRecord(start);
    cudaLaunchKernelEx(&config, MatMulKernelDSM, d_A, d_B, d_C, clusterSize);
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaMemcpy(C.elements, d_C.elements, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

bool testCorrectness(int clusterSize) {
    int size = 64;
    Matrix A, B, C, C_ref;
    A.width = A.height = A.stride = size;
    B.width = B.height = B.stride = size; 
    C.width = C.height = C.stride = size;
    C_ref.width = C_ref.height = C_ref.stride = size;
    
    A.elements = (int*)malloc(size * size * sizeof(int));
    B.elements = (int*)malloc(size * size * sizeof(int));
    C.elements = (int*)malloc(size * size * sizeof(int));
    C_ref.elements = (int*)malloc(size * size * sizeof(int));
    
    // Initialize matrices
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A.elements[i * size + j] = i + 1;
            B.elements[i * size + j] = j + 1;
        }
    }
    
    // CPU reference
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int sum = 0;
            for (int k = 0; k < size; k++) {
                sum += A.elements[i * size + k] * B.elements[k * size + j];
            }
            C_ref.elements[i * size + j] = sum;
        }
    }
    
    // GPU computation
    MatMul(A, B, C, clusterSize);
    
    // Verify
    bool correct = true;
    for (int i = 0; i < size * size; i++) {
        if (C.elements[i] != C_ref.elements[i]) {
            correct = false;
            break;
        }
    }
    
    free(A.elements); free(B.elements); free(C.elements); free(C_ref.elements);
    return correct;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <matrix_size> <cluster_size>" << std::endl;
        return 1;
    }
    
    int size = atoi(argv[1]);
    int clusterSize = atoi(argv[2]);
    
    if (size % BLOCK_SIZE != 0) {
        std::cout << "Matrix size must be multiple of " << BLOCK_SIZE << std::endl;
        return 1;
    }
    if (clusterSize < 2 || clusterSize > 8 || (clusterSize & (clusterSize - 1)) != 0) {
        std::cout << "Cluster size must be 2, 4, or 8" << std::endl;
        return 1;
    }
    
    // Test correctness
    if (!testCorrectness(clusterSize)) {
        std::cout << "Correctness test FAILED" << std::endl;
        return 1;
    }
    
    // Performance test
    Matrix A, B, C;
    A.width = A.height = A.stride = size;
    B.width = B.height = B.stride = size;
    C.width = C.height = C.stride = size;
    
    A.elements = (int*)malloc(size * size * sizeof(int));
    B.elements = (int*)malloc(size * size * sizeof(int));
    C.elements = (int*)malloc(size * size * sizeof(int));
    
    for (int i = 0; i < size * size; i++) {
        A.elements[i] = 1;
        B.elements[i] = 2;
    }
    
    MatMul(A, B, C, clusterSize);
    
    free(A.elements);
    free(B.elements);
    free(C.elements);
    
    return 0;
} 