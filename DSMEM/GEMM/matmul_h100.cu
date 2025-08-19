// Thread block size = 16x16 = 256 threads per block
#define BLOCK_SIZE 16
#define CLUSTER_SIZE 4

#include <iostream>
#include <cstdlib>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    int* elements;
} Matrix;
// Get a matrix element
__device__ int GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           int value)
{
    A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix, int);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C, int clusterSize)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(int);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(int);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(int);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    
    // Create timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    // Allow non-portable cluster size and launch with cluster dimension on H100
    cudaFuncSetAttribute(MatMulKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    
    // Record start time
    cudaEventRecord(start);
    
    if (clusterSize > 1) {
        // Use cluster launch for DSMEM features
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
        
        cudaLaunchKernelEx(&config, MatMulKernel, d_A, d_B, d_C, clusterSize);
    } else {
        // Standard kernel launch for single block clusters
        MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, clusterSize);
    }
    
    cudaEventRecord(stop);

    // Check for kernel launch errors and synchronize
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Post-kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Calculate and print execution time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    std::cout << milliseconds << std::endl;
    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C, int clusterSize)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Cluster group (H100 DSMEM)
    cg::cluster_group cluster = cg::this_cluster();
    int myRank = cluster.block_rank();

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    int Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        
        // Shared memory used to store Asub and Bsub respectively
        __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        // Optimized loading: each tile loaded once and divided among 4 blocks
        // For A tile: blocks 0,1 load first half rows, blocks 2,3 load second half rows
        int rowStart = (myRank < 2) ? 0 : BLOCK_SIZE/2;
        int rowEnd = (myRank < 2) ? BLOCK_SIZE/2 : BLOCK_SIZE;
        
        // Load Asub cooperatively - each block loads assigned rows
        if (row >= rowStart && row < rowEnd) {
            As[row][col] = GetElement(Asub, row, col);
        }
        
        // For B tile: blocks 0,2 load first half cols, blocks 1,3 load second half cols  
        int colStart = (myRank % 2 == 0) ? 0 : BLOCK_SIZE/2;
        int colEnd = (myRank % 2 == 0) ? BLOCK_SIZE/2 : BLOCK_SIZE;
        
        // Load Bsub cooperatively - each block loads assigned columns
        if (col >= colStart && col < colEnd) {
            Bs[row][col] = GetElement(Bsub, row, col);
        }
        
        // Synchronize within block and across cluster to ensure tiles are ready
        __syncthreads();
        cluster.sync();
        
        // Multiply Asub and Bsub together using DSMEM for remote data
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            int a_val, b_val;
            
            // Access A value (use DSMEM if not in my block's loaded rows)
            int a_owner = (e < BLOCK_SIZE/2) ? (myRank < 2 ? myRank : myRank - 2) : (myRank < 2 ? myRank + 2 : myRank);
            if (a_owner != myRank) {
                int (*As_remote)[BLOCK_SIZE] = cluster.map_shared_rank(As, a_owner);
                a_val = As_remote[row][e];
            } else {
                a_val = As[row][e];
            }
            
            // Access B value (use DSMEM if not in my block's loaded columns)
            int b_owner = (e < BLOCK_SIZE/2) ? (myRank % 2 == 0 ? myRank : myRank - 1) : (myRank % 2 == 0 ? myRank + 1 : myRank);
            if (b_owner != myRank) {
                int (*Bs_remote)[BLOCK_SIZE] = cluster.map_shared_rank(Bs, b_owner);
                b_val = Bs_remote[e][col];
            } else {
                b_val = Bs[e][col];
            }
            
            Cvalue += a_val * b_val;
        }
        
        // Synchronize before next tile load
        __syncthreads();
        cluster.sync();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

// Test function to check kernel correctness
bool testMatMulCorrectness() {
    int size = 32; // Small size for quick testing
    
    // Allocate host matrices
    Matrix A, B, C, C_ref;
    A.width = A.height = A.stride = size;
    B.width = B.height = B.stride = size;
    C.width = C.height = C.stride = size;
    C_ref.width = C_ref.height = C_ref.stride = size;
    
    A.elements = (int*)malloc(size * size * sizeof(int));
    B.elements = (int*)malloc(size * size * sizeof(int));
    C.elements = (int*)malloc(size * size * sizeof(int));
    C_ref.elements = (int*)malloc(size * size * sizeof(int));
    
    // Initialize matrices with known values
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A.elements[i * size + j] = i + 1;
            B.elements[i * size + j] = j + 1;
        }
    }
    
    // Compute reference result on CPU
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int sum = 0;
            for (int k = 0; k < size; k++) {
                sum += A.elements[i * size + k] * B.elements[k * size + j];
            }
            C_ref.elements[i * size + j] = sum;
        }
    }
    
    // Compute GPU result
    MatMul(A, B, C, 1);
    
    // Compare results
    bool correct = true;
    for (int i = 0; i < size * size; i++) {
        if (C.elements[i] != C_ref.elements[i]) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": GPU=" << C.elements[i] 
                     << " CPU=" << C_ref.elements[i] << std::endl;
            break;
        }
    }
    
    // Clean up
    free(A.elements);
    free(B.elements);
    free(C.elements);
    free(C_ref.elements);
    
    return correct;
}

// Test main function
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <matrix_size> <cluster_size>" << std::endl;
        return 1;
    }
    
    int size = atoi(argv[1]);
    //int clusterSize = atoi(argv[2]);
    int clusterSize = CLUSTER_SIZE;
    if (size % BLOCK_SIZE != 0) {
        std::cout << "Matrix size must be multiple of " << BLOCK_SIZE << std::endl;
        return 1;
    }
    if (clusterSize != 1 && clusterSize != 2 && clusterSize != 4 && clusterSize != 8) {
        std::cout << "Cluster size must be 1, 2, 4, or 8" << std::endl;
        return 1;
    }
    
    // First test correctness
    std::cout << "\nTesting kernel correctness..." << std::endl;
    bool isCorrect = testMatMulCorrectness();
    if (isCorrect) {
        std::cout << "\n✓ Kernel correctness test PASSED" << std::endl;
    } else {
        std::cout << "\n✗ Kernel correctness test FAILED" << std::endl;
        return 1;
    }
    ---------------------------------------------------------------
    Then run performance test
    std::cout << "\nRunning performance measurement..." << std::endl;
    
    Allocate host matrices

    Matrix A, B, C;
    A.width = A.height = A.stride = size;
    B.width = B.height = B.stride = size;
    C.width = C.height = C.stride = size;
    
    A.elements = (int*)malloc(size * size * sizeof(int));
    B.elements = (int*)malloc(size * size * sizeof(int));
    C.elements = (int*)malloc(size * size * sizeof(int));
    
    // Initialize matrices with simple values
    for (int i = 0; i < size * size; i++) {
        A.elements[i] = 1;
        B.elements[i] = 2;
    }
    
    // Perform matrix multiplication
    MatMul(A, B, C, clusterSize);
    
    // Clean up
    free(A.elements);
    free(B.elements);
    free(C.elements);
    
    return 0;
}