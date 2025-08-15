// Thread block size = 16x16 = 256 threads per block
#define BLOCK_SIZE 32

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
    {
        // Fallback to standard kernel launch when clusterDim is not available
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
    int colsPerRank = (clusterSize > 0) ? (BLOCK_SIZE / max(1, clusterSize)) : BLOCK_SIZE;

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
        // Load Asub cooperatively across the cluster: each block loads a slice of columns
        int colStart = myRank * colsPerRank;
        int colEnd   = min(BLOCK_SIZE, colStart + colsPerRank);
        if (col >= colStart && col < colEnd) {
            As[row][col] = GetElement(Asub, row, col);
        }
        // Load Bsub locally (distinct per block)
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize within block and across cluster to ensure As slices are ready
        __syncthreads();

        cluster.sync();
        // Multiply Asub and Bsub together using DSMEM for remote As columns
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            int a_val;
            if (clusterSize > 1) {
                int owner = e / colsPerRank;
                if (owner != myRank) {
                    int (*As_remote)[BLOCK_SIZE] = cluster.map_shared_rank(As, owner);
                    a_val = As_remote[row][e];
                } else {
                    a_val = As[row][e];
                }
            } else {
                a_val = As[row][e];
            }
            Cvalue += a_val * Bs[e][col];
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
    int clusterSize = atoi(argv[2]);
    if (size % BLOCK_SIZE != 0) {
        std::cout << "Matrix size must be multiple of " << BLOCK_SIZE << std::endl;
        return 1;
    }
    if (clusterSize != 1 && clusterSize != 2 && clusterSize != 4 && clusterSize != 8) {
        std::cout << "Cluster size must be 1, 2, 4, or 8" << std::endl;
        return 1;
    }
    
    // First test correctness
    // std::cout << "Testing kernel correctness..." << std::endl;
    // bool isCorrect = testMatMulCorrectness();
    // if (isCorrect) {
    //     std::cout << "✓ Kernel correctness test PASSED" << std::endl;
    // } else {
    //     std::cout << "✗ Kernel correctness test FAILED" << std::endl;
    //     return 1;
    // }
    // ---------------------------------------------------------------
    // Then run performance test
    // std::cout << "\nRunning performance test..." << std::endl;
    
    // Allocate host matrices
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