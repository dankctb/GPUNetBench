#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define CLUSTER_SIZE 8

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
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
// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication with DSMEM and timing
float MatMul(const Matrix A, const Matrix B, Matrix C)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    Matrix d_A, d_B, d_C;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    size_t sharedMemSize = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    
    cudaEventRecord(start);
    MatMulKernel<<<dimGrid, dimBlock, sharedMemSize>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements); cudaFree(d_B.elements); cudaFree(d_C.elements);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return milliseconds;
}
// DSMEM Matrix multiplication kernel with cluster size 8
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    float Cvalue = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    extern __shared__ float Bs_dsmem[];
    float (*Bs)[BLOCK_SIZE] = (float (*)[BLOCK_SIZE])Bs_dsmem;
    
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        As[row][col] = GetElement(Asub, row, col);
        
        // Only block 0 in cluster loads B tile to DSMEM
        if (cluster.block_rank() == 0) {
            Matrix Bsub = GetSubMatrix(B, m, blockCol);
            Bs[row][col] = GetElement(Bsub, row, col);
        }
        
        __syncthreads();
        cluster.sync();
        
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        
        __syncthreads();
        cluster.sync();
    }
    SetElement(Csub, row, col, Cvalue);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
    if (N % BLOCK_SIZE != 0) {
        printf("Matrix size must be multiple of %d\n", BLOCK_SIZE);
        return 1;
    }
    
    Matrix A, B, C;
    A.width = A.stride = A.height = N;
    B.width = B.stride = B.height = N;
    C.width = C.stride = C.height = N;
    
    A.elements = (float*)malloc(N * N * sizeof(float));
    B.elements = (float*)malloc(N * N * sizeof(float));
    C.elements = (float*)malloc(N * N * sizeof(float));
    
    for (int i = 0; i < N * N; i++) {
        A.elements[i] = (float)(rand() % 100) / 10.0f;
        B.elements[i] = (float)(rand() % 100) / 10.0f;
    }
    
    float time_ms = MatMul(A, B, C);
    printf("%d,%.3f\n", N, time_ms);
    
    free(A.elements); free(B.elements); free(C.elements);
    return 0;
}