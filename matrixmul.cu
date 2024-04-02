#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


//http://www.techdarting.com/2014/03/matrix-multiplication-in-cuda-using.html


// This code assumes that your device support block size of 1024
#define MAX_RANGE 9999

const unsigned int TILE_WIDTH = 32;


#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];   // Tile size of 32x32
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int ph = 0; ph < (((numAColumns - 1) / TILE_WIDTH) + 1); ph++) {
        if ((Row < numARows) && (threadIdx.x + (ph * TILE_WIDTH)) < numAColumns) {
            sA[threadIdx.y][threadIdx.x] = A[(Row * numAColumns) + threadIdx.x + (ph * TILE_WIDTH)];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (Col < numBColumns && (threadIdx.y + ph * TILE_WIDTH) < numBRows) {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + ph * TILE_WIDTH) * numBColumns + Col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; ++j) {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < numCRows && Col < numCColumns) {
        C[Row * numCColumns + Col] = Cvalue;
    }
}

void matMultiplyOnHost(float *A, float *B, float *C, int numARows,
                       int numAColumns, int numBRows, int numBColumns,
                       int numCRows, int numCColumns) {
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numAColumns; j++) {
            C[i * numCColumns + j] = 0.0;
            for (int k = 0; k < numCColumns; k++) {
                C[i * numCColumns + j] += A[i * numAColumns + k] * B[k * numBColumns + j];
            }
        }
    }
    return;
}

int main(int argc, char **argv) {
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *hostComputedC;
    float *deviceA;
    float *deviceB;
    float *deviceC;

    // Please adjust rows and columns according to you need.
    int numARows = 512; // number of rows in the matrix A
    int numAColumns = 512; // number of columns in the matrix A
    int numBRows = 512; // number of rows in the matrix B
    int numBColumns = 512; // number of columns in the matrix B

    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    hostA = (float *) malloc(sizeof(float) * numARows * numAColumns);
    hostB = (float *) malloc(sizeof(float) * numBRows * numBColumns);

    for (int i = 0; i < numARows * numAColumns; i++) {
        hostA[i] = (rand() % MAX_RANGE) / 2.0;
    }
    for (int i = 0; i < numBRows * numBColumns; i++) {
        hostB[i] = (rand() % MAX_RANGE) / 2.0;
    }

    // Setting numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;

    hostC = (float *) malloc(sizeof(float) * numCRows * numCColumns);
    hostComputedC = (float *) malloc(sizeof(float) * numCRows * numCColumns);

    // Allocating GPU memory
    gpu_errchk(cudaMalloc((void **) &deviceA, sizeof(float) * numARows * numAColumns));
    gpu_errchk(cudaMalloc((void **) &deviceB, sizeof(float) * numBRows * numBColumns));
    gpu_errchk(cudaMalloc((void **) &deviceC, sizeof(float) * numCRows * numCColumns));

    // Copy memory to the GPU 
    gpu_errchk(cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice));
    gpu_errchk(cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice));

    // Initialize the grid and block dimensions 
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((numCColumns / TILE_WIDTH) + 1, (numCRows / TILE_WIDTH) + 1, 1);

    //@@ Launch the GPU Kernel here
    matrixMultiplyShared <<<dimGrid, dimBlock>>>
                                       (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaError_t err1 = cudaPeekAtLastError();
    cudaDeviceSynchronize();
    printf("Got CUDA error ... %s \n", cudaGetErrorString(err1));

    // Copy the results in GPU memory back to the CPU    
    gpu_errchk(cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost));

    matMultiplyOnHost(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    for (int i = 0; i < numCColumns * numCRows; i++) {
        if (hostComputedC[i] != hostC[i]) {
            printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns,
                   i % numCColumns, hostComputedC[i], hostC[i]);
            break;
        }
    }
    // Free the GPU memory
    gpu_errchk(cudaFree(deviceA));
    gpu_errchk(cudaFree(deviceB));
    gpu_errchk(cudaFree(deviceC));

    free(hostA);
    free(hostB);
    free(hostC);
    free(hostComputedC);

    return 0;
}