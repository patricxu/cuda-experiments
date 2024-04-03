#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Dimensions for matrix A (2x3), B (3x2), and hence C will be (2x2)
    int m = 2, n = 2, k = 3;
    float alpha = 1.0f, beta = 0.0f;

    // Define matrices A and B in row-major order
    float h_A[] = {1.0, 2.0, 3.0, // Matrix A
                   4.0, 5.0, 6.0};
    float h_B[] = {9.0, 8.0,       // Matrix B
                   7.0, 6.0,
                   5.0, 4.0};
    float h_C[4]; // Result matrix C will be stored here

    // Allocate device memory for matrices A, B, and C
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m*k*sizeof(float));
    cudaMalloc(&d_B, k*n*sizeof(float));
    cudaMalloc(&d_C, m*n*sizeof(float));

    // Copy matrices from the host to the device
    cublasSetMatrix(m, k, sizeof(float), h_A, m, d_A, m);
    cublasSetMatrix(k, n, sizeof(float), h_B, k, d_B, k);

    // Perform matrix multiplication with B transposed
    // Note that since B is being transposed, its dimensions are effectively swapped for the operation,
    // so we pass n as the leading dimension of B in cublasSgemm
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, m);

    // Copy the result back to the host memory
    cublasGetMatrix(m, n, sizeof(float), d_C, m, h_C, m);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    // Print the result matrix C
    std::cout << "Result matrix C:" << std::endl;
    for(int i = 0; i < m*n; ++i) {
        std::cout << h_C[i] << " ";
        if((i + 1) % n == 0) std::cout << std::endl;
    }

    return 0;
}
