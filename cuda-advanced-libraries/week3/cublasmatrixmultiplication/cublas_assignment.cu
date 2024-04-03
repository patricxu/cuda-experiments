#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <cublas_v2.h>

#define HA 2
#define WA 9
#define WB 2
#define HB WA 
#define WC WB   
#define HC HA 
#define M HA
#define N WB
#define K WA
#define index(i,j,ld) (((j)*(ld))+(i))

void printMat(float*P,int uWP,int uHP){
  int i,j;
  for(i=0;i<uHP;i++){

      printf("\n");

      for(j=0;j<uWP;j++)
          printf("%f ",P[index(i,j,uHP)]);
  }
}

// __host__ float* initializeHostMemory(int height, int width, bool random, float nonRandomValue) {
//   // TODO allocate host memory of type float of size height * width called hostMatrix

//   // TODO fill hostMatrix with either random data (if random is true) else set each value to nonRandomValue

//   return hostMatrix;
// }

__host__ float *initializeDeviceMemoryFromHostMemory(int height, int width, float *hostMatrix) {
  // TODO allocate device memory of type float of size height * width called deviceMatrix
  float *deviceMatrix;
  int nElements = height * width * sizeof(float);
  cudaMalloc((void**)&deviceMatrix, nElements);

  // TODO set deviceMatrix to values from hostMatrix
  cudaMemcpy(deviceMatrix, hostMatrix, nElements, cudaMemcpyHostToDevice);
  return deviceMatrix;
}

__host__ float *retrieveDeviceMemory(int height, int width, float *deviceMatrix, float *hostMemory) {
  // TODO get matrix values from deviceMatrix and place results in hostMemory
  cudaMemcpy(hostMemory, deviceMatrix, height * width * sizeof(float), cudaMemcpyDeviceToHost);
  return hostMemory;
}

__host__ void printMatrices(float *A, float *B, float *C){
  printf("\nMatrix A:\n");
  printMat(A,WA,HA);
  printf("\n");
  printf("\nMatrix B:\n");
  printMat(B,WB,HB);
  printf("\n");
  printf("\nMatrix C:\n");
  printMat(C,WC,HC);
  printf("\n");
}

__host__ int freeMatrices(float *A, float *B, float *C, float *AA, float *BB, float *CC){
  free( A );  free( B );  free ( C );
  cublasStatus status = cublasFree(AA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasFree(BB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }
  status = cublasFree(CC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int  main (int argc, char** argv) {
  cublasStatus status;
  cublasInit();

  // TODO initialize matrices A and B (2d arrays) of floats of size based on the HA/WA and HB/WB to be filled with random data
  float *A = new float[HA * WA];
  float *B = new float[HB * WB];
  float *C = new float[HC * WC];
  float *AA;
  float *BB;
  float *CC;

  for(int i = 0; i < HA * WA; i++){
      // A[i] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
      A[i] = i;
  }

  for(int i = 0; i < HB * WB; i++){
      // B[i] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
      B[i] = i;
  }

  if( A == 0 || B == 0){
    return EXIT_FAILURE;
  } else {
    // TODO create arrays of floats C filled with random value
    for(int i = 0; i < HC * WC; i++){
      C[i] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
    }

    // TODO create arrays of floats alpha filled with 1's
    float alpha = 1.0;
    // TODO create arrays of floats beta filled with 0's
    float beta = 0.0;

    // TODO use initializeDeviceMemoryFromHostMemory to create AA from matrix A
    AA = initializeDeviceMemoryFromHostMemory(HA, WA, A);
    // TODO use initializeDeviceMemoryFromHostMemory to create BB from matrix B
    BB = initializeDeviceMemoryFromHostMemory(HB, WB, B);
    // TODO use initializeDeviceMemoryFromHostMemory to create CC from matrix C
    CC = initializeDeviceMemoryFromHostMemory(HC, WC, C);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // TODO perform Single-Precision Matrix to Matrix Multiplication, GEMM, on AA and BB and place results in CC
    cublasSgemm(
      handle, 
      CUBLAS_OP_T, CUBLAS_OP_T, 
      M, N, K, 
      &alpha,
      AA, K, 
      BB, N,
      &beta, 
      CC, M
    );


    C = retrieveDeviceMemory(HC, WC, CC, C);

    printMatrices(A, B, C);

    freeMatrices(A, B, C, AA, BB, CC);
    
    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! shutdown error (A)\n");
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

}
