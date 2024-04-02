#include <stdio.h>
#define MAX_RANGE 9999


__global__ void matrixAdd(float* ma, float* mb, float* mc, int nElements)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if(threadId < nElements)
  {
    mc[threadId] = ma[threadId] + mb[threadId];
  }

}


// Based on https://cuda-programming.blogspot.com/2013/01/how-to-query-device-properties-and.html
int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  printf("Number of GPU Devices: %d\n", nDevices);
  
  // You will need to track the minimum or maximum for one or more device properties, so initialize them here
  int currentChosenDeviceNumber = -1; // Will not choose a device by default 
  int maxSharedMem = 0;

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Device Compute Major: %d Minor: %d\n", prop.major, prop.minor);
    printf("  Max Thread Dimensions: [%d][%d][%d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Device Clock Rate (KHz): %d\n", prop.clockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Registers Per Block: %d\n", prop.regsPerBlock);
    printf("  Registers Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("  Shared Memory Per Block: %zu\n", prop.sharedMemPerBlock);
    printf("  Shared Memory Per Multiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("  Total Constant Memory (bytes): %zu\n", prop.totalConstMem);
    printf("  Total Global Memory (bytes): %zu\n", prop.totalGlobalMem);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    // You can set the current chosen device property based on tracked min/max values
    if(maxSharedMem < prop.sharedMemPerBlock)
    {
      currentChosenDeviceNumber = i;
      maxSharedMem = prop.sharedMemPerBlock;
    }
  }

  // Create logic to actually choose the device based on one or more device properties
  cudaSetDevice(currentChosenDeviceNumber);

    // Print out the chosen device as below
  printf("The chosen GPU device has an index of: %d\n",currentChosenDeviceNumber); 

  int nElements = 1000;
  int threadPerBlock = 32;
  int nGrid = (nElements + threadPerBlock)/threadPerBlock + 1;
  float *h_ma, *h_mb, *h_mc, *d_ma, *d_mb, *d_mc;
  int size = nElements * sizeof(float);
  h_ma = (float*)malloc(size);
  h_mb = (float*)malloc(size);
  h_mc = (float*)malloc(size);

  for(int i = 0; i < nElements; i++)
  {
    h_ma[i] = rand() % MAX_RANGE;
    h_mb[i] = rand() % MAX_RANGE;
    h_mc[i] = 0.0;
  }

  cudaMalloc((void**)&d_ma, size);
  cudaMalloc((void**)&d_mb, size);
  cudaMalloc((void**)&d_mc, size);

  cudaMemcpy(d_ma, h_ma, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mb, h_mb, size, cudaMemcpyHostToDevice);

  matrixAdd<<<nGrid, threadPerBlock>>>(d_ma, d_mb, d_mc, nElements);
  cudaDeviceSynchronize();

  cudaMemcpy(h_mc, d_mc, size, cudaMemcpyDeviceToHost);

  cudaFree(d_ma);
  cudaFree(d_mb);
  cudaFree(d_mc);


  free(h_ma);
  free(h_mb);
  free(h_mc);

  return 0;
}