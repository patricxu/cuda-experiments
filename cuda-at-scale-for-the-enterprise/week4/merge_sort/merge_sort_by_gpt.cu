#include <cuda_runtime.h>
#include <iostream>


__global__ void mergeKernel(int *d_array, int *d_temp, int size, int width) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int start = 2 * idx * width;
    int middle = start + width;
    int end = min(start + 2 * width, size);
    middle = middle < size ? middle : size;
    int i = start, j = middle, k = start;

    while (i < middle && j < end) {
        if (d_array[i] <= d_array[j]) {
            d_temp[k++] = d_array[i++];
        } else {
            d_temp[k++] = d_array[j++];
        }
    }

    while (i < middle) d_temp[k++] = d_array[i++];
    while (j < end) d_temp[k++] = d_array[j++];

    // Copy sorted temp array to original array
    for (int i = start; i < end; ++i) {
        d_array[i] = d_temp[i];
    }
}


// Assume mergeKernel is defined as above.

void cudaMergeSort(int *h_array, int size) {
    int *d_array, *d_temp;
    size_t bytes = size * sizeof(int);

    // Allocate memory on the device
    cudaMalloc(&d_array, bytes);
    cudaMalloc(&d_temp, bytes);

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);

    // Calculate number of threads and blocks
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    // Width of the subarrays to merge grows by powers of 2
    for (int width = 1; width < size; width *= 2) {
        mergeKernel<<<blocks, threads>>>(d_array, d_temp, size, width);
        cudaDeviceSynchronize();
    }

    // Copy sorted array back to host
    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_array);
    cudaFree(d_temp);
}

int main() {
    int h_array[128];
    for(int i=0; i<128; i++)
    {
        h_array[i] = rand() % 999;
    }
    int size = sizeof(h_array) / sizeof(h_array[0]);

    cudaMergeSort(h_array, size);

    // Print sorted array
    for (int i = 0; i < size; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}