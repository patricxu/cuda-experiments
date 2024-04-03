#include "merge_sort.h"
#include <iostream>
// #include <helper_cuda.h>
#include <cuda.h>


#define min(a, b) (a < b ? a : b)
// Based on https://github.com/kevin-albert/cuda-mergesort/blob/master/mergesort.cu

__host__ std::tuple<dim3, dim3, int> parseCommandLineArguments(int argc, char** argv) 
{
    int numElements = 32;
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'x':
                    toSet = &threadsPerBlock.x;
                    break;
                case 'y':
                    toSet = &threadsPerBlock.y;
                    break;
                case 'z':
                    toSet = &threadsPerBlock.z;
                    break;
                case 'X':
                    toSet = &blocksPerGrid.x;
                    break;
                case 'Y':
                    toSet = &blocksPerGrid.y;
                    break;
                case 'Z':
                    toSet = &blocksPerGrid.z;
                    break;
                case 'n':
                    i++;
                    numElements = std::stoi(argv[i]);
                    break;
            }
            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
    }
    return {threadsPerBlock, blocksPerGrid, numElements};
}

__host__ long *generateRandomLongArray(int numElements)
{
    //TODO generate random array of long integers of size numElements
    long *randomLongs;
    randomLongs = (long*)malloc(numElements * sizeof(long));

    for(int i = 0; i < numElements; i++)
    {
        randomLongs[i] = rand() % 1000;
    }

    return randomLongs;
}

__host__ void printHostMemory(long *host_mem, int num_elments)
{
    // Output results
    for(int i = 0; i < num_elments; i++)
    {
        printf("%ld ",host_mem[i]);
    }
    printf("\n");
}

__host__ int main(int argc, char** argv) 
{

    auto[threadsPerBlock, blocksPerGrid, numElements] = parseCommandLineArguments(argc, argv);

    long *data = generateRandomLongArray(numElements);

    printf("Unsorted data: ");
    printHostMemory(data, numElements);

    data = mergesort(data, numElements, threadsPerBlock, blocksPerGrid);

    printf("Sorted data: ");
    printHostMemory(data, numElements);
}

__host__ std::tuple <long* ,long* ,dim3* ,dim3*> allocateMemory(long* data, int numElements, dim3* threadsDim, dim3* blocksDim)
{
    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    long *D_data, *D_swp;
    dim3 *D_threads, *D_blocks;

    // Actually allocate the two arrays
    cudaMalloc(&D_data, numElements * sizeof(long));
    cudaMalloc(&D_swp, numElements * sizeof(long));
    cudaMalloc(&D_threads, sizeof(dim3));
    cudaMalloc(&D_blocks, sizeof(dim3));

    // Copy from our input list into the first array
    cudaMemcpy(D_data, data, numElements * sizeof(long), cudaMemcpyHostToDevice);

    // Copy the thread / block info to the GPU as well
    cudaMemcpy(D_threads, threadsDim, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, blocksDim, sizeof(dim3), cudaMemcpyHostToDevice);

    return {D_data, D_swp, D_threads, D_blocks};
}

__host__ long* mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid)
 {
    auto[D_data, D_swp, D_threads, D_blocks] = allocateMemory(data, size, &threadsPerBlock, &blocksPerGrid);
    long* tmp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    // TODO Initialize timing metrics variable(s). The implementation of this is up to you
    

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //

    // printf("size %ld nThreads %ld\n", size, nThreads);
    int i = 0;
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(D_data, D_swp, size, width, slices, D_threads, D_blocks); //TODO You will need to populate arguments for the kernel
        cudaDeviceSynchronize();

        // Switch the input / output arrays instead of copying them around
        tmp = D_data;
        D_data = D_swp;
        D_swp = tmp;

        i++;
        // if(width==2)
        // break;
    }

    cudaDeviceSynchronize();
    if(i%2 == 0)
        cudaMemcpy(data, D_swp, size * sizeof(long), cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(data, D_data, size * sizeof(long), cudaMemcpyDeviceToHost);
    // TODO calculate and print to stdout kernel execution time

    // Free the GPU memory
    cudaFree(D_data);
    cudaFree(D_swp);
    cudaFree(D_blocks);
    cudaFree(D_threads);
    return data;
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);

    // TODO initialize 3 long variables start, middle, and end
    // middle and end do not have values set,
    // while start is set to the width of the merge sort data span * the thread index * number of slices that this kernel will sort
    long start, middle, end;
    start = width * idx * slices;
    for (long slice = 0; slice < slices; slice++) {
        // Break from loop when the start variable is >= size of the input array
        if(start >= size)
            break;

        // Set middle to be minimum middle index (start index plus 1/2 width) and the size of the input array
        middle = min((start + (width >> 1)), size);
        // Set end to the minimum of the end index (start index plus the width of the current data window) and the size of the input array
        end = min((start + width), size);

        // printf("start %ld, middle %ld, end %ld, size %ld\n", start, middle, end, size);
        // Perform bottom up merege given the two available arrays and the start, middle, and end variables
        gpu_bottomUpMerge(source, dest, start, middle, end);
        // Increase the start index by the width of the current data window
        start += width;
    }
}

//
// Finally, sort something gets called by gpu_mergesort() for each slice
// Note that the pseudocode below is not necessarily 100% complete you may want to review the merge sort algorithm.
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;

    // Create a for look that iterates between the start and end indexes
    for (int k = start; k < end; k++) {
        // if i is before the middle index and (j is the final index or the value at i <  the value at j)
        if (i < middle && (j >= end || source[i] < source[j]) ) {
            // set the value in the destination array at index k to the value at index i in the source array
            dest[k] = source[i];
            // increment i
            i++;
        } else {
            // set the value in the destination array at index k to the value at index j in the source array
            dest[k] = source[j];
            // increment j
            j++;
        }
    }
}