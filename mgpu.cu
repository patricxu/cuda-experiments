#include <cstdint>
#include <iostream>
#include "helpers.cuh"
#include "encryption.cuh"

void encrypt_cpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters, bool parallel=true) {

    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        data[entry] = permute64(entry, num_iters);
}

__global__ 
void decrypt_gpu(uint64_t * data, uint64_t num_entries, 
                 uint64_t num_iters) {

    const uint64_t thrdID = blockIdx.x*blockDim.x+threadIdx.x;
    const uint64_t stride = blockDim.x*gridDim.x;

    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        data[entry] = unpermute64(data[entry], num_iters);
}

bool check_result_cpu(uint64_t * data, uint64_t num_entries,
                      bool parallel=true) {

    uint64_t counter = 0;

    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        counter += data[entry] == entry;

    return counter == num_entries;
}

int main (int argc, char * argv[]) {

    Timer timer;
    Timer overall;

    const uint64_t num_entries = 1UL << 26;
    const uint64_t num_iters = 1UL << 10;
    const bool openmp = true;


    uint64_t num_gpu;
    cudaGetDeviceCount((int*)&num_gpu);
    uint64_t* data_gpu[num_gpu];

    timer.start();
    uint64_t * data_cpu;
    cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
    timer.stop("allocate memory");
    check_last_error();

    timer.start();
    encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
    timer.stop("encrypt data on CPU");

    overall.start();
    const uint64_t chunckSize = sdiv(num_entries, num_gpu);
    for (int gpu = 0; gpu < num_gpu; gpu++){
        cudaSetDevice(gpu);

        int lower = gpu * chunckSize;
        int upper = min(lower + chunckSize, num_entries);
        int width = upper - lower;
        cudaMalloc(&data_gpu[gpu], sizeof(uint64_t)*width);

        cudaMemcpy(
            data_gpu[gpu], 
            data_cpu+lower, 
            sizeof(uint64_t)*width, 
            cudaMemcpyHostToDevice
        );
        
        decrypt_gpu<<<80*32, 64>>>(data_gpu[gpu], width, num_iters);

        cudaMemcpy(
            data_cpu+lower,
            data_gpu[gpu],
            width*sizeof(uint64_t),
            cudaMemcpyDeviceToHost
        );
    }
    cudaDeviceSynchronize();
    overall.stop("MGPU HtoD->kernel->DtoH");

    timer.start();
    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << ( success ? "passed" : "failed")
              << std::endl;
    timer.stop("checking result on CPU");

    timer.start();
    cudaFreeHost(data_cpu);
    for (int gpu = 0; gpu < num_gpu; gpu++)
        cudaFree(data_gpu[gpu]);
    timer.stop("free memory");
    check_last_error();
}
