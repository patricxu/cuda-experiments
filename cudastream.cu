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
    const uint64_t nStream = 5;
    cudaStream_t streams[nStream];

    timer.start();
    uint64_t * data_cpu, * data_gpu;
    cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
    cudaMalloc    (&data_gpu, sizeof(uint64_t)*num_entries);
    timer.stop("allocate memory");
    check_last_error();

    timer.start();
    encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
    timer.stop("encrypt data on CPU");

    overall.start();
    const uint64_t chunckSize = sdiv(num_entries, nStream);
    for (int i = 0; i < nStream; i++){
        cudaStreamCreate(&streams[i]);

        int lower = i * chunckSize;
        int upper = min(lower + chunckSize, num_entries);
        int width = upper - lower;
        cudaMemcpyAsync(
            data_gpu+lower,
            data_cpu+lower,
            width*sizeof(uint64_t),
            cudaMemcpyHostToDevice,
            streams[i]
        );
        
        decrypt_gpu<<<80*32, 64, 0, streams[i]>>>(data_gpu+lower, width, num_iters);

        cudaMemcpyAsync(
            data_cpu+lower,
            data_gpu+lower,
            width*sizeof(uint64_t),
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }
    cudaDeviceSynchronize();
    overall.stop("stream HtoD->kernel->DtoH");

    timer.start();
    for (int i = 0; i < nStream; i++)
        cudaStreamDestroy(streams[i]);
    timer.stop("stream destroy");

    timer.start();
    const bool success = check_result_cpu(data_cpu, num_entries, openmp);
    std::cout << "STATUS: test " 
              << ( success ? "passed" : "failed")
              << std::endl;
    timer.stop("checking result on CPU");

    timer.start();
    cudaFreeHost(data_cpu);
    cudaFree    (data_gpu);
    timer.stop("free memory");
    check_last_error();
}
