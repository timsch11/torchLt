#include "curand_kernel.h"
#include "util.cu"


__global__ void initMemCell(float* d_memorySection, float value) {
    d_memorySection[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

// returns pointer to device array of <size> (+ padding) inialized with 0
float* zeros(unsigned int size) {
    // returns a pointer to (first element of) an array (interpretation of dimension is up to the caller) of specified size filled with zeros; array lives in unified memory (on cpu and gpu)

    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // reserve memory
    float* d_memoryAllocation;
    CHECK_CUDA_ERROR(cudaMalloc(&d_memoryAllocation, blockThreadAllocation.first * blockThreadAllocation.second * sizeof(float)));

    // launch kernel
    initMemCell<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_memoryAllocation, 0.0f);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());

    return d_memoryAllocation;
}

// returns pointer to device array of <size> (+ padding) inialized with <value>
float* constants(unsigned int size, float value) {

    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // reserve memory
    float* d_memoryAllocation;
    CHECK_CUDA_ERROR(cudaMalloc(&d_memoryAllocation, blockThreadAllocation.first * blockThreadAllocation.second * sizeof(float)));  // blockThreadAllocation.first * blockThreadAllocation.second = size + padding

    // launch kernel
    initMemCell<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_memoryAllocation, value);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());

    return d_memoryAllocation;
}

// returns pointer to allocated but uninitalized device array of <size> (+ padding)
float* reserveMemoryOnDevice(unsigned int size) {
    // declare pointer
    float* memoryAlloc;

    // reserve actual space in memory, add some padding for thread efficiency
    CHECK_CUDA_ERROR(cudaMalloc(&memoryAlloc, (size + (size % BLOCK_SIZE)) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // return pointer 
    return memoryAlloc;
}

// WEIGHT INITIALIZATION FUNCTIONS


__global__ void cuda_weight_init(float* weights, unsigned int size, float scalingFactor, int seed) {
    // declaring random state
    curandState state;

    // set index
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    // init curand
    curand_init(seed + ind, blockIdx.x, 0, &state);

    // set weight
    weights[ind] = curand_normal(&state) * sqrtf(scalingFactor);
}

// fills matrix of specified shape with values sampled from a scaled random normal distribution: N~(0, sqrt(<scalingFactor>))
void weight_init(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, float scaling_factor, int seed) {

    // set size, add some padding to ensure that kernel runs efficiently but also does not override other memory cells
    unsigned int size = in_features * out_features;
    size += size % BLOCK_SIZE;

    // run kernel
    cuda_weight_init<<<in_features, out_features>>>(d_targetMemorySpace, size, scaling_factor, seed);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}