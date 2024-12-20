#include "curand_kernel.h"
#include "util.cu"


/**
 * @brief kernel for constant value initalization
 * @param d_memorySection pointer to memory section that is to be initalized
 * @param value value to fill in
 */
__global__ void initMemCell(float* d_memorySection, float value) {
    d_memorySection[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

/**
 * @brief returns pointer to newly created device array of <size> (+ padding) inialized with 0
 * @param size size of array to be initalized
 */
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

/**
 * @brief returns pointer to newly created device array of <size> (+ padding) inialized with <value>
 * @param size size of array to be initalized
 * @param value value to fill in
 */
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

/**
 * @brief returns pointer to allocated but uninitalized device float array of <size> (+ padding)
 * @param size size of memory section
 */
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

/**
 * @brief kernel for random initalization
 * @param weights pointer to (padded) memory section that is to be initalized
 * @param size size of tensor (=number of total elements)
 * @param scaling_factor variance of the normal distribution
 * @param seed determines the seed for the curand normal dist. function
 */
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

/**
 * @brief fills tensor of specified size with values sampled from a scaled random normal distribution: N~(0, sqrt(<scalingFactor>))
 * @param d_targetMemorySpace pointer to (padded) memory section that is to be initalized
 * @param size size of tensor (=number of total elements)
 * @param scaling_factor variance of the normal distribution
 * @param seed determines the seed for the curand normal dist. function
 */
void weight_init(float* d_targetMemorySpace, unsigned int size, float scaling_factor, int seed) {

    // add some padding to ensure that kernel runs efficiently but also does not override other memory cells
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // run kernel
    cuda_weight_init<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_targetMemorySpace, size, scaling_factor, seed);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}