#include "curand_kernel.h"
#include "util.cu"

/**
 * @brief returns pointer to allocated but uninitalized device float array of <size> (+ padding)
 * @param size size of memory section
 */
float* reserveMemoryOnDevice(unsigned int size) {
    // declare pointer
    float* memoryAlloc;

    // reserve actual space in memory, add some padding for thread efficiency
    CHECK_CUDA_ERROR(cudaMalloc(&memoryAlloc, (size + BLOCK_SIZE - (size % BLOCK_SIZE)) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // return pointer 
    return memoryAlloc;
}


/**
 * @brief kernel for constant value initalization
 * @param d_memorySection pointer to memory section that is to be initalized
 * @param value value to fill in
 */
__global__ void __initMemCell(float* d_memorySection, float value) {
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
    float* d_memoryAllocation = reserveMemoryOnDevice(blockThreadAllocation.first * blockThreadAllocation.second);

    // launch kernel
    __initMemCell<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_memoryAllocation, 0.0f);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());

    return d_memoryAllocation;
}

/**
 * @brief returns pointer to newly created device array of <size> (+ padding) inialized with <value>
 * @param size size of array to be initalized
 * @param value value to fill in
 */
float* constants(unsigned int size, float constant) {

    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // reserve memory
    float* d_memoryAllocation = reserveMemoryOnDevice(size);

    // launch kernel
    __initMemCell<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_memoryAllocation, constant);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());

    return d_memoryAllocation;
}

/**
 * @brief Fills already existing array with a constant value
 * @param size Size of array 
 * @param d_value Pointer to array to be filled with constants
 * @param constant Value to fill in
 */
void constants(float* d_value, unsigned int size, float constant) {

    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // launch kernel
    __initMemCell<<<blockThreadAllocation.first, blockThreadAllocation.second, 0, 0>>>(d_value, constant);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError()); 
}

/**
 * @brief Duplicates and transposes a matrix stored in device memory.
 *
 * This CUDA kernel transposes a square matrix of <size> stored in 
 * device memory
 *
 * @param d_source Pointer to the source matrix in device memory.
 * @param d_destination Pointer to the destination (result) in device memory).
 * @param size The size of the matrix (number of total entries).
 */
__global__ void __transposeMemDup(float* d_source, float* d_destination, int size) {
    int ind = size - blockIdx.x * blockDim.x + threadIdx.x;
    if (size >= 0) {
        d_destination[ind] = d_source[ind];
    }
}

/**
 * @brief Duplicates a tensor stored in device memory
 *
 * This CUDA kernel a tensor of <size> stored in 
 * device memory
 *
 * @param d_source Pointer to the source tensor in device memory.
 * @param d_destination Pointer to the destination tensor in device memory.
 * @param size The size of the matrix (number of total entries).
 */
__global__ void __memDup(float* d_source, float* d_destination) {
    // calculate index, block is responsible for arr[n] to arr[n+blockSize] elements to leverage coalescing access
    unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_destination[ind] = d_source[ind];
}

/**
 * @brief Duplicates an array on device memory
 * @param d_source Pointer to source array (that should be duplicated)
 * @param d_destination Pointer to destination array (that should hold result)
 * @param size Size of array to be duplicated
 * @param transpose Whether destination array should be the transposed version of source
 */

void cudaMemDup(float* d_source, float* d_destination, unsigned int size, bool transpose) {
    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // select which kernel to use for copying
    if (transpose) {
        __transposeMemDup<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_source, d_destination, size);
    } else {
        __memDup<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_source, d_destination);
    }

    // error checking 
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// WEIGHT INITIALIZATION FUNCTIONS

/**
 * @brief kernel for random initalization
 * @param weights pointer to (padded) memory section that is to be initalized
 * @param size size of tensor (=number of total elements)
 * @param scaling_factor variance of the normal distribution
 * @param seed determines the seed for the curand normal dist. function
 */
__global__ void __cuda_weight_init(float* weights, unsigned int size, float scalingFactor, int seed) {
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
    __cuda_weight_init<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_targetMemorySpace, size, scaling_factor, seed);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}