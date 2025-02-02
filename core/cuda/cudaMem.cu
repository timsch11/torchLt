#include "cudaMem.cuh"


float* reserveMemoryOnDevice(unsigned int size) {// declare pointer
    float* memoryAlloc = nullptr;

    // reserve actual space in memory, add some padding for thread efficiency
    cudaError_t allocStatus = cudaMalloc(&memoryAlloc, 
        (size + BLOCK_SIZE - (size % BLOCK_SIZE)) * sizeof(float));
    
    if (allocStatus != cudaSuccess || memoryAlloc == nullptr) {
        std::cout << std::string(cudaGetErrorString(allocStatus));
        return nullptr;
    }

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    return memoryAlloc;
}

__global__ void __initMemCell(float* d_memorySection, float value) {
    d_memorySection[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

float* zeros(unsigned int size) {
    // returns a pointer to (first element of) an array (interpretation of dimension is up to the caller) of specified size filled with zeros; array lives in unified memory (on cpu and gpu)

    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // reserve memory
    float* d_memoryAllocation = reserveMemoryOnDevice(blockThreadAllocation.first * blockThreadAllocation.second);

    if (d_memoryAllocation == nullptr) {
        printf("error when allocating memory");
        return nullptr;
    }

    // launch kernel
    __initMemCell<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_memoryAllocation, 0.0f);

    // check for errors
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cout << std::string(cudaGetErrorString(err));
        return nullptr;
    }

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return d_memoryAllocation;
}

float* constants(unsigned int size, float constant) {
    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // reserve memory
    float* d_memoryAllocation = reserveMemoryOnDevice(size);

    if (d_memoryAllocation == nullptr) {
        printf("error when allocating memory");
        return nullptr;
    }

    // launch kernel
    __initMemCell<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_memoryAllocation, constant);

     // check for errors
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cout << std::string(cudaGetErrorString(err));
        return nullptr;
    }

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return d_memoryAllocation;
}

cudaError_t constants(float* d_value, unsigned int size, float constant) {
    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // launch kernel
    __initMemCell<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_value, constant);

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __transposeMemDup(float* d_source, float* d_destination, int size) {
    int ind = size - blockIdx.x * blockDim.x + threadIdx.x;
    if (size >= 0) {
        d_destination[ind] = d_source[ind];
    }
}

__global__ void __memDup(float* d_source, float* d_destination) {
    // calculate index, block is responsible for arr[n] to arr[n+blockSize] elements to leverage coalescing access
    unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_destination[ind] = d_source[ind];
}

cudaError_t cudaMemDup(float* d_source, float* d_destination, unsigned int size, bool transpose) {
    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // select which kernel to use for copying
    if (transpose) {
        __transposeMemDup<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_source, d_destination, size);
    } else {
        __memDup<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_source, d_destination);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

// WEIGHT INITIALIZATION FUNCTIONS

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

cudaError_t weight_init(float* d_targetMemorySpace, unsigned int size, float scaling_factor, int seed) {

    // add some padding to ensure that kernel runs efficiently but also does not override other memory cells
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // run kernel
    __cuda_weight_init<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_targetMemorySpace, size, scaling_factor, seed);

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    return err;
}
