// Memory management utilities for CUDA operations
#include "cudaMem.cuh"

// Helper function to allocate device memory with padding for alignment
float* reserveMemoryOnDevice(unsigned int size) {
    float* memoryAlloc = nullptr;

    // reserve actual space in memory, add some padding for thread efficiency
    cudaError_t allocStatus = cudaMalloc(&memoryAlloc, 
        (size + BLOCK_SIZE - (size % BLOCK_SIZE)) * sizeof(float));
    
    if (allocStatus != cudaSuccess || memoryAlloc == nullptr) {
        std::cout << std::string(cudaGetErrorString(allocStatus));
        return nullptr;
    }
    
    return memoryAlloc;
}

// Initialize device memory with zeros
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

    return d_memoryAllocation;
}

cudaError_t constants(float* d_value, unsigned int size, float constant) {
    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // launch kernel
    __initMemCell<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_value, constant);

    // check for errors
    cudaError_t err = cudaGetLastError();

    return err;
}

__global__ void __memDup(float* d_source, float* d_destination) {
    // calculate index, block is responsible for arr[n] to arr[n+blockSize] elements to leverage coalescing access
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_destination[i] = d_source[i];
}

__global__ void __memDupTranspose(float* d_source, float* d_destination, unsigned int rows, unsigned int cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        unsigned int row = i / cols;
        unsigned int col = i % cols;
        d_destination[col * rows + row] = d_source[i];
    }
}

cudaError_t cudaMemDup(float* d_source, float* d_destination, unsigned int rows, unsigned int cols, bool transpose) {
    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(rows * cols);

    cudaError_t err;

    // select which kernel to use for copying
    if (transpose) {
        __memDupTranspose<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_source, d_destination, rows, cols);
        err = cudaGetLastError();
    } else {
        err = cudaMemcpyAsync(d_destination, d_source, rows * cols, cudaMemcpyDeviceToDevice, 0);
        //__memDup<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_source, d_destination);
    }

    return err;
}

__global__ void __memDupScaled(float* d_source, float* d_destination, float* scalar) {
    // calculate index, block is responsible for arr[n] to arr[n+blockSize] elements to leverage coalescing access
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_destination[i] = *scalar * d_source[i];
}

cudaError_t cudaMemDupScaled(float* d_source, float* d_destination, float* scalar, unsigned int size) {
    // calc block/thread allocation scheme 
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // copy and scale on the run
    __memDupScaled<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_source, d_destination, scalar);

    // check for errors
    cudaError_t err = cudaGetLastError();

    return err;
}

__global__ void __memDupScaled(float* d_source, float* d_destination, float scalar) {
    // calculate index, block is responsible for arr[n] to arr[n+blockSize] elements to leverage coalescing access
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_destination[i] = scalar * d_source[i];
}

cudaError_t cudaMemDupScaled(float* d_source, float* d_destination, float scalar, unsigned int size) {
    // calc block/thread allocation scheme 
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // copy and scale on the run
    __memDupScaled<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_source, d_destination, scalar);

    // check for errors
    cudaError_t err = cudaGetLastError();

    return err;
}

// WEIGHT INITIALIZATION FUNCTIONS

__global__ void __cuda_weight_init(float* weights, unsigned int size, float scalingFactor, int seed) {
    // declaring random state
    curandState state;

    // set index
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // init curand
    curand_init(seed + i, blockIdx.x, 0, &state);

    // set weight
    weights[i] = curand_normal(&state) * sqrtf(scalingFactor);
}

cudaError_t weight_init(float* d_targetMemorySpace, unsigned int size, float scaling_factor, int seed) {

    // add some padding to ensure that kernel runs efficiently but also does not override other memory cells
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // run kernel
    __cuda_weight_init<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(d_targetMemorySpace, size, scaling_factor, seed);

    // check for errors
    cudaError_t err = cudaGetLastError();
    
    return err;
}
