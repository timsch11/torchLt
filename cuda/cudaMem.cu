#include "cudaMath.cu"
#include "util.cu"


// MEMORY INITIALIZATION FUNCTIONS

__global__ void initZero(float* memorySection) {
    memorySection[blockIdx.x * blockDim.x + threadIdx.x] = 0;
}

float* zeros(int size) {
    // returns a pointer to (first element of) an array (interpretation of dimension is up to the caller) of specified size filled with zeros; array lives in unified memory (on cpu and gpu)

    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // reserve memory
    float* memoryAllocation;
    CHECK_CUDA_ERROR(cudaMalloc(&memoryAllocation, blockThreadAllocation.first * blockThreadAllocation.second * sizeof(float)));

    // launch kernel
    initZero<<<blockThreadAllocation.first, blockThreadAllocation.second>>>(memoryAllocation);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return memoryAllocation;
}

__global__ void copyValue(float* target, float* valueToCopy, int size) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < size) {
        target[ind] = valueToCopy[ind];
    }
}

float* copyValuesUnified(float* valueToCopy, int size) {
    // copies values of desire into unified memory, performs deepcopy
    float* output;
    cudaMallocManaged(&output, size * sizeof(float));

    int blockNum = size / 256;

    copyValue<<<blockNum + 1, BLOCK_SIZE>>>(output, valueToCopy, size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error while trying to copy values to cuda (unified): %s\n", cudaGetErrorString(error));
    }

    return output;
}

float* copyValues(float* valueToCopy, int size) {
    // copies values of desire into device memory, performs deepcopy
    float* output;
    cudaMalloc(&output, size * sizeof(float));

    int blockNum = size / 256;

    copyValue<<<blockNum + 1, BLOCK_SIZE>>>(output, valueToCopy, size);

    // do some error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error while trying to copy values to cuda: %s\n", cudaGetErrorString(error));
    }

    return output;
}

float* reserveMemoryOnDevice(unsigned int size) {
    // declare pointer
    float* memoryAlloc;

    // reserve actual space in memory, add some padding for thread efficiency
    CHECK_CUDA_ERROR(cudaMalloc(&memoryAlloc, size + (size / BLOCK_SIZE)));

    // return pointer 
    return memoryAlloc;
}