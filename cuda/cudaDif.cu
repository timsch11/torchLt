#include "../cudaNN/Tensor.h"
#include "cudaOperations.cu"


__global__ void relu_grad(float* vector, float* targetMemorySpace) {
    // todo
}

cudaError_t reluDif(float* vector, unsigned int size, float* targetMemorySpace) {
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    relu_grad<<<blocksThreads.first, blocksThreads.second>>>(vector, targetMemorySpace);
    return CHECK_CUDA_ERROR(cudaGetLastError());
}