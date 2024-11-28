#include "cudaMath.cu"


__global__ void reluGrad_kernel(float* targetMemorySpace, float* vector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector[i] > 0) {
        targetMemorySpace[i] = 1;
    } else {
        targetMemorySpace[i] = 0;
    }
}

cudaError_t reluGrad(float* targetMemorySpace, float* vector, unsigned int size) {
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    reluGrad_kernel<<<blocksThreads.first, blocksThreads.second>>>(targetMemorySpace, vector);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    return cudaSuccess;
}