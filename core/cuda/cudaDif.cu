#include "cudaDif.cuh"


__global__ void __reluGrad(float* d_targetMemorySpace, float* d_vector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_vector[i] > 0) {
        d_targetMemorySpace[i] = 1;
    } else {
        d_targetMemorySpace[i] = 0;
    }
}

cudaError_t reluGrad(float* d_targetMemorySpace, float* d_vector, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    // execute computation
    __reluGrad<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_vector);

    return cudaGetLastError();
}

__global__ void __sigmoidGrad(float* d_targetMemorySpace, float* d_tensor) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    float val = d_tensor[ind];
    d_targetMemorySpace[ind] = val * (1 - val);
}

cudaError_t sigmoidGrad(float* d_targetMemorySpace, float* d_tensor, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    // execute computation
    __sigmoidGrad<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_tensor);

    return cudaGetLastError();
}

__global__ void __tanhGrad(float* d_targetMemorySpace, float* d_tensor) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    float val = d_tensor[ind];
    d_targetMemorySpace[ind] = 1 - (val*val);
}

cudaError_t tanhGrad(float* d_targetMemorySpace, float* d_tensor, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    // execute computation
    __tanhGrad<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_tensor);

    return cudaGetLastError();
}