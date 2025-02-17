#include "weightUpdate.cuh"


__global__ void __momentumUpdate(float* d_targetMemorySpace, float* d_pastGradients, float* d_gradient, float lr, float beta, float ibeta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = beta * d_pastGradients[i] + ibeta * d_gradient[i];
    d_pastGradients[i] = val;
    d_targetMemorySpace[i] -= lr*val;
}

cudaError_t momentumUpdate(float* d_targetMemorySpace, float* d_pastGradients, float* d_gradient, unsigned int size, float lr, float beta, bool async) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    __momentumUpdate<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_pastGradients, d_gradient, lr, beta, 1 - beta);

    // check for errors
    cudaError_t err = cudaGetLastError();

    if (!async) {
        // synchronize before continuing with host code
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    return err;
}