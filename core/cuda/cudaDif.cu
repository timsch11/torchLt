#include "cudaDif.cuh"


__global__ void __reluGrad(float* d_targetMemorySpace, float* d_vector, float* d_droot_dthis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_vector[i] > 0) {
        d_targetMemorySpace[i] = d_droot_dthis[i];
    } else {
        d_targetMemorySpace[i] = 0;
    }
}

__global__ void __reluGrad(float* d_targetMemorySpace, float* d_vector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_vector[i] > 0) {
        d_targetMemorySpace[i] = 1;
    } else {
        d_targetMemorySpace[i] = 0;
    }
}

cudaError_t reluGrad(float* d_targetMemorySpace, float* d_vector, float* d_droot_dthis, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    if (d_droot_dthis == nullptr) {
        // execute computation
        __reluGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_vector);
    } else {
        // execute computation with prior gradient
        __reluGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_vector, d_droot_dthis);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __sigmoidGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = d_tensor[i];
    d_targetMemorySpace[i] = val * (1 - val) * d_droot_dthis[i];
}

__global__ void __sigmoidGrad(float* d_targetMemorySpace, float* d_tensor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = d_tensor[i];
    d_targetMemorySpace[i] = val * (1 - val);
}

cudaError_t sigmoidGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    if (d_droot_dthis == nullptr) {
        // execute computation
        __sigmoidGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor);
    } else {
        // execute computation with prior gradient
        __sigmoidGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor, d_droot_dthis);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __tanhGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = d_tensor[i];
    d_targetMemorySpace[i] = (1 - (val*val)) * d_droot_dthis[i];
}

__global__ void __tanhGrad(float* d_targetMemorySpace, float* d_tensor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = d_tensor[i];
    d_targetMemorySpace[i] = 1 - (val*val);
}

cudaError_t tanhGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    if (d_droot_dthis == nullptr) {
        // execute computation
        __tanhGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor);
    } else {
        // execute computation with prior gradient
        __tanhGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor, d_droot_dthis);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __l2LossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual, float* d_droot_dthis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // result of L2 loss is scalar -> d_droot_dthis is just a single value
    d_targetMemorySpace[i] = 2 * *d_droot_dthis * (d_predicted[i] - d_actual[i]);
}

__global__ void __l2LossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[i] = 2 * (d_predicted[i] - d_actual[i]);
}

cudaError_t l2LossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual, float* d_droot_dthis, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    if (d_droot_dthis == nullptr) {
        // execute computation
        __l2LossGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_predicted, d_actual);
    } else {
        // execute computation with prior gradient
        __l2LossGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_predicted, d_actual, d_droot_dthis);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}