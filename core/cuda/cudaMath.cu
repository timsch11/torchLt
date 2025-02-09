#include "cudaMath.cuh"


// MATH FUNCTIONS


__global__ void __addTensorEntries(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[idx] = d_tensor1[idx] + d_tensor2[idx];
}

cudaError_t tensoradd(float* d_targetMemorySpace, float* d_tensor1, unsigned int tensorSize1, float* d_tensor2, unsigned int tensorSize2) {

    // check for vector compatibility
    if (tensorSize1 != tensorSize2) {
        printf("tensors to be added have different shapes\n");
        return cudaErrorInvalidValue;
    }

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(tensorSize1);
    
    __addTensorEntries<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor1, d_tensor2);

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __subtractVecEntries(float* d_targetMemorySpace, float* d_vec1, float* d_vec2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[i] = d_vec1[i] - d_vec2[i];
}

cudaError_t vecsub(float* d_targetMemorySpace, float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {

    // check for vector compatibility
    if (vectorSize1 != vectorSize2) {
        printf("vectors to be subtracted have different shapes\n");
        return cudaErrorInvalidValue;
    }

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(vectorSize1);
    
    __subtractVecEntries<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_vector1, d_vector2);

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __scaleEntries(float* d_targetMemorySpace, float* d_tensor, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[idx] = d_tensor[idx] * scalar;
}

cudaError_t scaletensor(float* d_targetMemorySpace, float* d_tensor, unsigned int tensorSize, float scalar) {

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(tensorSize);
    
    __scaleEntries<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor, scalar);

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __hadamard(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[i] = d_tensor1[i] * d_tensor2[i];
}

cudaError_t hadamard(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2, std::pair<unsigned int, unsigned int> shape) {

    // calculate num of fields
    unsigned int size = (shape.second == 0) ? shape.first : shape.first * shape.second;

    // calculate #Block and #Thread
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    
    // let kernel do its work
    __hadamard<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor1, d_tensor2);

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // return error
    return err;
}
