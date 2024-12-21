#include "util.cu"


/**
 * @brief CUDA kernel for computing the gradient of ReLU activation function
 * 
 * This kernel calculates the derivative of the ReLU (Rectified Linear Unit) function
 * for each element in the input vector. The derivative is 1 for positive inputs
 * and 0 for negative inputs or zero.
 * 
 * @param targetMemorySpace Pointer to the output array where derivatives will be stored
 * @param vector Pointer to the input array containing values
 * 
 * @note Each thread processes one element of the input vector
 * @warning Assumes that the memory is already allocated and the array sizes match
 */
__global__ void __reluGrad(float* targetMemorySpace, float* vector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector[i] > 0) {
        targetMemorySpace[i] = 1;
    } else {
        targetMemorySpace[i] = 0;
    }
}

/**
 * @brief Computes the gradient of the ReLU activation function
 * 
 * This function launches a CUDA kernel to calculate the gradient of the ReLU
 * activation function. The gradient is 1 for positive inputs and 0 for
 * negative inputs.
 * 
 * @param targetMemorySpace Pointer to device memory where the gradients will be stored
 * @param vector Pointer to device memory containing the input values
 * @param size Number of elements in the input vector
 * 
 * @return cudaError_t Returns any CUDA errors that occurred during kernel execution
 */
cudaError_t reluGrad(float* targetMemorySpace, float* vector, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    __reluGrad<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(targetMemorySpace, vector);

    return cudaGetLastError();
}


/*int main() {
    float h_bias[3] = {1.0f, -2.0f, 3.0f};
    float *bias;
    cudaMalloc(&bias, 3 * sizeof(float));
    cudaMemcpy(bias, h_bias, 3 * sizeof(float), cudaMemcpyHostToDevice);

    reluGrad(bias, bias, 3);

    cudaMemcpy(h_bias, bias, 3*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<3; i++) {
        std::cout << h_bias[i] << " ";
    }
}*/