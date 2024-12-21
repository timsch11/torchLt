#include <iostream>
#include <stdexcept>
#include <string>
#include "util.cu"


// MATH FUNCTIONS

/**
 * @brief kernel for adding entries of a tensor
 * @param d_targetMemorySpace pointer to memory section that should hold the result
 * @param d_tensor1 pointer to tensor1
 * @param d_tensor2 pointer to tensor2
 */
__global__ void __addTensorEntries(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[idx] = d_tensor1[idx] + d_tensor2[idx];
}

/**
 * @brief adds entries of a tensor
 * @param d_targetMemorySpace pointer to memory section that should hold the result
 * @param d_tensor1 pointer to tensor 1
 * @param tensorSize1 size of tensor 1
 * @param d_tensor2 pointer to tensor 2
 * @param tensorSize1 size of tensor 2
 * @return cudaSuccess_t or error
 */
// adds the values of two vectors and stores the result in <targetMemorySpace>, Note: for efficiency reasons size of targetMemorySpace must be rounded up to a multiple of blocksize
cudaError_t tensoradd(float* d_targetMemorySpace, float* d_tensor1, unsigned int tensorSize1, float* d_tensor2, unsigned int tensorSize2) {

    // check for vector compatibility
    if (tensorSize1 != tensorSize2) {
        printf("tensors to be added have different shapes\n");
        return cudaErrorInvalidValue;
    }

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(tensorSize1);
    
    __addTensorEntries<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_tensor1, d_tensor2);
    return cudaGetLastError();
}

/**
 * @brief Subtracts corresponding elements of two vectors and stores the result in the target memory space.
 *
 * This operation is performed on the GPU using CUDA. Each thread is responsible for computing the difference
 * of the elements at the index it is assigned to.
 *
 * @param d_targetMemorySpace Pointer to the memory space where the result of the subtraction will be stored.
 * @param d_vec1 Pointer to the first input vector.
 * @param d_vec2 Pointer to the second input vector.
 * @param ind Index of the element to be processed by the current thread.
*/
__global__ void __subtractVecEntries(float* d_targetMemorySpace, float* d_vec1, float* d_vec2) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[ind] = d_vec1[ind] - d_vec2[ind];
}


/**
 * @brief Subtracts two vectors element-wise on the GPU: d_targetMemorySpace = d_vector1 - d_vector2
 * 
 * @param d_targetMemorySpace Pointer to device memory where the result will be stored
 * @param d_vector1 Pointer to first device vector to be subtracted from
 * @param vectorSize1 Size of the first vector
 * @param d_vector2 Pointer to second device vector to subtract
 * @param vectorSize2 Size of the second vector
 * 
 * @return cudaError_t Returns cudaSuccess if successful, cudaErrorInvalidValue if vectors have different sizes
 * 
 * @note This function verifies vector compatibility before performing the operation
 * @note Function synchronizes the device and checks for CUDA errors after kernel execution
 */
cudaError_t vecsub(float* d_targetMemorySpace, float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {

    // check for vector compatibility
    if (vectorSize1 != vectorSize2) {
        printf("vectors to be subtracted have different shapes\n");
        return cudaErrorInvalidValue;
    }

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(vectorSize1);
    
    __subtractVecEntries<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_vector1, d_vector2);

    return cudaGetLastError();
}

/**
 * @brief Scales all entries of a tensor by a scalar value in parallel on the GPU
 * 
 * This CUDA kernel multiplies each element of the input tensor by a scalar value
 * and stores the result in the target memory space.
 * 
 * @param[out] d_targetMemorySpace Pointer to the destination array in device memory
 * @param[in] d_tensor Pointer to the source tensor in device memory
 * @param[in] scalar The scaling factor to multiply each element by
 * 
 * @note The kernel assumes that the input arrays are properly allocated and have
 *       sufficient size for the number of threads being launched
 */
__global__ void __scaleEntries(float* d_targetMemorySpace, float* d_tensor, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[idx] = d_tensor[idx] * scalar;
}

/**
 * @brief Scales each element of a tensor by a scalar value on the GPU
 * 
 * @param d_targetMemorySpace Pointer to GPU memory where the scaled tensor will be stored
 * @param d_tensor Pointer to the input tensor in GPU memory
 * @param tensorSize Number of elements in the tensor
 * @param scalar The value to multiply each tensor element by
 * 
 * @return cudaError_t Returns cudaSuccess if scaling operation completed successfully otherwise returns error
 * 
 * @note This function launches a CUDA kernel to perform element-wise multiplication
 *       The block and thread allocation is automatically computed based on tensor size
 */
cudaError_t scaletensor(float* d_targetMemorySpace, float* d_tensor, unsigned int tensorSize, float scalar) {

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(tensorSize);
    
    __scaleEntries<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_tensor, scalar);
    return cudaGetLastError();
}

/**
 * @brief Performs element-wise multiplication (Hadamard product) of two tensors.
 * 
 * This CUDA kernel function multiplies corresponding elements from two input tensors
 * and stores the result in a target memory space.
 *
 * @param d_targetMemorySpace Pointer to device memory where the result will be stored
 * @param d_tensor1 Pointer to the first input tensor in device memory
 * @param d_tensor2 Pointer to the second input tensor in device memory
 * 
 * @note The kernel assumes that all input arrays have the same dimensions and
 *       sufficient memory has been allocated for the operation.
 * @note The function doesn't perform bounds checking - proper grid and block dimensions
 *       must be set by the caller.
 */
__global__ void __hadamard(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[ind] = d_tensor1[ind] * d_tensor2[ind];
}

/**
 * @brief Performs element-wise Hadamard (element-wise) multiplication of two tensors on GPU
 * 
 * @param d_targetMemorySpace Pointer to device memory where result will be stored
 * @param d_tensor1 Pointer to first input tensor in device memory
 * @param d_tensor2 Pointer to second input tensor in device memory
 * @param shape Pair containing dimensions of the tensors (rows, columns). If columns=0, tensor is treated as 1D
 * 
 * @return cudaError_t Returns cudaSuccess if operation completed successfully otherwise returns error
 * 
 * @note Input tensors must have the same dimensions
 * @note All pointers must point to pre-allocated device memory
 */
cudaError_t hadamard(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2, std::pair<unsigned int, unsigned int> shape) {

    // calculate num of fields
    unsigned int size = (shape.second == 0) ? shape.first : shape.first * shape.second;

    // calculate #Block and #Thread
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    
    // let kernel do its work
    __hadamard<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_tensor1, d_tensor2);

    // return error
    return cudaGetLastError();
}


/*int main() {
    // Allocate host memory for results
float* h_bias = new float[256];  // Host memory

// Device allocation
float* d_bias = constants(25, 1);
// cudaMalloc(&d_bias, 5 * sizeof(float));

// Initialize with zeros


float* d_bias2 = constants(25, -2);

// Initialize with zeros
// weight_init(d_bias, 5, 1, 10.75f, 1234567);

float* d_target = zeros(25);

cudaEvent_t e1, e2;
cudaEventCreate(&e1);
cudaEventCreate(&e2);

cudaStream_t s1;
cudaStreamCreate(&s1);


cudaEventRecord(e1, s1);
matmatT_matmul<<<5, 5, 5*sizeof(float), s1>>>(d_bias, d_bias2, 5, d_target);
cudaEventRecord(e2, s1);
CHECK_CUDA_ERROR(cudaGetLastError());
cudaDeviceSynchronize();

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, e1, e2);

std::cout << milliseconds << "ms passed";

// Copy from device (zero_init) to device (d_bias)
// cudaMemcpy(d_bias, zero_init, 5 * sizeof(float), cudaMemcpyDeviceToDevice);

// Copy from device to host for printing
cudaMemcpy(h_bias, d_target, 256 * sizeof(float), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

// Check for errors
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA error");
}

// Print values from host memory
for (int i = 0; i < 256; i++) {
    std::cout << h_bias[i] << " ";
}

// Cleanup
delete[] h_bias;
cudaFree(d_bias);
cudaFree(d_bias2);
}*/