#ifndef CUDA_MATH
#define CUDA_MATH


#include <cuda_runtime.h>

#include "cublas_v2.h"

#include <stdexcept>
#include <string>

#include "util.h"


/**
 * @brief kernel for adding entries of a tensor
 * @param d_targetMemorySpace pointer to memory section that should hold the result
 * @param d_tensor1 pointer to tensor1
 * @param d_tensor2 pointer to tensor2
 */
__global__ void __addTensorEntries(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2);

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
__global__ void __subtractVecEntries(float* d_targetMemorySpace, float* d_vec1, float* d_vec2);

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
__global__ void __scaleEntries(float* d_targetMemorySpace, float* d_tensor, float scalar);

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
__global__ void __hadamard(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2);

// Public interface functions

/**
 * @brief adds entries of a tensor
 * @param d_targetMemorySpace pointer to memory section that should hold the result
 * @param d_tensor1 pointer to tensor 1
 * @param tensorSize1 size of tensor 1
 * @param d_tensor2 pointer to tensor 2
 * @param tensorSize1 size of tensor 2
 * @return cudaSuccess_t or error
 */
cudaError_t tensoradd(float* d_targetMemorySpace, float* d_tensor1, unsigned int tensorSize1, 
                     float* d_tensor2, unsigned int tensorSize2);

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
 */
cudaError_t vecsub(float* d_targetMemorySpace, float* d_vector1, unsigned int vectorSize1,
                   float* d_vector2, unsigned int vectorSize2);

/**
 * @brief Scales each element of a tensor by a scalar value on the GPU
 * 
 * @param d_targetMemorySpace Pointer to GPU memory where the scaled tensor will be stored
 * @param d_tensor Pointer to the input tensor in GPU memory
 * @param tensorSize Number of elements in the tensor
 * @param scalar The value to multiply each tensor element by
 * 
 * @return cudaError_t Returns cudaSuccess if scaling operation completed successfully
 */
cudaError_t scaletensor(float* d_targetMemorySpace, float* d_tensor, unsigned int tensorSize, 
                       float scalar);

/**
 * @brief Performs element-wise Hadamard (element-wise) multiplication of two tensors on GPU
 * 
 * @param d_targetMemorySpace Pointer to device memory where result will be stored
 * @param d_tensor1 Pointer to first input tensor in device memory
 * @param d_tensor2 Pointer to second input tensor in device memory
 * @param shape Pair containing dimensions of the tensors (rows, columns). If columns=0, tensor is treated as 1D
 * @param async Whether to wait for the update to complete
 * 
 * @return cudaError_t Returns cudaSuccess if operation completed successfully
 */
cudaError_t hadamard(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2, std::pair<unsigned int, unsigned int> shape);

/** 
 * @brief Perfrorms d_targetMemorySpace = d_vector1 - (scalar * d_vector2)
 * @param d_targetMemorySpace Pointer to device memory where result will be stored
 * @param d_vector1 Pointer to first input tensor in device memory
 * @param d_vector2 Pointer to second input tensor in device memory
 * @param vectorSize1 Number of entries in Tensor1
 * @param vectorSize2 Number of entries in Tensor2
 * @param scalar Scalar
 * @param async Whether to wait for the update to complete
*/
cudaError_t scaledSubtraction(float* d_targetMemorySpace, float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2, float scalar, bool async);

#endif