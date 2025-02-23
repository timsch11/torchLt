#ifndef CUDA_DIF
#define CUDA_DIF


#include <cuda_runtime.h>
#include "util.h"


/**
 * @brief CUDA kernel for computing the gradient of ReLU activation function
 * 
 * This kernel calculates the derivative of the ReLU (Rectified Linear Unit) function
 * for each element in the input vector. The derivative is 1 for positive inputs
 * and 0 for negative inputs or zero.
 * 
 * @param targetMemorySpace Pointer to the output array where derivatives will be stored
 * @param vector Pointer to the input array containing values
 * @param d_droot_dthis Gradient of the root of this computation wrt to the operation before this operation
 * 
 * @note Each thread processes one element of the input vector
 * @warning Assumes that the memory is already allocated and the array sizes match
 */
__global__ void __reluGrad(float* d_targetMemorySpace, float* d_vector, float* d_droot_dthis);

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
__global__ void __reluGrad(float* d_targetMemorySpace, float* d_vector);

/**
 * @brief Computes the gradient of the ReLU activation function
 * 
 * This function launches a CUDA kernel to calculate the gradient of the ReLU
 * activation function. The gradient is 1 for positive inputs and 0 for
 * negative inputs.
 * 
 * @param targetMemorySpace Pointer to device memory where the gradients will be stored
 * @param vector Pointer to device memory containing the input values
 * @param d_droot_dthis Gradient of the root of this computation wrt to the operation before this operation, neglected if nullptr
 * @param size Number of elements in the input vector
 * 
 * @return cudaError_t Returns any CUDA errors that occurred during kernel execution
 */
cudaError_t reluGrad(float* d_targetMemorySpace, float* d_vector, float* d_droot_dthis, unsigned int size);

/**
 * @brief CUDA kernel function to compute the gradient of the sigmoid function.
 *
 * This kernel computes the gradient of the sigmoid function for each element in the input tensor.
 * The result is stored in the target memory space.
 *
 * @param d_targetMemorySpace Pointer to the device memory where the result will be stored.
 * @param d_tensor Pointer to the device memory containing the input tensor.
 * @param d_droot_dthis Gradient of the root of this computation wrt to the operation before this operation, neglected if nullptr
 */
__global__ void __sigmoidGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis);

/**
 * @brief CUDA kernel function to compute the gradient of the sigmoid function.
 *
 * This kernel computes the gradient of the sigmoid function for each element in the input tensor.
 * The result is stored in the target memory space.
 *
 * @param d_targetMemorySpace Pointer to the device memory where the result will be stored.
 * @param d_tensor Pointer to the device memory containing the input tensor.
 */
__global__ void __sigmoidGrad(float* d_targetMemorySpace, float* d_tensor);

/**
 * @brief Launches the CUDA kernel to compute the gradient of the sigmoid function.
 *
 * This function computes the optimal block and thread distribution, and then launches the
 * __sigmoidGrad kernel to compute the gradient of the sigmoid function for each element in the input tensor.
 *
 * @param d_targetMemorySpace Pointer to the device memory where the result will be stored.
 * @param d_tensor Pointer to the device memory containing the input tensor.
 * @param d_droot_dthis Gradient of the root of this computation wrt to the operation before this operation, neglected if nullptr
 * @param size The number of elements in the input tensor.
 * @return cudaError_t Returns the error that occurred during the kernel launch or cudaSuccess_t
 */
cudaError_t sigmoidGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis, unsigned int size);

/**
 * @brief CUDA kernel function to compute the gradient of the tanh function.
 *
 * This kernel computes the gradient of the tanh function for each element in the input tensor.
 * The result is stored in the target memory space.
 *
 * @param d_targetMemorySpace Pointer to the device memory where the result will be stored.
 * @param d_droot_dthis Gradient of the root of this computation wrt to the operation before this operation, neglected if nullptr
 * @param d_tensor Pointer to the device memory containing the input tensor.
 */
__global__ void __tanhGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis);

/**
 * @brief CUDA kernel function to compute the gradient of the tanh function.
 *
 * This kernel computes the gradient of the tanh function for each element in the input tensor.
 * The result is stored in the target memory space.
 *
 * @param d_targetMemorySpace Pointer to the device memory where the result will be stored.
 * @param d_tensor Pointer to the device memory containing the input tensor.
 */
__global__ void __tanhGrad(float* d_targetMemorySpace, float* d_tensor);

/**
 * @brief Launches the CUDA kernel to compute the gradient of the tanh function.
 *
 * This function computes the optimal block and thread distribution, and then launches the
 * __tanhGrad kernel to compute the gradient of the tanh function for each element in the input tensor.
 *
 * @param d_targetMemorySpace Pointer to the device memory where the result will be stored.
 * @param d_tensor Pointer to the device memory containing the input tensor.
 * @param d_droot_dthis Gradient of the root of this computation wrt to the operation before this operation, neglected if nullptr
 * @param size The number of elements in the input tensor.
 * @return cudaError_t Returns the error that occurred during the kernel launch or cudaSuccess_t
 */
cudaError_t tanhGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis, unsigned int size);

/**
 * @brief CUDA kernel function to compute the gradient of the L2 loss function.
 *
 * This kernel computes the gradient of the L2 loss function wrt to d_predicted. 
 * (It is useless to compute gradient wrt the actual values)
 * The result is stored in the target memory space.
 *
 * @param d_targetMemorySpace Pointer to the device memory where the result will be stored.
 * @param d_predicted Pointer to the device memory containing the prediction.
 * @param d_actual Pointer to the device memory containing the actual labels.
 * @param d_droot_dthis Gradient of the root of this computation wrt to the operation before this operation, neglected if nullptr
 */
__global__ void __l2LossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual, float* d_droot_dthis);

/**
 * @brief CUDA kernel function to compute the gradient of the L2 loss function.
 *
 * This kernel computes the gradient of the L2 loss function wrt to d_predicted. 
 * (It is useless to compute gradient wrt the actual values)
 * The result is stored in the target memory space.
 *
 * @param d_targetMemorySpace Pointer to the device memory where the result will be stored.
 * @param d_predicted Pointer to the device memory containing the prediction.
 * @param d_actual Pointer to the device memory containing the actual labels.
 */
__global__ void __l2LossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual);

/**
 * @brief Launches the CUDA kernel to compute the gradient of the L2 loss function wrt to d_predicted. 
 * (It is useless to compute gradient wrt the labels)
 *
 * This function computes the optimal block and thread distribution, and then launches the
 * __l2LossGrad kernel to compute the gradient of the l2 loss function.
 *
 * @param d_targetMemorySpace Pointer to the device memory where the result will be stored.
 * @param d_predicted Pointer to the device memory containing the prediction.
 * @param d_actual Pointer to the device memory containing the actual labels.
 * @param d_droot_dthis Gradient of the root of this computation wrt to the operation before this operation, neglected if nullptr
 * @param size The number of elements in the input tensors (no shape check!).
 * @return cudaError_t Returns the error that occurred during the kernel launch or cudaSuccess_t
 */
cudaError_t l2LossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual, float* d_droot_dthis, unsigned int size);

__global__ void __crossEntropyLossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual, float* d_droot_dthis);

__global__ void __crossEntropyLossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual);

cudaError_t crossEntropyLossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual, float* d_droot_dthis, unsigned int size);

// Computes gradient of softmax function
cudaError_t softmaxGrad(float* d_targetMemorySpace, float* d_softmax_output, float* d_droot_dthis, unsigned int size);

#endif