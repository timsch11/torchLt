#ifndef CUDA_NN
#define CUDA_NN


#include "util.h"
#include "cudaMem.cuh"
#include "cudaMath.cuh"
#include "cudaDif.cuh"

#include <cuda_runtime.h>


/**
 * @brief Applies ReLU (Rectified Linear Unit) activation function element-wise to a vector
 * 
 * This CUDA kernel implements the ReLU activation function which returns x if x > 0,
 * and 0 otherwise for each element in the input vector.
 * 
 * @param d_targetMemorySpace Pointer to device memory where the results will be stored
 * @param vector Pointer to device memory containing input vector elements
 * 
 * @note Each thread processes one element of the vector
 * @note The function assumes the input arrays have sufficient memory allocated
 * @note Caller must ensure proper grid and block dimensions are set
 */
__global__ void __relu(float* d_targetMemorySpace, float* vector);

/**
 * @brief Applies ReLU activation function element-wise to input vector on GPU
 * 
 * @param d_targetMemorySpace Pointer to device memory where result will be stored
 * @param d_vector Pointer to device memory containing input vector
 * @param size Number of elements in the vector
 * 
 * @return cudaError_t CUDA error code indicating success or failure
 * 
 * This function launches a CUDA kernel that applies the ReLU activation function
 * to each element of the input vector. The computation is distributed across
 * GPU blocks and threads calculated by computeBlockThreadAllocation().
 */
cudaError_t relu(float* d_targetMemorySpace, float* d_vector, unsigned int size);

/**
 * @brief Allocates memory and applies ReLU activation function to a vector on GPU
 * 
 * This function allocates memory on the device and applies the ReLU (Rectified Linear Unit)
 * activation function element-wise to the input vector.
 * 
 * @param d_vector Pointer to the input vector in device memory
 * @param size Number of elements in the vector
 * @return float* Pointer to the newly allocated result vector in device memory
 * 
 * @note The caller is responsible for freeing the returned memory
 */
float* reluAlloc(float* d_vector, unsigned int size);

/**
 * @brief Kernel that applies the sigmoid function element wise to a tensor
 * @param d_targetMemorySpace Pointer to memory section where result should go
 * @param d_tensor Input tensor
 */
__global__ void __sigmoid(float* d_targetMemorySpace, float* d_tensor);

/**
 * @brief Applies the sigmoid function to every element of d_tensor, stores result in d_targetMemorySpace
 * @param d_targetMemorySpace Pointer to memory section where result should go
 * @param d_tensor Input tensor
 * @param size Size of tensor
 * @return cudaSuccess_t or error
 */
cudaError_t sigmoid(float* d_targetMemorySpace, float* d_tensor, unsigned int size);

/**
 * @brief Applies the sigmoid function to every element of d_tensor, stores result in newly allocated memory section
 * @param d_tensor Input tensor
 * @param size Size of tensor
 * @return Pointer to newly allocated memory section that holds result
 */
float* sigmoidAlloc(float* d_tensor, unsigned int size);

/**
 * @brief Kernel that applies the tanh function element wise to a tensor
 * @param d_targetMemorySpace Pointer to memory section where result should go
 * @param d_tensor Input tensor
 */
__global__ void __tanh(float* d_targetMemorySpace, float* d_tensor);

/**
 * @brief Applies the tanh function to every element of d_tensor, stores result in d_targetMemorySpace
 * @param d_targetMemorySpace Pointer to memory section where result should go
 * @param d_tensor Input tensor
 * @param size Size of tensor
 * @return cudaSuccess_t or error
 */
cudaError_t tanh(float* d_targetMemorySpace, float* d_tensor, unsigned int size);

/**
 * @brief Applies the tanh function to every element of d_tensor, stores result in newly allocated memory section
 * @param d_tensor Input tensor
 * @param size Size of tensor
 * @return Pointer to newly allocated memory section that holds result
 */
float* tanhAlloc(float* d_tensor, unsigned int size);

/**
 * @brief Initializes weights using the Kaiming He initialization method.
 * 
 * This function implements the Kaiming He weight initialization, which is particularly 
 * useful for neural networks using ReLU activation functions. It helps maintain the 
 * variance of the weights throughout the network layers.
 * 
 * @param d_targetMemorySpace Pointer to the device memory where weights will be initialized
 * @param in_features Number of input features/nodes
 * @param out_features Number of output features/nodes
 * @param seed Random seed for weight initialization
 * 
 * @note The scaling factor is calculated as 2.0 / out_features according to He initialization
 */
cudaError_t kaiming_he(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed);

/**
 * @brief Initializes weights using Xavier initialization method
 * 
 * Xavier initialization helps in maintaining the variance of activations and gradients
 * across the network layers, which helps in better training convergence.
 *
 * @param d_targetMemorySpace Pointer to device memory where weights will be initialized
 * @param in_features Number of input features
 * @param out_features Number of output features
 * @param seed Random seed for weight initialization
 * 
 * @note The scaling factor is set to 1/out_features for this implementation
 */
cudaError_t xavier(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed);


// updates the given weight matrix (passed as pointer to float array), performs error checking 
void updateWeightMatrix(float* d_weightMatrixToUpdate, float* d_gradient, unsigned int in_features, unsigned int out_features, float learningRate);

// updates the given bias vector (passed as pointer to float array), performs error checking 
void updateBiasVector(float* d_biasVectorToUpdate, float* d_gradient, unsigned int out_features, float learningRate);


/**
 * @brief Performs element-wise multiplication (Hadamard product) of two tensors on GPU
 * 
 * @param d_tensor1 Pointer to the first tensor in device memory
 * @param shapeT1 Shape of the first tensor as (rows, columns)
 * @param d_tensor2 Pointer to the second tensor in device memory 
 * @param shapeT2 Shape of the second tensor as (rows, columns)
 * 
 * @return float* Pointer to the resulting tensor array in device memory
 * 
 * @throws std::runtime_error if tensor shapes are incompatible
 * 
 * @note Memory for the result is allocated on device and needs to be freed by caller
 * @note Input tensors must have identical shapes
 */
float* hadamardAlloc(float* d_tensor1, std::pair<unsigned int, unsigned int> shapeT1, float* d_tensor2, std::pair<unsigned int, unsigned int> shapeT2);

/**
 * @brief Adds two tensors (vectors) element-wise on GPU and allocates memory for the result
 * 
 * @param d_vector1 Pointer to the first vector in device memory
 * @param vectorSize1 Size of the first vector
 * @param d_vector2 Pointer to the second vector in device memory
 * @param vectorSize2 Size of the second vector
 * 
 * @return float* Pointer to the resulting vector in device memory
 * 
 * @throws std::runtime_error If vector sizes are not equal
 * 
 * @note Memory for the result is allocated on the device and needs to be freed by the caller
 * @note Both input vectors must be pre-allocated on the device
 */
float* tensoraddAlloc(float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2);

/**
 * @brief Subtracts two vectors element-wise and allocates memory for the result on device
 * 
 * This function performs element-wise subtraction of two vectors on CUDA device:
 * result = vector1 - vector2
 * 
 * @param d_vector1    Pointer to first input vector on device memory
 * @param vectorSize1  Size of first input vector
 * @param d_vector2    Pointer to second input vector on device memory
 * @param vectorSize2  Size of second input vector
 * 
 * @return            Pointer to result vector on device memory
 * 
 * @throws std::runtime_error If input vector sizes are not equal
 * 
 * @note Memory for result is allocated on device and should be freed by caller
 */
float* tensorsubAlloc(float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2);

/**
 * @brief Computes the matrix multiplication of A and B and stores result in newly allocated array
 * @param handle Pointer to the cuBLAS handle
 * @param transa 
 * @param transb
 * @param ax Number of rows of A (ShapeX)
 * @param ay Number of columns of A (ShapeY)
 * @param bx Number of rows of B (ShapeX)
 * @param by Number of columns of B (ShapeY)
 */
float* matmulAlloc(cublasHandle_t* handle, int ax, int ay, int bx, int by, const float *A, const float *B);


#endif