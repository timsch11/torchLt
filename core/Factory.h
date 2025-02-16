#ifndef FACTORY
#define FACTORY

#include "Tensor.h"
#include <utility>

/**
 * @brief Factory for creating a Tensor of specified shape to be initalized with values from the specified initalization_function
 * @param _shape Shape of the Tensor
 * @param _track_gradient Whether to track gradients for this tensor
 * @param seed Seed to be used if initalization is random
 * @param initalization_function Function to use for value initalization, params must match (float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed)
 */
Tensor* createTensorFromInitFunction(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed, 
    cudaError_t(*initalization_function)(float*, unsigned int, unsigned int, int));


/**
 * @brief Creates a Tensor object initialized with Xavier initialization.
 *
 * This function creates a Tensor object with the specified shape, gradient tracking option, 
 * and seed for random number generation. The Tensor is initialized using the Xavier initialization method.
 *
 * @param _shape A pair representing the shape of the Tensor (rows, columns).
 * @param _track_gradient A boolean indicating whether to track gradients for this Tensor.
 * @param seed An integer seed for random number generation.
 * @return A pointer to the created Tensor object.
 */
Tensor* createTensorWithXavierInit(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed);

/**
 * @brief Creates a Tensor object initialized with Kaiming He initialization.
 *
 * This function creates a Tensor object with the specified shape, gradient tracking option, 
 * and seed for random number generation. The Tensor is initialized using the Kaiming He initialization method.
 *
 * @param _shape A pair representing the shape of the Tensor (rows, columns).
 * @param _track_gradient A boolean indicating whether to track gradients for this Tensor.
 * @param seed An integer seed for random number generation.
 * @return A pointer to the created Tensor object.
 */
Tensor* createTensorWithKaimingHeInit(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed);

/**
 * @brief Creates a Tensor initalized with <constant>
 * @param _shape Shape of the Tensor
 * @param _track_gradient Whether to track gradients for this tensor
 * @param constant Constant that should be used to fill Tensor
 */
Tensor* createTensorWithConstants(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, float constant);

/**
 * @brief Factory for creating a Tensor as result of a unary operation
 * @param _d_value Pointer to device memory containing tensor values
 * @param _shape Shape of the tensor as (rows, columns) pair
 * @param _track_gradient Whether to track gradients for this tensor
 * @param _gradFunction Function pointer to gradient computation function
 * @param _d_funcArg1 Pointer to input tensor for the operation
 * @param _shapeFuncArg1 Shape of the input tensor
 */
Tensor* createTensorFromDevice(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, 
    void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1);

/**
 * @brief Factory for creating a Tensor as result of a binary operation
 * @param _d_value Pointer to device memory containing tensor values 
 * @param _shape Shape of the tensor as (rows, columns) pair
 * @param _track_gradient Whether to track gradients for this tensor
 * @param _gradFunction Function pointer to gradient computation function
 * @param _d_funcArg1 Pointer to first input tensor
 * @param _shapeFuncArg1 Shape of first input tensor
 * @param _d_funcArg2 Pointer to second input tensor
 * @param _shapeFuncArg2 Shape of second input tensor
 */
Tensor* createTensorFromDevice(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, 
    void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, 
    Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2);

/**
 * @brief Factory for creating a leaf Tensor
 * @param _d_value Pointer to device memory containing tensor values
 * @param _shape Shape of the tensor as (rows, columns) pair
 * @param _track_gradient Whether to track gradients for this tensor
 */
Tensor* createTensorFromDevice(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient);

/**
 * @brief Factory for creating a leaf Tensor from an array residing on CPU
 * @param _h_value Pointer to host memory containing tensor values
 * @param _shape Shape of the tensor as (rows, columns) pair
 * @param _track_gradient Whether to track gradients for this tensor
 */
Tensor* createTensorFromHost(float* _h_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient);

/**
 * @brief Initalizes the cuda device context and throws an error if no device is available
 */
void init();

/**
 * @brief Waits until weights updates completed.
 */
void sync();


#endif