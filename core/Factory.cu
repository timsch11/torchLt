#include "Tensor.h"


/**
 * @brief Factory for creating a Tensor of specified shape to be initalized with values from the specified initalization_function
 * @param _shape Shape of the Tensor
 * @param _track_gradient Whether to track gradients for this tensor
 * @param seed Seed to be used if initalization is random
 * @param initalization_function Function to use for value initalization, params must match (float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed)
 */
Tensor* createTensorFromFunction(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed, void(*initalization_function)(float*, unsigned int, unsigned int, int)) {
    Tensor* obj;

    try {
        obj = &Tensor(_shape, _track_gradient, seed, initalization_function);
    } catch(std::runtime_error exc) {
        delete obj;
        throw std::runtime_error("Error when trying to create Tensor: " + std::string(exc.what()) + "\n");
    }

    return obj;
}

/**
 * @brief Factory for creating a Tensor as result of a unary operation
 * @param _d_value Pointer to device memory containing tensor values
 * @param _shape Shape of the tensor as (rows, columns) pair
 * @param _track_gradient Whether to track gradients for this tensor
 * @param _gradFunction Function pointer to gradient computation function
 * @param _d_funcArg1 Pointer to input tensor for the operation
 * @param _shapeFuncArg1 Shape of the input tensor
 */
Tensor* createTensorFromDeviceUnary(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1) {
    Tensor* obj;

    try {
        obj = &Tensor(_d_value, _shape, _track_gradient, _gradFunction, _d_funcArg1, _shapeFuncArg1);
    } catch(std::runtime_error exc) {
        delete obj;
        throw std::runtime_error("Error when trying to create Tensor: " + std::string(exc.what()) + "\n");
    }

    return obj;
}

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
Tensor* createTensorFromDeviceBinary(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2) {
    Tensor* obj;

    try {
        obj = &Tensor(_d_value, _shape, _track_gradient, _gradFunction, _d_funcArg1, _shapeFuncArg1, _d_funcArg2, _shapeFuncArg2);
    } catch(std::runtime_error exc) {
        delete obj;
        throw std::runtime_error("Error when trying to create Tensor: " + std::string(exc.what()) + "\n");
    }

    return obj;
}

/**
 * @brief Factory for creating a leaf Tensor
 * @param _d_value Pointer to device memory containing tensor values
 * @param _shape Shape of the tensor as (rows, columns) pair
 * @param _track_gradient Whether to track gradients for this tensor
 */
Tensor* createTensorFromDeviceLeaf(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient) {
    Tensor* obj;

    try {
        obj = &Tensor(_d_value, _shape, _track_gradient);
    } catch(std::runtime_error exc) {
        delete obj;
        throw std::runtime_error("Error when trying to create Tensor: " + std::string(exc.what()) + "\n");
    }

    return obj;
}

/**
 * @brief Factory for creating a leaf Tensor from an array residing on CPU
 * @param _d_value Pointer to device memory containing tensor values
 * @param _shape Shape of the tensor as (rows, columns) pair
 * @param _track_gradient Whether to track gradients for this tensor
 * @note Copies array to GPU
 */
Tensor* createTensorFromHostLeaf(float* _h_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient) {
    float* d_value;
    unsigned int size = _shape.first * _shape.second;

    cudaError_t allocationError = cudaMalloc(&d_value, size);
    if (allocationError != cudaSuccess) {
        std::string errorString = "cudaMalloc failed: " + (std::string) cudaGetErrorString(allocationError) + "\n";

        free(d_value);
        cudaFree(d_value);

        throw std::runtime_error(errorString);
    }

    cudaError_t copyError = cudaMemcpy(d_value, _h_value, size * sizeof(float), cudaMemcpyHostToDevice);
    if (allocationError != cudaSuccess) {
        std::string errorString = "cudaMalloc failed: " + std::string(cudaGetErrorString(allocationError)) + "\n";
        
        free(d_value);
        cudaFree(d_value);

        throw std::runtime_error(errorString);
    }

    Tensor* obj;

    try {
        obj = &Tensor(d_value, _shape, _track_gradient);
    } catch(std::runtime_error exc) {
        delete obj;
        throw std::runtime_error("Error when trying to create Tensor: " + std::string(exc.what()) + "\n");
    }

    return obj;
}