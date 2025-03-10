#ifndef TENSOR
#define TENSOR


#include <cuda_runtime.h>

#include "cublas_v2.h"
#include <cublasLt.h>

#include <string>
#include <utility>

#include <iostream>
#include <stdexcept>

#include "cuda/cudaNN.cuh"


class Tensor {

    /**
         * @brief Core data members for the Tensor class representing a mathematical tensor with GPU support
         * 
         * @member d_value Pointer to device memory storing the tensor values
         * @member d_gradient Pointer to device memory storing the gradient values
         * @member shape Pair representing tensor dimensions (rows, columns)
         * @member track_gradient Flag indicating whether gradients should be computed for this tensor
         * @member leaf Flag indicating if tensor is a leaf node in its computational graph
         * @member gradientSet Flag indicating if gradient has been computed and set
         * @member d_funcArg1 Pointer to first tensor argument used in gradient computation (nullptr if unused)
         * @member d_funcArg2 Pointer to second tensor argument used in gradient computation (nullptr if unused)
         * @member graphStream CUDA stream shared across computational graph for synchronized operations
         * @member nodeStream Individual CUDA stream for asynchronous operations specific to this tensor, i.e. weight updates
         * @member lowerGraphSize Number of nodes in the computational graph below this tensor (including self)
         * @member shapeFuncArg1 Dimensions of first tensor argument
         * @member shapeFuncArg2 Dimensions of second tensor argument
         * @member gradFunction Function pointer to gradient computation routine
         * 
         * @note This implementation supports GPU-accelerated tensor operations using CUDA
         * @note The tensor supports automatic differentiation through computational graph tracking
         * @note Attributes that start with d_ reside in GPU memory
         */

    private:

        float* d_value;
        float* d_gradient;

        // tensor shape

        std::pair<unsigned int, unsigned int> shape;

        bool track_gradient;  // whether gradient should be tracked for this tensor
        bool leaf;            // displays whether this tensor is a leaf on its computational graph (=if true there will be no gradient function or args)
        bool gradientSet;     // displays if gradient has been set

        // arguments for gradient function
        // if gradient function only needs one argument: d_funcArg2 = nullptr

        Tensor* d_funcArg1;  
        Tensor* d_funcArg2;
        Tensor* d_funcArg3; // if set: 1. weight 2. input 3. bias

        // garbage collector 
        int* refCount;

        // shapes of function arguments

        std::pair<unsigned int, unsigned int> shapeFuncArg1;
        std::pair<unsigned int, unsigned int> shapeFuncArg2;
        std::pair<unsigned int, unsigned int> shapeFuncArg3;

        // gradient function of operation this node resulted from

        void (*gradFunction)(Tensor*);

    public:
    
        /**
         * @brief Constructor for creating a Tensor of specified shape initalized with values from the specified initalization_function
         * @param _shape Shape of the Tensor
         * @param _track_gradient Whether to track gradients for this tensor
         * @param seed Seed to be used if initalization is random
         * @param initalization_function Function to use for value initalization, params must match (float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed)
         */
        Tensor(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed, cudaError_t(*initalization_function)(float*, unsigned int, unsigned int, int));
        
        /**
         * @brief Constructor for creating a Tensor as result of a unary operation
         * @param _d_value Pointer to device memory containing tensor values
         * @param _shape Shape of the tensor as (rows, columns) pair
         * @param _track_gradient Whether to track gradients for this tensor
         * @param _gradFunction Function pointer to gradient computation function
         * @param _d_funcArg1 Pointer to input tensor for the operation
         * @param _shapeFuncArg1 Shape of the input tensor
         */
        Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1);
        
        /**
         * @brief Constructor for creating a Tensor as result of a binary operation
         * @param _d_value Pointer to device memory containing tensor values 
         * @param _shape Shape of the tensor as (rows, columns) pair
         * @param _track_gradient Whether to track gradients for this tensor
         * @param _gradFunction Function pointer to gradient computation function
         * @param _d_funcArg1 Pointer to first input tensor
         * @param _shapeFuncArg1 Shape of first input tensor
         * @param _d_funcArg2 Pointer to second input tensor
         * @param _shapeFuncArg2 Shape of second input tensor
         */
        Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2);
        
        /**
         * @brief Constructor for creating a Tensor as result of a forward pass
         * @param _d_value Pointer to device memory containing tensor values 
         * @param _shape Shape of the tensor as (rows, columns) pair
         * @param _track_gradient Whether to track gradients for this tensor
         * @param _gradFunction Function pointer to gradient computation function
         * @param _d_funcArg1 Pointer to first input tensor (=weight)
         * @param _shapeFuncArg1 Shape of first input tensor
         * @param _d_funcArg2 Pointer to second input tensor (=input)
         * @param _shapeFuncArg2 Shape of second input tensor
         * @param _d_funcArg3 Pointer to third input tensor (=bias)
         * @param _shapeFuncArg3 Shape of third input tensor
         */
        Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2, Tensor* _d_funcArg3, std::pair<unsigned int, unsigned int> _shapeFuncArg3);

        /**
         * @brief Constructor for creating a Tensor of specified shape initalized with constant value
         * @param _shape Shape of the Tensor
         * @param _track_gradient Whether to track gradients for this tensor
         * @param constant Constant that should be used to fill Tensor
         */
        Tensor(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, float constant);
        
        /**
         * @brief Constructor for creating a leaf Tensor
         * @param _d_value Pointer to device memory containing tensor values
         * @param _shape Shape of the tensor as (rows, columns) pair
         * @param _track_gradient Whether to track gradients for this tensor
         */
        Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient);

        /**
         * @brief Destructor that frees device memory and handles cuBLAS cleanup
         */
        ~Tensor();

        // garbage collector utils

        /**
        * @brief Reference counting management for tensor objects
        * 
        * addReference(): Increments the reference counter for this tensor
        * indicating another object is using this tensor.
        * 
        * removeReference(): Decrements the reference counter and deletes
        * the tensor object if no more references exist (counter reaches 0).
        * This implements automatic memory management through reference counting.
        * 
        * @warning Improper use of these methods may lead to memory leaks or
        * premature object deletion. Always ensure references are properly
        * tracked.
        */
        void addReference();

        /**
        * @brief Decrements the reference count and deletes the tensor if no references remain
        * 
        * This method is part of the reference counting mechanism for memory management.
        * It decrements the reference counter and if it reaches zero or below,
        * the tensor object is deleted from memory.
        * 
        * @warning This method may delete the current object, so no member access
        *          should be performed after calling this method
        */
        void removeReference();

        /**
         * @brief Return the reference count of this Tensor
         */
        int getReferenceCount();

        // GETTER

        /**
         * @brief Gets the pointer to the tensor's value data on GPU memory
         * @return float* Pointer to the tensor's value data stored on device (GPU) memory
         */
        float* getValue() const;

        /**
         * @brief Gets the pointer to the tensor's value data on CPU memory
         * @return float* Pointer to the tensor's value data stored on host (CPU) memory
         * @note take care of the deletion since the return value is a pointer
         */
        float* getValueCPU() const;

        /**
         * @brief Gets the pointer to the tensor's gradient data on GPU memory
         * @return float* Pointer to the tensor's gradient data stored on device (GPU) memory
         */
        float* getGradient() const;

        /**
         * @brief Gets the pointer to the tensor's gradient data on CPU memory
         * @return float* Pointer to the tensor's value data stored on host (CPU) memory
         * @note take care of the deletion since the return value is a pointer
         */
        float* getGradientCPU() const;

        /**
         * @brief Get the number of rows in the tensor
         * @return unsigned int - number of rows
         */
        unsigned int getShapeX() const;

        /**
         * @brief Get the number of columns in the tensor
         * @return unsigned int - number of columns
         */
        unsigned int getShapeY() const;

        /**
         * @brief Get the shape of the tensor as a pair of dimensions
         * @return std::pair<unsigned int, unsigned int> - (rows, columns)
         */
        std::pair<unsigned int, unsigned int> getShape() const;

        /**
         * @brief Get total number of elements in the tensor
         * @return unsigned int - product of rows and columns
         */
        unsigned int getSize() const;

        /**
         * @brief Retrieves rows from the tensor (0 indexed).
         *
         * This function returns a pointer to a new Tensor that represents the specified row of the current tensor.
         *
         * @param fromRow The starting row index of the sub-tensor (inclusive).
         * @param toRow The ending row index of the sub-tensor (exclusive).
         * @return Tensor* Pointer to a Tensor representing the specified rows.
         */
        Tensor* getRows(unsigned int fromRow, unsigned int toRow);

        /**
         * @brief Retrieves columns from the tensor (0 indexed).
         *
         * This function returns a pointer to a new Tensor that represents the specified column of the current tensor.
         *
         * @param row The row index of the element to retrieve.
         * @param col The column index of the element to retrieve.
         * @return Tensor* Pointer to a Tensor representing the specified column.
         */
        Tensor* getCols(unsigned int fromCol, unsigned int toCol);

        /**
         * @brief Retrieves a specific element from the tensor (0 indexed).
         *
         * This function returns a pointer to a new Tensor that holds the value at the specified row and column in the current tensor.
         *
         * @param row The row index of the element to retrieve.
         * @param col The column index of the element to retrieve.
         * @return Tensor* Pointer to a Tensor representing the element at the specified location.
         */
        Tensor* getVal(unsigned int row, unsigned int col);

        /**
         * @brief Retrieves a sub-tensor defined by the specified range (0 indexed).
         *
         * This function returns a pointer to a new Tensor that represents a sub-region of the current tensor,
         * defined by the starting and ending indices for both rows and columns.
         *
         * @param fromRow The starting row index of the sub-tensor (inclusive).
         * @param toRow The ending row index of the sub-tensor (exclusive).
         * @param fromCol The starting column index of the sub-tensor (inclusive).
         * @param toCol The ending column index of the sub-tensor (exclusive).
         * @return Tensor* Pointer to a Tensor representing the specified sub-region.
         */
        Tensor* get(unsigned int fromRow, unsigned int toRow, unsigned int fromCol, unsigned int toCol);

        /**
         * @brief Returns a deepcopy of this Tensor without gradient data
         * @return Deepcopy of this Tensor
         */
        Tensor* deepcopy();

        /**
         * @brief Get pointer to first argument tensor used in operation
         * @return Tensor* - pointer to first argument tensor
         */
        Tensor* getArg1() const;

        /**
         * @brief Get pointer to second argument tensor used in operation
         * @return Tensor* - pointer to second argument tensor
         */
        Tensor* getArg2() const;

        /**
         * @brief Get pointer to third argument tensor used in operation
         * @return Tensor* - pointer to third argument tensor
         */
        Tensor* getArg3() const;

        /**
         * @brief Get shape of first argument tensor used in operation
         * @return std::pair<unsigned int, unsigned int> - (rows, columns) of first argument
         */
        std::pair<unsigned int, unsigned int> getShapeArg1() const;

        /**
         * @brief Get shape of second argument tensor used in operation
         * @return std::pair<unsigned int, unsigned int> - (rows, columns) of second argument
         */
        std::pair<unsigned int, unsigned int> getShapeArg2() const;

        /**
         * @brief Get shape of third argument tensor used in operation
         * @return std::pair<unsigned int, unsigned int> - (rows, columns) of third argument
         */
        std::pair<unsigned int, unsigned int> getShapeArg3() const;

        /**
         * @brief Check if gradients should be tracked for this tensor
         * @return true if gradients are being tracked, false otherwise
         */
        bool getTrackGradient() const;

        /**
         * @brief Check if this tensor is a leaf node (has no dependencies)
         * @return true if tensor is a leaf node, false if it depends on other tensors
         */
        bool isLeaf() const;

        /**
         * @brief Check if gradient has been computed and set for this tensor
         * @return true if gradient is set, false otherwise
         */
        bool isGradientSet() const;

        // SETTER

        /**
         * @brief Enables gradient tracking
         */
        void enableGradientTracking();

        /**
         * @brief Disables gradient tracking
         */
        void disableGradientTracking();

        /**
         * @brief Changes the gradient status of the tensor (if gradient is tracked)
         * @param _gradientSet Boolean indicating whether gradient is set
         */
        void changeGradientSet(bool _gradientSet);

        /**
         * @brief Performs backward pass gradient computation through the computational graph
         * @note Calls the gradient function stored in the tensor to compute gradients if gradient tracking is enabled
         */
        void backward();

        /**
         * @brief Do not use with sigmoid or tanh: Performs asynchronous-optimized backpropagation combined with an SGD step
         * @note Requires sync() to be called afterwards
         */
        void asyncbackpropsgd(float lr);

        // update value without gradient tracking (i.e. for updating the weights of a neural network)
        // void addNoGrad() # TODO

        // SHAPE COMPARISON

        /**
         * @brief Checks if two tensors have the same shape
         * @param other The tensor to compare shape with
         * @return true if shapes are identical, false otherwise
         */
        bool sameShape(Tensor other) const;

        /**
         * @brief Checks if this tensor can be matrix multiplied with another tensor
         * @param other The tensor to check matrix multiplication compatibility with
         * @return true if tensors can be matrix multiplied, false otherwise
         */
        bool matMulCompatible(Tensor other) const;

        /**
         * @brief Error handling functions for various error types in Tensor operations
         * 
         * @param err Error object to check (float* pointer)
         * @param errorText Error message to display if error occurs
         * 
         * These overloaded functions handle different types of errors:
         * - CUDA runtime errors (cudaError_t)
         * - cuBLAS errors (cublasStatus_t) 
         * - Memory allocation errors (null float pointers)
         * - Generic error messages (string only)
         *
         * If an error is detected:
         * 1. Prints error message to console
         * 2. Frees Tensor resources
         * 3. Exits program
         *
         * @note These are const member functions that don't modify the Tensor state
         */
        void handleError(cudaError_t err, std::string errorText) const;

        /**
         * @brief Error handling functions for various error types in Tensor operations
         * 
         * @param err Error object to check (cudaError_t)
         * @param errorText Error message to display if error occurs
         * 
         * These overloaded functions handle different types of errors:
         * - CUDA runtime errors (cudaError_t)
         * - cuBLAS errors (cublasStatus_t) 
         * - Memory allocation errors (null float pointers)
         * - Generic error messages (string only)
         *
         * If an error is detected:
         * 1. Prints error message to console
         * 2. Frees Tensor resources
         * 3. Exits program
         *
         * @note These are const member functions that don't modify the Tensor state
         */
        void handleError(cublasStatus_t err, std::string errorText) const;

        /**
         * @brief Error handling functions for various error types in Tensor operations
         * 
         * @param err Error object to check (cublasStatus_t)
         * @param errorText Error message to display if error occurs
         * 
         * These overloaded functions handle different types of errors:
         * - CUDA runtime errors (cudaError_t)
         * - cuBLAS errors (cublasStatus_t) 
         * - Memory allocation errors (null float pointers)
         * - Generic error messages (string only)
         *
         * If an error is detected:
         * 1. Prints error message to console
         * 2. Frees Tensor resources
         * 3. Exits program
         *
         * @note These are const member functions that don't modify the Tensor state
         */
        void handleError(float* err, std::string errorText) const;

        /**
         * @brief Error handling functions for various error types in Tensor operations
         * 
         * @param errorText Error message to display if error occurs
         * 
         * These overloaded functions handle different types of errors:
         * - CUDA runtime errors (cudaError_t)
         * - cuBLAS errors (cublasStatus_t) 
         * - Memory allocation errors (null float pointers)
         * - Generic error messages (string only)
         *
         * If an error is detected:
         * 1. Prints error message to console
         * 2. Frees Tensor resources
         * 3. Exits program
         *
         * @note These are const member functions that don't modify the Tensor state
         */
        void handleError(std::string errorText) const;

        // MATH

        /**
        * @brief Performs matrix multiplication between this tensor and another tensor using cuBLAS.
        * 
        * This method multiplies two matrices using the NVIDIA cuBLAS library's sgemm operation which performs standard matrix matrix multiplication (works for vectors, too).
        * 
        * @param other The tensor to multiply with (right-hand operand)
        * @return Tensor* A new tensor containing the result of the matrix multiplication
        * @throws std::runtime_error If matrices are incompatible for multiplication or if cuBLAS operation fails
        * 
        * @note Matrix dimensions must be compatible for multiplication:
        *       If this tensor is [m x n], other tensor must be [n x p]
        *       The resulting tensor will be [m x p]
        * 
        * @see matMulCompatible()
        */
        Tensor* matmul(Tensor &other);

        /**
         * @brief Performs a dot product operation between two tensors.
         * 
         * @param other The tensor to perform the dot product with.
         * @return Tensor* Pointer to a new tensor containing the dot product result.
         */
        Tensor* dot(Tensor &other);

        /*Tensor* scale(float scalar);*/

        /**
        * @brief Performs element-wise (Hadamard) multiplication of two tensors.
        * 
        * This method multiplies each element of the current tensor with the corresponding
        * element of the input tensor, resulting in a new tensor of the same shape.
        * The operation is performed on the GPU using CUDA.
        * 
        * @param other The tensor to multiply element-wise with the current tensor
        * @return Tensor* A pointer to a new tensor containing the result of the Hadamard product
        *                 The new tensor maintains the gradient information for backpropagation
        * 
        * @note Both tensors must have the same shape for the operation to be valid
        * @note The resulting tensor is allocated on the GPU
        */
        Tensor* hadamardProduct(Tensor &other);

        /**
         * @brief Adds two tensors element-wise
         * @param other The tensor to add to this tensor
         * @return Pointer to new tensor containing the element-wise sum
         */
        Tensor* add(Tensor &other);

        /**
         * @brief Subtracts two tensors element-wise
         * @param other The tensor to subtract from this tensor
         * @return Pointer to new tensor containing the element-wise difference
         */
        Tensor* sub(Tensor &other);

        /* Optimization */
        
        /**
         * @brief Performs one iteration of stochstic gradient descent on this tensors values. Synchronized.
         * @param lr Learning rate
         * @note Requires gradient to be set
         */
        void sgd(float lr);

        /**
         * @brief Performs one iteration of stochstic gradient descent on this tensors values. Asynchronous, see sync().
         * @param lr Learning rate
         * @note Requires gradient to be set
         */
        void asyncsgd(float lr);

        /* Neural Network */

        /**
         * @brief Performs a standard forward pass. 
         */
        Tensor* sfpass(Tensor &weight, Tensor &bias);

        // OPERATOR OVERLOADING
        
        /**
         * @brief Operator overload for addition
         * @param other The tensor to add to this tensor
         * @return Pointer to new tensor containing the element-wise sum
         */
        Tensor* operator+(Tensor &other);

        /**
         * @brief Operator overload for subtraction  
         * @param other The tensor to subtract from this tensor
         * @return Pointer to new tensor containing the element-wise difference
         */
        Tensor* operator-(Tensor &other);

        /**
         * @brief Operator overload for matrix multiplication
         * @param other The tensor to multiply with this tensor
         * @return Pointer to new tensor containing the matrix product
         */
        Tensor* operator*(Tensor &other);

        /**
         * @brief Operator overload for Hadamard (element-wise) product
         * @param other The tensor to multiply element-wise with this tensor
         * @return Pointer to new tensor containing the Hadamard product
         */
        Tensor* operator%(Tensor &other);

        // PRINTING

        /**
         * @brief Returns stream representing the shape and value of the current Tensor
         */
        friend std::ostream& operator<<(std::ostream &s, const Tensor &tensor);

        /**
         * @brief prints stream representing the shape and value of the current Tensor's value
         */
        void printValue() const;

        /**
         * @brief prints stream representing the shape and value of the current Tensor's gradient
         */
        void printGradient() const;

        // ACTIVATION

        /**
         * @brief Applies ReLU activation function element-wise
         * @return Pointer to new tensor with ReLU applied
         */
        Tensor* relu();

        /**
         * @brief Applies sigmoid activation function element-wise
         * @return Pointer to new tensor with sigmoid applied
         */
        Tensor* sigmoid();

        /**
         * @brief Applies tanh activation function element-wise
         * @return Pointer to new tensor with tanh applied
         */
        Tensor* tanh();

        /**
         * @brief Applies softmax activation function to vector
         * @return Pointer to new tensor with softmax applied
         */
        Tensor* softmax();

        // LOSS FUNCTIONS

        /**
         * @brief Calculates l2 loss
         * @return Pointer to new tensor with l2 loss of <this> and <other>
         */
        Tensor* l2(Tensor &other);

        /**
         * @brief Does not calculate the L2 loss! Can be used for backpropagation if the actual value of the loss is of no interest
         * @return Pointer to new tensor that can be used only for backpropagation of the l2 loss
         * @note Only use for backpropagation: The actual value of the resulting Tensor points to allocated but uninitalized memory!
         */
        Tensor* l2NoVal(Tensor &other);

        /**
         * @brief Applies softmax and calculates categorical cross entropy loss between to distributions
         * @return Pointer to new tensor with l2 loss of <this> and <other>
         */
        Tensor* categoricalCrossEntropy(Tensor &other);

        // Matrix operations

        /**
         * @brief Returns a new Tensor holding the transpose of this Tensor
         */
        Tensor* transpose();
};

#endif
