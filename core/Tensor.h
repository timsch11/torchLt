#include "cublas_v2.h"
#include <iostream>

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

        // streams for synchronized/concurrent execution

        cudaStream_t* graphStream;  // used for operations inside of the computational graph that require synchronization
                                   // all nodes of a computational graph share the same graphStream
        cudaStream_t nodeStream;   // used for operations inside of the computational graph that do not require synchronization
                                   // every Tensor has a unique nodeStream

        // size of the computational graph lower than this node (including this node)

        unsigned int lowerGraphSize;

        // shapes of function arguments

        std::pair<unsigned int, unsigned int> shapeFuncArg1;
        std::pair<unsigned int, unsigned int> shapeFuncArg2;

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
        Tensor(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed, void(*initalization_function)(float*, unsigned int, unsigned int, int));
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
         * @brief Get pointer to CUDA stream associated with this tensor's computational graph
         * @return Pointer to cudaStream_t stream used for asynchronous operations
         */
        cudaStream_t* getGraphStream() const;
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
        /**
         * @brief Get size of computational graph below this tensor
         * @return Number of nodes in the subgraph where this tensor is root
         */
        unsigned int getLowerGraphSize() const;

        // SETTER

        /**
         * @brief Changes the gradient status of the tensor (if gradient is tracked)
         * @param _gradientSet Boolean indicating whether gradient is set
         */
        void changeGradientSet(bool _gradientSet);
        /**
         * @brief Sets the CUDA stream for gradient computation for this tensor and all preceding nodes in computation graph
         * @param graphStream Pointer to the CUDA stream to be set
         */
        void setGraphStreamForSubgraph(cudaStream_t* graphStream);
        /**
         * @brief Performs backward pass gradient computation through the computation graph
         * @note Calls the gradient function stored in the tensor to compute gradients if gradient tracking is enabled
         */
        void backward();

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
        Tensor* matmul(Tensor & other);
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
        /**
         * @brief Operator overload for addition
         * @param other The tensor to add to this tensor
         * @return Pointer to new tensor containing the element-wise sum
         */

        // OPERATOR OVERLOADING
        
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
};