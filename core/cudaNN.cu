#include "cudaMem.cu"
#include "cudaMath.cu"
#include "cudaDif.cu"

/*All of the below implemented function return a pointer to the corresponding result location, this location is equal to the one passed as a parameter*/


// ACTIVATION FUNCTIONS

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
__global__ void relu_kernel(float* d_targetMemorySpace, float* vector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector[i] > 0) {
        d_targetMemorySpace[i] = vector[i];
    } else {
        d_targetMemorySpace[i] = 0;
    }
}

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
cudaError_t relu(float* d_targetMemorySpace, float* d_vector, unsigned int size) {
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    relu_kernel<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_vector);
    return cudaGetLastError();
}

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
float* reluAlloc(float* d_vector, unsigned int size) {
    float* result = reserveMemoryOnDevice(size);
    relu(result, d_vector, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return result;
}


// DIFFERENTIATION OF ACTIVATION FUNCTIONS


// DIFFERENTIATION OF LOSS FUNCTIONS

// WEIGHT INITIALIZATION

// applies the kaiming he inititalizaion to a memory location of sepcified size
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
void kaiming_he(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for kaiming he init
    float scaling_factor = 2.0 / out_features;

    weight_init(d_targetMemorySpace, in_features * out_features, scaling_factor, seed);
}

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
void xavier(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for xavier init
    float scaling_factor = 1.0 / out_features;

    weight_init(d_targetMemorySpace, in_features * out_features, scaling_factor, seed);
}


// TODO
// FORWARD PASS

// performs a forward pass, stores result in specified memory locoation, assumes all tensors are in gpu memory, does error checking
/*void forward_layer(float* d_outputMemoryLocation, float* d_weights, float* d_bias, float* d_input, int inputSize, int in_features, int out_features) {
    try {

        // perform matrix multiplication with weights and input
        float* output = matvecmul(d_weights, out_features, in_features, d_input, inputSize);

        // add bias to result of matrix multiplication
        vecadd(output, inputSize, input, inputSize, output);

        return output;
    }
    catch(const std::runtime_error& e) {
        std::cerr << e.what() << '\n';
    }
}*/

// WEIGHT UPDATE

// updates the given weight matrix (passed as pointer to float array), performs error checking 
void updateWeightMatrix(float* d_weightMatrixToUpdate, float* d_gradient, unsigned int in_features, unsigned int out_features, float learningRate) {

}

// updates the given bias vector (passed as pointer to float array), performs error checking 
void updateBiasVector(float* d_biasVectorToUpdate, float* d_gradient, unsigned int out_features, float learningRate) {
    
}

// MATH


// performs the hadamard product and stores result in newly created memory allocation, returns pointer
float* hadamardAlloc(float* d_tensor1, std::pair<unsigned int, unsigned int> shapeT1, float* d_tensor2, std::pair<unsigned int, unsigned int> shapeT2) {

    // check for compatibility
    if (shapeT1.first != shapeT2.first || shapeT1.second != shapeT2.second) {
        throw std::runtime_error("incompatible shapes for hadamard product");
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(shapeT1.first * shapeT1.second);

    // perform computation
    hadamard(d_result, d_tensor1, d_tensor2, shapeT1);

    // return pointer to result
    return d_result;

} // TOBEOPTIMIZED


// performs vector addition and stores the result in a newly created array (residing on gpu)
float* vecaddAlloc(float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {
    // check for compatibility
    if (vectorSize1 != vectorSize2) {
        throw std::runtime_error("incompatible shapes for vector addition");
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(vectorSize1);

    // perform computation
    tensoradd(d_result, d_vector1, vectorSize1, d_vector2, vectorSize2);

    // return pointer to result
    return d_result;

} // TOBEOPTIMIZED

// performs vector subtraction and stores the result in a newly created array (residing on gpu)
float* vecsubAlloc(float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {
    // check for compatibility
    if (vectorSize1 != vectorSize2) {
        throw std::runtime_error("incompatible shapes for vector subtraction");
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(vectorSize1);

    // perform computation
    vecsub(d_result, d_vector1, vectorSize1, d_vector2, vectorSize2);

    // return pointer to result
    return d_result;

} // TOBEOPTIMIZED