#include "cudaMath.cu"
#include "cudaMem.cu"


/*All of the below implemented function return a pointer to the corresponding result location, this location is equal to the one passed as a parameter*/


// MEMORY 


// ACTIVATION FUNCTIONS

__global__ void relu_kernel(float* d_targetMemorySpace, float* vector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector[i] > 0) {
        d_targetMemorySpace[i] = vector[i];
    } else {
        d_targetMemorySpace[i] = 0;
    }
}

// applies the relu activation function to a vector of arbitrary size (however only vectors are allowed no other type of tensors)
cudaError_t relu(float* d_targetMemorySpace, float* d_vector, unsigned int size) {
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    relu_kernel<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_vector);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    return cudaSuccess;
}


// DIFFERENTIATION OF ACTIVATION FUNCTIONS


// DIFFERENTIATION OF LOS FUNCTIONS

// WEIGHT INITIALIZATION

// applies the kaiming he inititalizaion to a memory location of sepcified size
void kaiming_he(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for kaiming he init
    float scaling_factor = 2.0 / out_features;

    weight_init(d_targetMemorySpace, in_features, out_features, scaling_factor, seed);
}

// applies the xavier inititalizaion to a memory location of sepcified size
void xavier(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for xavier init
    float scaling_factor = 1.0 / out_features;

    weight_init(d_targetMemorySpace, in_features, out_features, scaling_factor, seed);
}


// TODO
// FORWARD PASS

// performs a forward pass, stores result in specified memory locoation, assumes all tensors are in gpu memory, does error checking
void forward_layer(float* d_outputMemoryLocation, float* d_weights, float* d_bias, float* d_input, int inputSize, int in_features, int out_features) {
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
}

// WEIGHT UPDATE

// updates the given weight matrix (passed as pointer to float array), performs error checking 
void updateWeightMatrix(float* d_weightMatrixToUpdate, float* d_gradient, unsigned int in_features, unsigned int out_features, float learningRate) {

}

// updates the given bias vector (passed as pointer to float array), performs error checking 
void updateBiasVector(float* d_biasVectorToUpdate, float* d_gradient, unsigned int out_features, float learningRate) {
    
}