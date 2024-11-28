#include "util.cu"


/*All of the below implemented function return a pointer to the corresponding result location, this location is equal to the one passed as a parameter*/

// Activation Functions

__global__ relu_kernel(float* vector, float* targetMemorySpace) {

}

// applies the relu activation function to a vector of arbitrary size (however only vectors are allowed no other type of tenors)
float* relu(float* targetMemorySpace, float* vector, unsigned int length) {

}


// Weight initialization techniques

// applies the kaiming he inititalizaion to a memory location of sepcified size
float* kaiming_he(float* targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for kaiming he init
    float scaling_factor = 2.0 / out_features;

    return weight_init(targetMemorySpace, in_features, out_features, scaling_factor, seed);
}

// applies the xavier inititalizaion to a memory location of sepcified size
float* xavier(float* targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for xavier init
    float scaling_factor = 1.0 / out_features;

    return weight_init(targetMemorySpace, in_features, out_features, scaling_factor, seed);
}





// TODO
// FORWARD PASS FUNCTIONS (LAYER)

float* forward_layer(float* weights, float* bias, float* input, int inputSize, int in_features, int out_features) {
    // assume weights matrix is out_features x in_features
    // weights and bias are already on cuda (shared memory)

    // load input into shared memory
    float* d_input = copyValuesUnified(input, inputSize);

    int size = in_features * out_features;

    try {

        // perform matrix multiplication with weights and input
        float* output = matvecmul(weights, out_features, in_features, input, inputSize);

        // add bias to result of matrix multiplication
        vecadd(output, inputSize, input, inputSize, output);

        return output;
    }
    catch(const std::runtime_error& e) {
        std::cerr << e.what() << '\n';
    }
}