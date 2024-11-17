#include <iostream>
#include "curand_kernel.h"


// WEIGHT INITIALIZATION FUNCTIONS

__global__ void cuda_weight_init(float* weights, int layer_size, int size, float scaling_factor, int seed) {
    // declaring random state
    curandState state;

    // set index
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    // init curand
    curand_init(seed + ind, blockIdx.x, 0, &state);

    if (ind < size) {
        weights[ind] = curand_normal(&state) * sqrtf(scaling_factor);
    }
}


float* weight_init(int in_features, int out_features, float scaling_factor, int seed) {

    // declare weights
    float* weights;

    // set size
    int size = in_features * out_features;

    // allocate memory
    cudaMallocManaged(&weights, size * sizeof(float));

    // run kernel
    cuda_weight_init<<<in_features, out_features>>>(weights, out_features, size, scaling_factor, seed);

    // error handling
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // return pointer to shared memory (between cpu AND gpu)
    return weights;
}


float* kaiming_he(int in_features, int out_features, int seed) {
    // set scaling factor for kaiming he init
    float scaling_factor = 2.0 / out_features;

    return weight_init(in_features, out_features, scaling_factor, seed);
}

float* xavier(int in_features, int out_features, int seed) {
    // set scaling factor for xavier init
    float scaling_factor = 1.0 / out_features;

    return weight_init(in_features, out_features, scaling_factor, seed);
}


// FORWARD PASS FUNCTIONS (LAYER)

float* forward_layer(float* weights, float* bias, float* input, int in_features, int out_features) {
    // declare output
    float* output;
    return weights;
}


int main() {
    int in_features = 4;
    int out_features = 4;

    float* weights = kaiming_he(in_features, out_features, 1);
    
    for (int i = 0; i < in_features * out_features; i++) {
        std::cout << weights[i] << std::endl;
    }

    return 0;
}