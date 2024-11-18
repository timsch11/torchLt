#include <iostream>
#include <stdexcept>
#include "curand_kernel.h"


// MEMORY INITIALIZATION FUNCTIONS

__global__ void initZero(float* memorySection) {
    memorySection[blockIdx.x * blockDim.x + threadIdx.x] = -1;
}

float* zeros(int size) {
    float* memoryAllocation;

    cudaMallocManaged(&memoryAllocation, size*sizeof(float));

    // maybe think about block/thread distribution
    initZero<<<1, size>>>(memoryAllocation);
    cudaDeviceSynchronize();

    return memoryAllocation;
}

__global__ void copyValue(float* target, float* valueToCopy, int size) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < size) {
        // deepcopy?
        target[ind] = valueToCopy[ind];
    }
}

float* copyValues(float* valueToCopy, int size) {
    float* output;

    cudaMallocManaged(&output, size * sizeof(float));

    int blockSize = 256;
    int blockNum = size / 256;

    copyValue<<<blockNum, blockSize>>>(output, valueToCopy, size);

    return output;
}


// MATH FUNCTIONS

__global__ void matmulCol(float* mat, float* vec) {
    // every block tackles multiplication of a column, with a thread for each column entry
    mat[blockDim.x * threadIdx.x + blockIdx.x] *= vec[blockIdx.x];
}

float* matvecmul(float* mat, int matRows, int matCols, float* vec, int vecSize) {
    // mat is used for calculations and result
    if (matCols != vecSize) {
        throw std::runtime_error("Incompatible shapes");
    }
    // multiplying, result goes to mat
    matmulCol<<<matCols, matRows>>>(mat, vec);
    cudaDeviceSynchronize();

    // adding up

    return mat;
}


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

float* forward_layer(float* weights, float* bias, float* input, int inputSize, int in_features, int out_features) {
    // weights and bias are already on cuda (shared memory)

    int size = in_features * out_features;

    // reserve memory for calculations, init with weights
    float* calc = copyValues(weights, size);

    // reserve memory for output, init with zeros
    float* output = zeros(out_features);

    // add try-catch for matvecmul !!!
    // matvecmul uses mat for calculation space
    try
    {
        // check again
        matvecmul(calc, out_features, in_features, input, inputSize);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << e.what() << '\n';
    }
    

    return weights;
}


int main() {
    int in_features = 4;
    int out_features = 4;

    float* weights = zeros(6);

    float* inp = zeros(2);

    for (int i = 0; i < 6; i++) {
        std::cout << weights[i] << std::endl;
    }

    matvecmul(weights, 3, 2, inp, 2);

    for (int i = 0; i < 6; i++) {
        std::cout << weights[i] << std::endl;
    }

    return 0;

    // TODO: finc matrix mul bug and change zeros back to actually zero
}