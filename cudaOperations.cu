#include <iostream>
#include <stdexcept>
#include "curand_kernel.h"


// MEMORY INITIALIZATION FUNCTIONS

__global__ void initZero(float* memorySection) {
    memorySection[blockIdx.x * blockDim.x + threadIdx.x] = -1;
}

float* zeros(int size) {
    // returns a pointer to (first element of) an array (interpretation of dimension is up to the caller) of specified size filled with zeros; array lives in unified memory (on cpu and gpu)

    // reserve memory
    float* memoryAllocation;
    cudaMallocManaged(&memoryAllocation, size * sizeof(float));

    // maybe think about block/thread distribution
    // definitively a TODO (for later)
    initZero<<<1, size>>>(memoryAllocation);
    cudaDeviceSynchronize();

    // check for errors
    cudaError_t error  = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error when init with zeros: %s\n", cudaGetErrorString(error));
    }

    return memoryAllocation;
}

__global__ void copyValue(float* target, float* valueToCopy, int size) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    if (ind < size) {
        target[ind] = valueToCopy[ind];
    }
}

float* copyValuesUnified(float* valueToCopy, int size) {
    // copies values of desire into unified memory, performs deepcopy
    float* output;
    cudaMallocManaged(&output, size * sizeof(float));

    int blockSize = 256;
    int blockNum = size / 256;

    copyValue<<<blockNum + 1, blockSize>>>(output, valueToCopy, size);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error while trying to copy values to cuda (unified): %s\n", cudaGetErrorString(error));
    }

    return output;
}

float* copyValues(float* valueToCopy, int size) {
    // copies values of desire into device memory, performs deepcopy
    float* output;
    cudaMalloc(&output, size * sizeof(float));

    int blockSize = 256;
    int blockNum = size / 256;

    copyValue<<<blockNum + 1, blockSize>>>(output, valueToCopy, size);

    // do some error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error while trying to copy values to cuda: %s\n", cudaGetErrorString(error));
    }

    return output;
}

// MATH FUNCTIONS

__global__ void matmulCol(float* mat, float* vec, int blocks) {
    // every block tackles multiplication of a column, with a thread for each column entry
    mat[blocks * threadIdx.x + blockIdx.x] *= vec[blockIdx.x];
}

__global__ void matmulAddScaledVectorRow(float* targetMemorySpace, float* matrixToAddUp, int matCols) {
    // assumes matrix shape and targetMemorySpace (vector) are compatible
    float sum = 0;
    for (int i=0; i<matCols; i++) {
        sum += matrixToAddUp[threadIdx.x * matCols + i];
    }
    targetMemorySpace[threadIdx.x] = sum;
}

float* matvecmul(float* mat, int matRows, int matCols, float* vec, int vecSize) {
    // check for compatibility
    if (matCols != vecSize) {
        throw std::runtime_error("Incompatible shapes");
    }

    // mulitplication of the respective vectors matrix shape does not change yet
    // create space for multiplication
    float* calc = copyValues(mat, matRows * matCols);

    // multiplying, result goes to calc
    matmulCol<<<matCols, matRows>>>(calc, vec, matCols);
    cudaDeviceSynchronize();

    // error handling
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error after matmulCol (func matvecmul): %s\n", cudaGetErrorString(error));
        throw std::runtime_error("CUDA error after matmulCol (func matvecmul)");
    }

    // adding up the scaled vectors
    // create space for output
    float* result = copyValuesUnified(vec, vecSize);

    matmulAddScaledVectorRow<<<1, matRows>>>(result, calc, matCols);
    cudaDeviceSynchronize();

    // error handling the second
    error  = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error when adding up in (func matvecmul): %s\n", cudaGetErrorString(error));
        throw std::runtime_error("CUDA error after matmulCol (func matvecmul)");
    }

    // free memory only used for mulitplication
    cudaFree(calc);

    error  = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error when freeing memory (func matvecmul): %s\n", cudaGetErrorString(error));
        throw std::runtime_error("CUDA error after matmulCol (func matvecmul)");
    }

    return result;
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

    try {
        // check again
        float* output = matvecmul(weights, out_features, in_features, input, inputSize);
    }
    catch(const std::runtime_error& e) {
        std::cerr << e.what() << '\n';
    }
    

    return weights;
}


int main() {
    float* weights = zeros(15);

    float* inp = zeros(5);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error while zero initialization: %s\n", cudaGetErrorString(error));
    }

    for (int i = 0; i < 15; i++) {
        std::cout << weights[i] << std::endl;
    }

    std::cout << std::endl;

    float* weightsModified = matvecmul(weights, 3, 5, inp, 5);

    for (int i = 0; i < 3; i++) {
        std::cout << weightsModified[i] << std::endl;
    }

    cudaFree(weights);
    cudaFree(weightsModified);
    cudaFree(inp);

    return 0;

    // TODO: find matrix mul bug and change zeros back to actually zero
}