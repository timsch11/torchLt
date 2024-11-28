#include <iostream>
#include <stdexcept>
#include "util.cu"
#include "curand_kernel.h"


// MATH FUNCTIONS

__global__ void matmulCol(float* mat, float* vec) {
    // every block tackles multiplication of a column, with a thread for each column entry
    mat[blockDim.x * threadIdx.x + blockIdx.x] *= vec[blockIdx.x];
}

__global__ void matmulAddScaledVectorRow(float* targetMemorySpace, float* matrixToAddUp, int matCols) {
    // assumes matrix shape and targetMemorySpace (vector) are compatible
    float sum = 0;
    for (int i=0; i<matCols; i++) {
        sum += matrixToAddUp[threadIdx.x * matCols + i];
    }
    targetMemorySpace[threadIdx.x] = sum;
}

float* matvecmul(float* mat, int matRows, int matCols, float* vec, int vecSize, float* targetMemorySpace) {
    // check for compatibility
    if (matCols != vecSize) {
        throw std::runtime_error("Incompatible shapes");
    }

    // mulitplication of the respective vectors matrix shape does not change yet
    // create space for multiplication
    float* calc = copyValues(mat, matRows * matCols);


    cudaEvent_t e1, e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);

    cudaStream_t s1;
    cudaStreamCreate(&s1);


    // multiplying, result goes to calc
    cudaEventRecord(e1, s1);
    matmulCol<<<matCols, matRows, 0, s1>>>(calc, vec);
    cudaEventRecord(e2, s1);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, e1, e2);

    std::cout << milliseconds << "ms passed";

    // error handling
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error after matmulCol (func matvecmul): %s\n", cudaGetErrorString(error));
        throw std::runtime_error("CUDA error after matmulCol (func matvecmul)");
    }

    // adding up the scaled vectors
    matmulAddScaledVectorRow<<<1, matRows>>>(targetMemorySpace, calc, matCols);
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

    return targetMemorySpace;
}

// adds one column each
__global__ void addVecEntries(float* vec1, float* vec2, float* targetMemorySpace) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    targetMemorySpace[idx] = vec1[idx] + vec2[idx];
}

// adds the values of two vectors and stores the result in <targetMemorySpace>, Note: for efficiency reasons size of targetMemorySpace must be rounded up to a multiple of blocksize
cudaError_t vecadd(float* vector1, unsigned int vectorSize1, float* vector2, unsigned int vectorSize2, float* targetMemorySpace) {
    // check for vector compatibility
    if (vectorSize1 != vectorSize2) {
        printf("vectors to be added have different shapes\n");
        return cudaErrorInvalidValue;
    }

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(vectorSize1);
    
    addVecEntries<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(vector1, vector2, targetMemorySpace);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    return cudaSuccess;
}
/*
__global__ hadamard_kernel(float* targetMemorySpace, float* tensor1, float* tensor2) {
    targetMemorySpace[blockIdx.x * blockDim.x * threadIdx.x] = 
}

cudaError_t hadamard(float* targetMemorySpace, float* tensor1, float* tensor2, std::pair<unsigned int, unsigned int> shape) {
    if (targetMemorySpace == nullptr) {
        // TODO
    }

    // calculate #Block and #Thread
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(vectorSize1);
    hadamard_kernel<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(targetMemorySpace, tensor1, tensor2);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return cudaSuccess;
}
*/

// WEIGHT INITIALIZATION FUNCTIONS


__global__ void cuda_weight_init(float* weights, unsigned int size, float scalingFactor, int seed) {
    // declaring random state
    curandState state;

    // set index
    int ind = blockDim.x * blockIdx.x + threadIdx.x;

    // init curand
    curand_init(seed + ind, blockIdx.x, 0, &state);

    // set weight
    weights[ind] = curand_normal(&state) * sqrtf(scalingFactor);
}

// fills matrix of specified shape with values sampled from a random normal distribution N~(0, sqrt(<scalingFactor>))
void weight_init(float* targetMemorySpace, unsigned int in_features, unsigned int out_features, float scaling_factor, int seed) {

    // set size, add some padding to ensure that kernel runs efficiently but also does not override other memory cells
    unsigned int size = in_features * out_features;
    size += size % BLOCK_SIZE;

    // run kernel
    cuda_weight_init<<<in_features, out_features>>>(targetMemorySpace, size, scaling_factor, seed);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


/*
int main() {
    float* bias = zeros(3);
    float* inp = zeros(3);

    Tensor t1 = Tensor(bias, 3, 0, false);
    Tensor t2 = Tensor(inp, 3, 0, false);

    Tensor* t3 = t1 + t2;

    std::cout << t3->getValue();
    std::cout << t1.getValue();
    std::cout << t2.getValue();

    return 0;
    /*float* weights = zeros(1048576);

    float* bias = zeros(1024);

    float* inp = zeros(1024);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error while zero initialization: %s\n", cudaGetErrorString(error));
    }

    std::cout << std::endl;
    
    float* weightsModified = forward_layer(weights, bias, inp, 1024, 1024, 1024); // (weights, 3, 5, inp, 5);

    /*for (int i = 0; i < 3; i++) {
        std::cout << weightsModified[i] << std::endl;
    }
    

    cudaFree(weights);
    cudaFree(weightsModified);
    cudaFree(inp);

    return 0;
}
*/ 