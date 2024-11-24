#include <iostream>
#include <stdexcept>
#include "curand_kernel.h"
#include "Tensor.h"


// set preferred block size
#define BLOCK_SIZE 256

std::pair<unsigned int, unsigned int> computeBlockThreadAllocation(int size) {
    unsigned int blockNum = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int threadNum = BLOCK_SIZE;
    return {blockNum, threadNum};
}


// error checking
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}


// MEMORY INITIALIZATION FUNCTIONS


__global__ void initZero(float* memorySection) {
    memorySection[blockIdx.x * blockDim.x + threadIdx.x] = 0;
}

float* zeros(int size) {
    // returns a pointer to (first element of) an array (interpretation of dimension is up to the caller) of specified size filled with zeros; array lives in unified memory (on cpu and gpu)

    // calc block/thread allocation scheme
    int blockNum = size / 256;
    int memOffset = size % BLOCK_SIZE;

    // reserve memory
    float* memoryAllocation;
    CHECK_CUDA_ERROR(cudaMallocManaged(&memoryAllocation, size * sizeof(float) + memOffset));

    // launch kernel
    initZero<<<blockNum+1, BLOCK_SIZE>>>(memoryAllocation);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

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

    int blockNum = size / 256;

    copyValue<<<blockNum + 1, BLOCK_SIZE>>>(output, valueToCopy, size);

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

    int blockNum = size / 256;

    copyValue<<<blockNum + 1, BLOCK_SIZE>>>(output, valueToCopy, size);

    // do some error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error while trying to copy values to cuda: %s\n", cudaGetErrorString(error));
    }

    return output;
}

float* reserveMemoryOnDevice(unsigned int size) {
    // declare pointer
    float* memoryAlloc;

    // reserve actual space in memory, add some padding for thread efficiency
    CHECK_CUDA_ERROR(cudaMalloc(&memoryAlloc, size + (size / BLOCK_SIZE)));

    // return pointer 
    return memoryAlloc;
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
    matmulCol<<<matCols, matRows, 0, s1>>>(calc, vec, matCols);
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
    
    addVecEntries<<<blocksThreads.first, blocksThreads.second>>>(vector1, vector2, targetMemorySpace);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    return cudaSuccess;
}


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

// returns a pointer to a weight matrix of specified shape with values sampled from a normal distribution N~(0, sqrt(<scalingFactor>))
float* weight_init(int in_features, int out_features, float scaling_factor, int seed) {

    // declare weights
    float* weights;

    // set size, add some padding to ensure that kernel runs efficiently but also does not override other memory cells
    int size = in_features * out_features;
    size += size % BLOCK_SIZE;

    // allocate memory
    CHECK_CUDA_ERROR(cudaMallocManaged(&weights, size * sizeof(float)));

    // run kernel
    cuda_weight_init<<<in_features, out_features>>>(weights, size, scaling_factor, seed);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

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