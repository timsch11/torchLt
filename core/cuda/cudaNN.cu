#include "cudaNN.cuh"


// ACTIVATION FUNCTIONS


__global__ void __relu(float* d_targetMemorySpace, float* vector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector[i] > 0) {
        d_targetMemorySpace[i] = vector[i];
    } else {
        d_targetMemorySpace[i] = 0;
    }
}

cudaError_t relu(float* d_targetMemorySpace, float* d_vector, unsigned int size) {
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    __relu<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_vector);

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return cudaGetLastError();
}

float* reluAlloc(float* d_vector, unsigned int size) {
    float* result = reserveMemoryOnDevice(size);
    relu(result, d_vector, size);
    return result;
}

__global__ void __sigmoid(float* d_targetMemorySpace, float* d_tensor) {
    unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[ind] = 1.0f / (1.0f + expf(-d_tensor[ind]));
}

cudaError_t sigmoid(float* d_targetMemorySpace, float* d_tensor, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    // execute computation
    __sigmoid<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_tensor);

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // return cudaSuccess_t or error
    return cudaGetLastError();
}

float* sigmoidAlloc(float* d_tensor, unsigned int size) {
    // allocate required memory (+padding)
    float* d_result = reserveMemoryOnDevice(size);

    // check for errors
    CHECK_CUDA_ERROR(sigmoid(d_result, d_tensor, size));

    return d_result;
}

__global__ void __tanh(float* d_targetMemorySpace, float* d_tensor) {
    unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[ind] = 1.0f - (2.0f / (expf(2.0f * d_tensor[ind]) + 1));
}

cudaError_t tanh(float* d_targetMemorySpace, float* d_tensor, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    // execute computation
    __tanh<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_tensor);

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // return cudaSuccess_t or error
    return cudaGetLastError();
}

float* tanhAlloc(float* d_tensor, unsigned int size) {
    // allocate required memory (+padding)
    float* d_result = reserveMemoryOnDevice(size);

    // check for errors
    CHECK_CUDA_ERROR(tanh(d_result, d_tensor, size));

    return d_result;
}

// WEIGHT INITIALIZATION


void kaiming_he(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for kaiming he init
    float scaling_factor = 2.0 / out_features;

    weight_init(d_targetMemorySpace, in_features * out_features, scaling_factor, seed);
}

void xavier(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for xavier init
    float scaling_factor = 1.0 / out_features;

    weight_init(d_targetMemorySpace, in_features * out_features, scaling_factor, seed);
}

// WEIGHT UPDATE

// updates the given weight matrix (passed as pointer to float array), performs error checking 
void updateWeightMatrix(float* d_weightMatrixToUpdate, float* d_gradient, unsigned int in_features, unsigned int out_features, float learningRate) {

}

// updates the given bias vector (passed as pointer to float array), performs error checking 
void updateBiasVector(float* d_biasVectorToUpdate, float* d_gradient, unsigned int out_features, float learningRate) {
    
}

// MATH


float* hadamardAlloc(float* d_tensor1, std::pair<unsigned int, unsigned int> shapeT1, float* d_tensor2, std::pair<unsigned int, unsigned int> shapeT2) {

    // check for compatibility
    if (shapeT1.first != shapeT2.first || shapeT1.second != shapeT2.second) {
        throw std::runtime_error("incompatible shapes for hadamard product");
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(shapeT1.first * shapeT1.second);

    // perform computation
    CHECK_CUDA_ERROR(hadamard(d_result, d_tensor1, d_tensor2, shapeT1));

    // return pointer to result
    return d_result;

} 

float* tensoraddAlloc(float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {
    // check for compatibility
    if (vectorSize1 != vectorSize2) {
        throw std::runtime_error("incompatible shapes for vector addition");
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(vectorSize1);

    // perform computation
    CHECK_CUDA_ERROR(tensoradd(d_result, d_vector1, vectorSize1, d_vector2, vectorSize2));

    // return pointer to result
    return d_result;
}

float* tensorsubAlloc(float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {
    // check for compatibility
    if (vectorSize1 != vectorSize2) {
        throw std::runtime_error("incompatible shapes for vector subtraction");
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(vectorSize1);

    // perform computation
    CHECK_CUDA_ERROR(vecsub(d_result, d_vector1, vectorSize1, d_vector2, vectorSize2));

    // return pointer to result
    return d_result;
}

float* matmulAlloc(cublasHandle_t* handle, int ax, int ay, int bx, int by, const float *A, const float *B) {
    if (ay != bx) {
        throw std::runtime_error("invalid shapes for matrix multiplciation");
    }

    // allocate memory
    float* C = reserveMemoryOnDevice(ax * by);
    
    float alpha = 1.0f;
    float beta = 0.0f;

    // call cuBLAS
    // c++ uses row major format, cuBLAS uses column major format
    // rowMajor(A) = columnMajor(A)T
    // this function essentially computes C = (BT AT)T
    cublasStatus_t matmulStatus = cublasSgemm_v2(*handle,
                CUBLAS_OP_N, CUBLAS_OP_N, // No transpose for both A and B
                by, ax, bx,
                &alpha, B, by, // A is m x k
                A, bx, // B is k x n
                &beta, C, by); // C is m x n

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (matmulStatus != CUBLAS_STATUS_SUCCESS) {
        cudaFree(C);
        std::cout << std::string(cublasGetStatusString(matmulStatus));
        throw std::runtime_error("matrix multiplication failed: " + 
            std::string(cublasGetStatusString(matmulStatus)));
    }

    // return pointer to result
    return C;
}