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
    __relu<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_vector);

    cudaError_t err = cudaGetLastError();
    
    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

float* reluAlloc(float* d_vector, unsigned int size) {
    float* result = reserveMemoryOnDevice(size);

    if (result == nullptr) {
        return nullptr;
    }

    cudaError_t err = relu(result, d_vector, size);

    if (err != cudaSuccess) {
        std::cout << "Cuda error when performing relu: " << std::string(cudaGetErrorString(err));
        return nullptr;
    }

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
    __sigmoid<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor);

    cudaError_t err = cudaGetLastError();
    
    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

float* sigmoidAlloc(float* d_tensor, unsigned int size) {
    // allocate required memory (+padding)
    float* d_result = reserveMemoryOnDevice(size);

    if (d_result == nullptr) {
        return nullptr;
    }

    // check for errors
    cudaError_t err = sigmoid(d_result, d_tensor, size);


    if (err != cudaSuccess) {
        std::cout << "Cuda error when performing sigmoid: " << std::string(cudaGetErrorString(err));
        return nullptr;
    }

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
    __tanh<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor);

    cudaError_t err = cudaGetLastError();
    
    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

float* tanhAlloc(float* d_tensor, unsigned int size) {
    // allocate required memory (+padding)
    float* d_result = reserveMemoryOnDevice(size);

    if (d_result == nullptr) {
        return nullptr;
    }

    // check for errors
    cudaError_t err = tanh(d_result, d_tensor, size);

    if (err != cudaSuccess) {
        std::cout << "Cuda error when performing tanh: " << std::string(cudaGetErrorString(err));
        return nullptr;
    }

    return d_result;
}

// WEIGHT INITIALIZATION


cudaError_t kaiming_he(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for kaiming he init
    float scaling_factor = 2.0 / out_features;

    return weight_init(d_targetMemorySpace, in_features * out_features, scaling_factor, seed);
}

cudaError_t xavier(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for xavier init
    float scaling_factor = 1.0 / out_features;

    return weight_init(d_targetMemorySpace, in_features * out_features, scaling_factor, seed);
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
        printf("incompatible shapes for hadamard product");
        return nullptr;
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(shapeT1.first * shapeT1.second);

    if (d_result == nullptr) {
        printf("Error when allocating memory in hadamardAlloc");
        return nullptr;
    }

    // perform computation
    cudaError_t err = hadamard(d_result, d_tensor1, d_tensor2, shapeT1);

    if (err != cudaSuccess) {
        printf("Error when performing hadamard");
        return nullptr;
    }

    // return pointer to result
    return d_result;

} 

float* tensoraddAlloc(float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {
    // check for compatibility
    if (vectorSize1 != vectorSize2) {
        printf("incompatible shapes for vector addition");
        return nullptr;
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(vectorSize1);

    if (d_result == nullptr) {
        printf("Error when allocating memory in hadamardAlloc");
        return nullptr;
    }

    // perform computation
    cudaError_t err = tensoradd(d_result, d_vector1, vectorSize1, d_vector2, vectorSize2);

    if (err != cudaSuccess) {
        printf("Error when performing add");
        return nullptr;
    }

    // return pointer to result
    return d_result;
}

float* tensorsubAlloc(float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {
    // check for compatibility
    if (vectorSize1 != vectorSize2) {
        printf("incompatible shapes for vector subtraction");
        return nullptr;
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(vectorSize1);

    if (d_result == nullptr) {
        printf("Error when allocating memory in hadamardAlloc");
        return nullptr;
    }

    // perform computation
    cudaError_t err = vecsub(d_result, d_vector1, vectorSize1, d_vector2, vectorSize2);

    if (err != cudaSuccess) {
        printf("Error when performing sub");
        return nullptr;
    }

    // return pointer to result
    return d_result;
}

float* matmulAlloc(cublasHandle_t* handle, int ax, int ay, int bx, int by, const float *A, const float *B) {
    if (ay != bx) {
        printf("invalid shapes for matrix multiplciation");
        return nullptr;
    }

    // allocate memory
    float* C = reserveMemoryOnDevice(ax * by);

    if (C == nullptr) {
        printf("Error when allocating memory for result of matmul");
        return nullptr;
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t matmulStatus;

    // call cuBLAS
    // c++ uses row major format, cuBLAS uses column major format
    // rowMajor(A) = columnMajor(A)T
    // this function essentially computes C = (BT AT)T
    matmulStatus = cublasSgemm_v2(*handle,
                CUBLAS_OP_N, CUBLAS_OP_N, // No transpose for both A and B
                by, ax, bx,
                &alpha, B, by, // A is m x k
                A, bx, // B is k x n
                &beta, C, by); // C is m x n

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (matmulStatus != CUBLAS_STATUS_SUCCESS) {
        cudaFree(C);
        std::cout << "matrix multiplication failed: " << std::string(cublasGetStatusString(matmulStatus));
        return nullptr;
    }

    // return pointer to result
    return C;
}