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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[i] = 1.0f / (1.0f + expf(-d_tensor[i]));
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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[i] = 1.0f - (2.0f / (expf(2.0f * d_tensor[i]) + 1));
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


#define TILE_DIM 16


__global__ void __matMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    // Block index
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread index within the block
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Shared memory for sub-matrices A and B^T
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];

    float C_value = 0.0f;

    // Loop over sub-matrices of A and B^T
    for (int m = 0; m < (K + TILE_DIM - 1) / TILE_DIM; ++m) {
        // Load data into shared memory (A and B^T)
        if (row < TILE_DIM && m * TILE_DIM + col < K) {
            A_shared[row][col] = A[(blockRow * TILE_DIM + row) * K + m * TILE_DIM + col];
        } else {
            A_shared[row][col] = 0.0f;
        }
        if (col < TILE_DIM && m * TILE_DIM + row < K && blockCol * TILE_DIM + col < N) {
            B_shared[row][col] = B[(blockCol * TILE_DIM + col) * K + (m * TILE_DIM + row)];
        } else {
            B_shared[row][col] = 0.0f;
        }

        // Synchronize to ensure the data is loaded into shared memory
        __syncthreads();

        // Compute the partial dot product
        for (int k = 0; k < TILE_DIM; ++k) {
            C_value += A_shared[row][k] * B_shared[k][col];
        }

        // Synchronize to load the next sub-matrix
        __syncthreads();
    }

    // Store the result in the output matrix C
    if (row < TILE_DIM && blockRow * TILE_DIM + row < M && blockCol * TILE_DIM + col < N) {
        C[(blockRow * TILE_DIM + row) * N + blockCol * TILE_DIM + col] = C_value;
    }
}

float* matmul__ABT(int ax, int ay, int bx, int by, float *A, float *B, float *C) {
    // Check for dimension compatibility: inner dimensions must match.
    if (ay != by) {
        printf("invalid shapes for matrix multiplication AB^T");
        return nullptr;
    }

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((bx + TILE_DIM - 1) / TILE_DIM, (ax + TILE_DIM - 1) / TILE_DIM); // Grid size

    __matMulTiled<<<gridDim, blockDim>>>(A, B, C, ax, bx, ay);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    return C;
}

float* matmul__ATB(cublasHandle_t* handle, int ax, int ay, int bx, int by, const float *A, const float *B, float *C) {
    // Check for dimension compatibility: inner dimensions must match.
    if (ax != bx) {
        printf("invalid shapes for matrix multiplication");
        return nullptr;
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t matmulStatus;

    // (B^TA)^T = rowmajor(AB)
    matmulStatus = cublasSgemm_v2(*handle,
        CUBLAS_OP_N, CUBLAS_OP_T, // No (second) transpose for both A and B
        by, ay, bx, // # ax -> ay
        &alpha, B, by, // A is m x k
        A, ay, // B is k x n
        &beta, C, by); // C is m x n

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (matmulStatus != CUBLAS_STATUS_SUCCESS) {
        cudaFree(C);
        std::cout << "matrix multiplication failed: " << std::string(cublasGetStatusString(matmulStatus));
        return nullptr;
    }

    // C is stored in device memory; it represents the result in row-major format.
    return C;
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
                CUBLAS_OP_N, CUBLAS_OP_N, // No (second) transpose for both A and B
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

__global__ void __elementWiseL2Loss(float* d_result, float* d_predicted, float* d_actual) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float dif = d_predicted[i] - d_actual[i];
    d_result[i] = dif*dif;
}

__global__ void addUp(float* d_result, float* d_target, unsigned int elements, unsigned int stop) {
    float sum = 0.0f;

    // calculate start index
    unsigned int start = (blockIdx.x * blockDim.x + threadIdx.x) * elements;

    // add values of a section
    for (unsigned int i=start; (i < start + elements) && (i < stop); i++) {
        sum += d_target[i];
    }

    // perform mutually exclusive add operation
    atomicAdd(d_result, sum);
}

float* l2LossAlloc(float* d_predicted, float* d_actual, std::pair<unsigned int, unsigned int> shape_predicted, std::pair<unsigned int, unsigned int> shape_actual) {
    // check if Tensors are vectors
    if (shape_predicted.second != 1 || shape_actual.second != 1) {
        printf("Error: L2 Loss is only defined for two vectors of equal size, apply transpose if you want to calculate the L2 Loss of a row-vector\n\n");
        return nullptr;
    }

    // check if size of vectors is equal
    if (shape_predicted.first != shape_actual.first) {
        printf("Error: L2 Loss is only defined for two vectors of equal size.\n");
        return nullptr;
    }

    // allocate some memory to carry out calculations quicker
    float* d_calcMem = reserveMemoryOnDevice(shape_predicted.first);

    // check for errors during memory allocation
    if (d_calcMem == nullptr) {
        printf("Error: An error occured during memory allocation in l2LossAlloc\n\n");
        return nullptr;
    }

    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(shape_predicted.first);

    // calculate element-wise (predicted[i]-actual[i])^2 and store individual results (not summed up yet) in calcMem
    __elementWiseL2Loss<<<blocksThreads.first, blocksThreads.second>>>(d_calcMem, d_predicted, d_actual);

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // error checking/handling
    CHECK_CUDA_ERROR(cudaGetLastError());

    // allocate memory for result
    float* d_result = reserveMemoryOnDevice(1);

    // check for errors during memory allocation
    if (d_result == nullptr) {
        printf("Error: An error occurred during result memory allocation in l2LossAlloc.\n");
        return nullptr;
    }

    // init memory
    cudaMemset(d_result, 0.0f, sizeof(float));

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // hyperparamters, every thread should handle k elements
    unsigned int k = 10;
    unsigned int blockSize = 32;
    unsigned int blocks = (shape_predicted.first + (k * blockSize - 1)) / (k * blockSize);

    // add everything up
    addUp<<<blocks, blockSize>>>(d_result, d_calcMem, k, shape_predicted.first);

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // error checking/handling
    CHECK_CUDA_ERROR(cudaGetLastError());

    // free intermediate calculation memory
    CHECK_CUDA_ERROR(cudaFree(d_calcMem));

    return d_result;
}