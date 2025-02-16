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
    float scaling_factor = 2.0 / in_features;

    return weight_init(d_targetMemorySpace, in_features * out_features, scaling_factor, seed);
}

cudaError_t xavier(float* d_targetMemorySpace, unsigned int in_features, unsigned int out_features, int seed) {
    // set scaling factor for xavier init
    float scaling_factor = 2.0 / (in_features + out_features);

    return weight_init(d_targetMemorySpace, in_features * out_features, scaling_factor, seed);
}

float* forwardPass(cublasLtHandle_t* handle, const float* d_weight, const float* d_input, const float* d_bias, int m, int n, int k, cublasOperation_t opA, cublasOperation_t opB) {

    // Allocate device memory
    float* d_output = reserveMemoryOnDevice(m * n);

    // error check
    if (d_output == nullptr) {
        printf("Error: Memory allocation for gemm failed");
        cudaFree(d_output);
        exit(EXIT_FAILURE);
    }

    // Create matrix descriptors
    cublasLtMatrixLayout_t matW, matI, matB, matO;

    cublasLtOrder_t row_major_format = CUBLASLT_ORDER_ROW;

    bool transA = opA == CUBLAS_OP_T;
    bool transB = opB == CUBLAS_OP_T;

    // For weight matrix
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &matW, CUDA_R_32F, transA ? k : m, transA ? m : k, transA ? m : k));

    checkCublasStatus( cublasLtMatrixLayoutSetAttribute(
        matW, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_format, sizeof( row_major_format ) ) );

    // For input vector
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &matI, CUDA_R_32F, transB ? n : k, transB ? k : n, transB ? k : n));
    
    checkCublasStatus( cublasLtMatrixLayoutSetAttribute(
        matI, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_format, sizeof( row_major_format ) ) );

    // For bias vector
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &matB, CUDA_R_32F, m, n, n));

    checkCublasStatus( cublasLtMatrixLayoutSetAttribute(
        matB, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_format, sizeof(row_major_format)));

    // For output vector
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &matO, CUDA_R_32F, m, n, n));

    checkCublasStatus( cublasLtMatrixLayoutSetAttribute(
        matO, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_format, sizeof(row_major_format)));


    // Create operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Set transpose operations
    checkCublasStatus(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
        &opA, sizeof(opA)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
        &opB, sizeof(opB)));

    // Scale factors
    float alpha = 1.0f;
    float beta = 1.0f;

    // Perform matrix multiplication
    checkCublasStatus(cublasLtMatmul(
        *handle,
        operationDesc,
        &alpha,
        d_weight,
        matW,
        d_input,
        matI,
        &beta,
        d_bias,
        matB,
        d_output,
        matO,
        nullptr,
        nullptr,
        0,
        0));

    // wait for completion
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Cleanup
    cublasLtMatrixLayoutDestroy(matW);
    cublasLtMatrixLayoutDestroy(matI);
    cublasLtMatrixLayoutDestroy(matB);
    cublasLtMatrixLayoutDestroy(matO);
    cublasLtMatmulDescDestroy(operationDesc);

    return d_output;
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
        printf("Error when allocating memory in tensoraddAlloc");
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
        printf("incompatible shapes for vector addition");
        return nullptr;
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(vectorSize1);

    if (d_result == nullptr) {
        printf("Error when allocating memory in tensorsubAlloc");
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


#define TILE_DIM 16


__global__ void __matMulTiled(const float *A, const float *B, float *C, int M, int N, int K) {
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

float* matmul__ABT(int ax, int ay, int bx, int by, const float *A, const float *B, float *C) {
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

float* dotAlloc(cublasHandle_t* handle, float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {
    if (vectorSize1 != vectorSize2) {
        printf("Error: Incompatible shapes for dot product");
        return nullptr;
    }

    // allocate memory
    float* d_result = reserveMemoryOnDevice(1);

    if (d_result == nullptr) {
        printf("Error when allocating memory in dotAlloc");
        return nullptr;
    }

    // perform computation
    cublasStatus_t err = cublasSdot_v2(*handle, vectorSize1, d_vector1, 1, d_vector2, 1, d_result);

    if (err != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_result);
        std::cout << "matrix multiplication failed: " << std::string(cublasGetStatusString(err));
        return nullptr;
    }

    return d_result;
}

__global__ void __softmax_exp_sum(float* d_targetMemorySpace, float* d_vector, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float expx = 0.0f;
    if (i < size) {
        expx = expf(d_vector[i]);
        d_targetMemorySpace[i] = expx;
    }
    // Warp-level reduction using shuffle instructions
    float sum = expx;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // Each warp leader adds its partial sum to global sum
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&d_targetMemorySpace[size], sum);
    }
}

__global__ void __softmax_normalize(float* d_targetMemorySpace, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Normalize using the total sum stored at d_targetMemorySpace[size]
        d_targetMemorySpace[i] /= d_targetMemorySpace[size];
    }
}

float* softmaxAlloc(float* d_vector, unsigned int vectorSize) {
    // allocate required memory, add 1 to cache sum of e^xi 
    float* d_result = reserveMemoryOnDevice(vectorSize + 1);

    if (d_result == nullptr) {
        printf("Error: Softmax failed");
        return nullptr;
    }

    // initalize sum placeholder to zero
    CHECK_CUDA_ERROR(cudaMemset(d_result + vectorSize, 0, 1 * sizeof(float)));
    
    // compute block size
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(vectorSize);

    // Count bytes needed for shared memory
    size_t sharedMemorySize = BLOCK_REDUCTION_SIZE * sizeof(float);

    // launch kernel
    __softmax_exp_sum<<<blocksThreads.first, blocksThreads.second, sharedMemorySize, 0>>>(d_result, d_vector, vectorSize);
    __softmax_normalize<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_result, vectorSize);

    cudaError_t err = cudaGetLastError();
    
    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (err != cudaSuccess) {
        printf("Error when performing softmax");
        return nullptr;
    }

    // return pointer to result
    return d_result;
}

__global__ void __categoricalCrossEntropyLoss_exp_sum(float* d_targetMemorySpace, float* d_predicted, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        __shared__ float blockExpSum[BLOCK_REDUCTION_SIZE];

        // Initialize shared memory to 0
        if (threadIdx.x < BLOCK_REDUCTION_SIZE) {
            blockExpSum[threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Calculate exp(x) for softmax
        float expx = expf(d_predicted[i]);
        
        // Store exp(x) for later use
        d_targetMemorySpace[i] = expx;

        // Sum up exp values for softmax denominator
        atomicAdd(&blockExpSum[threadIdx.x / BLOCK_REDUCTION_LEFTOVER], expx);

        __syncthreads();

        if (threadIdx.x == 0) {
            float blockSum = 0;
            for (int j = 0; j < BLOCK_REDUCTION_SIZE; j++) {
                blockSum += blockExpSum[j];
            }
            atomicAdd(&d_targetMemorySpace[size], blockSum);
        }
    }
}

__global__ void __categoricalCrossEntropyLoss_final(float* d_targetMemorySpace, float* d_calcMem, float* d_actual, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        __shared__ float blockLossSum[BLOCK_REDUCTION_SIZE];

        // Initialize shared memory to 0
        if (threadIdx.x < BLOCK_REDUCTION_SIZE) {
            blockLossSum[threadIdx.x] = 0.0f;
        }

        __syncthreads();
        
        // Calculate softmax probability and cross entropy loss using the complete sum
        float softmax_prob = d_calcMem[i] / d_calcMem[size];
        float loss = -d_actual[i] * logf(fmaxf(softmax_prob, 1e-7f));
        
        // Sum up loss values
        atomicAdd(&blockLossSum[threadIdx.x / BLOCK_REDUCTION_LEFTOVER], loss);

        __syncthreads();

        if (threadIdx.x == 0) {
            float blockLoss = 0;
            for (int j = 0; j < BLOCK_REDUCTION_SIZE; j++) {
                blockLoss += blockLossSum[j];
            }
            atomicAdd(d_targetMemorySpace, blockLoss);
        }
    }
}

float* categoricalCrossEntropyLossAlloc(float* d_predicted, float* d_actual, unsigned int vectorSize) {
    // allocate memory for intermediate results and final loss
    // size + 1 for exp sum, + 1 for final loss
    float* d_calcMem = reserveMemoryOnDevice(vectorSize + 1);

    float* d_result = reserveMemoryOnDevice(1);


    if (d_calcMem == nullptr) {
        printf("Error: Categorical cross entropy loss failed - memory allocation failed\n");
        return nullptr;
    }

    // Initialize sum and loss to 0
    CHECK_CUDA_ERROR(cudaMemset(d_calcMem + vectorSize, 0, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_result, 0, sizeof(float)));
    
    // compute block size
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(vectorSize);

    // Count bytes needed for shared memory
    size_t sharedMemorySize = BLOCK_REDUCTION_SIZE * sizeof(float);

    // First kernel: compute exp values and their sum
    __categoricalCrossEntropyLoss_exp_sum<<<blocksThreads.first, blocksThreads.second, sharedMemorySize, 0>>>(d_calcMem, d_predicted, vectorSize);

    // Second kernel: compute cross entropy loss using complete sum
    __categoricalCrossEntropyLoss_final<<<blocksThreads.first, blocksThreads.second, sharedMemorySize, 0>>>(d_result, d_calcMem, d_actual, vectorSize);

    cudaError_t err = cudaGetLastError();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (err != cudaSuccess) {
        printf("Error when performing categorical cross entropy loss: %s\n", cudaGetErrorString(err));
        cudaFree(d_calcMem);
        return nullptr;
    }
    
    // Clean up intermediate results
    CHECK_CUDA_ERROR(cudaFree(d_calcMem));

    return d_result;
}