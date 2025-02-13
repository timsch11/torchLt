#include "cudaMath.cuh"


// MATH FUNCTIONS


__global__ void __addTensorEntries(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[idx] = d_tensor1[idx] + d_tensor2[idx];
}

cudaError_t tensoradd(float* d_targetMemorySpace, float* d_tensor1, unsigned int tensorSize1, float* d_tensor2, unsigned int tensorSize2) {

    // check for vector compatibility
    if (tensorSize1 != tensorSize2) {
        printf("tensors to be added have different shapes\n");
        return cudaErrorInvalidValue;
    }

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(tensorSize1);
    
    __addTensorEntries<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor1, d_tensor2);

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __subtractVecEntries(float* d_targetMemorySpace, float* d_vec1, float* d_vec2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[i] = d_vec1[i] - d_vec2[i];
}

cudaError_t vecsub(float* d_targetMemorySpace, float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {

    // check for vector compatibility
    if (vectorSize1 != vectorSize2) {
        printf("vectors to be subtracted have different shapes\n");
        return cudaErrorInvalidValue;
    }

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(vectorSize1);
    
    __subtractVecEntries<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_vector1, d_vector2);

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __scaleEntries(float* d_targetMemorySpace, float* d_tensor, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[idx] = d_tensor[idx] * scalar;
}

cudaError_t scaletensor(float* d_targetMemorySpace, float* d_tensor, unsigned int tensorSize, float scalar) {

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(tensorSize);
    
    __scaleEntries<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor, scalar);

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __hadamard(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[i] = d_tensor1[i] * d_tensor2[i];
}

cudaError_t hadamard(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2, std::pair<unsigned int, unsigned int> shape) {

    // calculate num of fields
    unsigned int size = (shape.second == 0) ? shape.first : shape.first * shape.second;

    // calculate #Block and #Thread
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    
    // let kernel do its work
    __hadamard<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor1, d_tensor2);

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // return error
    return err;
}

__global__ void __scaledSubtraction(float* d_targetMemorySpace, float* d_vec1, float* d_vec2, float scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[i] = d_vec1[i] - (scalar * d_vec2[i]);
}

cudaError_t scaledSubtraction(float* d_targetMemorySpace, float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2, float scalar, bool async) {

    // check for vector compatibility
    if (vectorSize1 != vectorSize2) {
        printf("vectors to be subtracted have different shapes\n");
        return cudaErrorInvalidValue;
    }

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(vectorSize1);
    
    __scaledSubtraction<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_vector1, d_vector2, scalar);

    if (!async) {
        // wait for completion
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    return err;
}

void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

cublasStatus_t gemm(cublasLtHandle_t* handle, const float* d_A, const float* d_B, float* d_C, int m, int n, int k, cublasOperation_t opA, cublasOperation_t opB) {

    // Create matrix descriptors
    cublasLtMatrixLayout_t matA, matB, matC;

    cublasLtOrder_t row_major_format = CUBLASLT_ORDER_ROW;
    
    bool transA = opA == CUBLAS_OP_T;
    bool transB = opB == CUBLAS_OP_T;

    // For A matrix
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &matA, CUDA_R_32F, transA ? k : m, transA ? m : k, transA ? m : k)); // swapped ld = cols,rows,rows

    checkCublasStatus( cublasLtMatrixLayoutSetAttribute(
        matA, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_format, sizeof( row_major_format ) ) );

    // For B matrix
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &matB, CUDA_R_32F, transB ? n : k, transB ? k : n, transB ? k : n)); // swapped ld = cols,rows,rows
    
    checkCublasStatus( cublasLtMatrixLayoutSetAttribute(
        matB, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_format, sizeof( row_major_format ) ) );

    // For C matrix
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &matC, CUDA_R_32F, m, n, n)); // swapped ld

    checkCublasStatus( cublasLtMatrixLayoutSetAttribute(
        matC, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_format, sizeof( row_major_format ) ) );

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

    
    //std::cout << "1";

    // Scale factors
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform matrix multiplication
    checkCublasStatus(cublasLtMatmul(
        *handle,
        operationDesc,
        &alpha,
        d_A,
        matA,
        d_B,
        matB,
        &beta,
        d_C,
        matC,
        d_C,
        matC,
        nullptr,
        nullptr,
        0,
        0));

    cublasLtMatrixLayoutDestroy(matA);
    cublasLtMatrixLayoutDestroy(matB);
    cublasLtMatrixLayoutDestroy(matC);
    cublasLtMatmulDescDestroy(operationDesc);

    return CUBLAS_STATUS_SUCCESS;
}

float* gemmA(cublasLtHandle_t* handle, const float* d_A, const float* d_B, int m, int n, int k, cublasOperation_t opA, cublasOperation_t opB) {

    // Allocate device memory
    float* d_C = reserveMemoryOnDevice(m * n);

    // error check
    if (d_C == nullptr) {
        printf("Error: Memory allocation for gemm failed");
        cudaFree(d_C);
        exit(EXIT_FAILURE);
    }

    // Create matrix descriptors
    cublasLtMatrixLayout_t matA, matB, matC;

    cublasLtOrder_t row_major_format = CUBLASLT_ORDER_ROW;

    bool transA = opA == CUBLAS_OP_T;
    bool transB = opB == CUBLAS_OP_T;

    // For A matrix
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &matA, CUDA_R_32F, transA ? k : m, transA ? m : k, transA ? m : k)); // swapped ld = cols,rows,rows

    checkCublasStatus( cublasLtMatrixLayoutSetAttribute(
        matA, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_format, sizeof( row_major_format ) ) );

    // For B matrix
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &matB, CUDA_R_32F, transB ? n : k, transB ? k : n, transB ? k : n)); // swapped ld = cols,rows,rows
    
    checkCublasStatus( cublasLtMatrixLayoutSetAttribute(
        matB, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_format, sizeof( row_major_format ) ) );

    // For C matrix
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &matC, CUDA_R_32F, m, n, n)); // swapped ld

    checkCublasStatus( cublasLtMatrixLayoutSetAttribute(
        matC, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major_format, sizeof(row_major_format)));


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
    float beta = 0.0f;

    // Perform matrix multiplication
    checkCublasStatus(cublasLtMatmul(
        *handle,
        operationDesc,
        &alpha,
        d_A,
        matA,
        d_B,
        matB,
        &beta,
        d_C,
        matC,
        d_C,
        matC,
        nullptr,
        nullptr,
        0,
        0));

    // wait for completion
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Cleanup
    cublasLtMatrixLayoutDestroy(matC);
    cublasLtMatmulDescDestroy(operationDesc);

    return d_C;
}
