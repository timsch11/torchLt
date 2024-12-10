#include <iostream>
#include <stdexcept>
#include <string>
#include "util.cu"


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

// ADD NEW VECTOR SHAPE CONVENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/*float* matvecmul(float* mat, int matRows, int matCols, float* vec, int vecSize, float* targetMemorySpace) {
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
}*/

// adds one entry each
__global__ void addVecEntries(float* d_targetMemorySpace, float* d_vec1, float* d_vec2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[idx] = d_vec1[idx] + d_vec2[idx];
}

// adds the values of two vectors and stores the result in <targetMemorySpace>, Note: for efficiency reasons size of targetMemorySpace must be rounded up to a multiple of blocksize
cudaError_t vecadd(float* d_targetMemorySpace, float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {

    // check for vector compatibility
    if (vectorSize1 != vectorSize2) {
        printf("vectors to be added have different shapes\n");
        return cudaErrorInvalidValue;
    }

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(vectorSize1);
    
    addVecEntries<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_vector1, d_vector2);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    return cudaSuccess;
}

// adds one column each
__global__ void subtractVecEntries(float* d_targetMemorySpace, float* d_vec1, float* d_vec2) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[ind] = d_vec1[ind] - d_vec2[ind];
}

// subtracts the values of two vectors and stores the result in <targetMemorySpace>, Note: for efficiency reasons size of targetMemorySpace must be rounded up to a multiple of blocksize
cudaError_t vecsub(float* d_targetMemorySpace, float* d_vector1, unsigned int vectorSize1, float* d_vector2, unsigned int vectorSize2) {

    // check for vector compatibility
    if (vectorSize1 != vectorSize2) {
        printf("vectors to be subtracted have different shapes\n");
        return cudaErrorInvalidValue;
    }

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(vectorSize1);
    
    subtractVecEntries<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_vector1, d_vector2);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    return cudaSuccess;
}

__global__ void scaleEntries(float* d_targetMemorySpace, float* d_tensor, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[idx] = d_tensor[idx] * scalar;
}

// performs scalar multiplication on the elements of an arbitrarily shaped tensor
cudaError_t scaletensor(float* d_targetMemorySpace, float* d_tensor, unsigned int tensorSize, float scalar) {

    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(tensorSize);
    
    scaleEntries<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_tensor, scalar);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    return cudaSuccess;
}

__global__ void hadamard_kernel(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[ind] = d_tensor1[ind] * d_tensor2[ind];
}

// computes the hadamard product between tensor1 and tensor2 and stores the result in targetMemorySpace, the tensor shapes must obviously match
cudaError_t hadamard(float* d_targetMemorySpace, float* d_tensor1, float* d_tensor2, std::pair<unsigned int, unsigned int> shape) {

    // calculate num of fields
    unsigned int size = (shape.second == 0) ? shape.first : shape.first * shape.second;

    // calculate #Block and #Thread
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    
    // let kernel do its work
    hadamard_kernel<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(d_targetMemorySpace, d_tensor1, d_tensor2);

    // error checking
    CHECK_CUDA_ERROR(cudaGetLastError());

    return cudaSuccess;
}

__global__ void matmat_matmul(float* d_matrix_1, float* d_matrix_2) {

}

__global__ void matTmat_matmul(float* d_matrix_1, float* d_matrix_2) {

}

/*__global__ void matmatT_matmul(float* d_matrix_1, float* d_matrix_2, unsigned int matrix_1_amount_of_columns, unsigned int matrix_2_amount_of_rows_before_transpose, float* d_targetMemorySpace) {
    // declare shared memory
    __shared__ float s_vec[BLOCK_SIZE];

    // initalize register variables (scope=Thread)
    float r_sum = 0.0f;
    unsigned int matrix1_ind = ;  // -> first entry of m1 to be multiplied 
    unsigned int matrix2_ind = blockIdx.x * blockDim.x;  // -> first entry of m2 to be multiplied
    unsigned int target_ind = ; // -> entry in target matrix to be updated

    // initalize shared memory (scope=Block): each thread initalizes one enty
    s_vec[threadIdx.x] = d_matrix_2[matrix2_ind + threadIdx.x];  // coalescing memory access to row of matrix2 (consecutively stored in memory)
    __syncthreads();

    for (int i=)
}*/

__global__ void matTmatT_matmul(float* d_matrix_1, float* d_matrix_2) {

}

// Performs standard matrix multiplication, optionally transposes either one or both of the given matrices for multiplication. Stores result in zero-initialized memory section.
cudaError_t matmul(float* d_matrix_1, std::pair<unsigned int, unsigned int> matrix_1_shape, bool transpose_matrix_1, float* d_matrix_2, std::pair<unsigned int, unsigned int> matrix_2_shape, bool transpose_matrix_2, float* targetMemorySpace) {
    if (matrix_1_shape.second != matrix_2_shape.first) {
        throw std::runtime_error("invalid matrix shapes for matrix multiplication: (" + std::to_string(matrix_1_shape.first) + ", " + std::to_string(matrix_1_shape.second) + ") and (" + std::to_string(matrix_2_shape.first) + ", " + std::to_string(matrix_2_shape.second) + ")");
    }

    cudaEvent_t e1, e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);

    cudaStream_t s1;
    cudaStreamCreate(&s1);


    // multiplying, result goes to calc
    cudaEventRecord(e1, s1);
    // matmatT_matmul<<<, 0, s1>>>();
    cudaEventRecord(e2, s1);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, e1, e2);

    std::cout << milliseconds << "ms passed";

}



__global__ void reluGrad_kernel(float* targetMemorySpace, float* vector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector[i] > 0) {
        targetMemorySpace[i] = 1;
    } else {
        targetMemorySpace[i] = 0;
    }
}

// computes gradient of relu function of tensor
cudaError_t reluGrad(float* d_targetMemorySpace, float* d_vector, unsigned int size) {
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    reluGrad_kernel<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_vector);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    return cudaSuccess;
}

/*int main() {
    // Allocate host memory for results
float* h_bias = new float[256];  // Host memory

// Device allocation
float* d_bias = constants(25, 1);
// cudaMalloc(&d_bias, 5 * sizeof(float));

// Initialize with zeros


float* d_bias2 = constants(25, -2);

// Initialize with zeros
// weight_init(d_bias, 5, 1, 10.75f, 1234567);

float* d_target = zeros(25);

cudaEvent_t e1, e2;
cudaEventCreate(&e1);
cudaEventCreate(&e2);

cudaStream_t s1;
cudaStreamCreate(&s1);


cudaEventRecord(e1, s1);
matmatT_matmul<<<5, 5, 5*sizeof(float), s1>>>(d_bias, d_bias2, 5, d_target);
cudaEventRecord(e2, s1);
CHECK_CUDA_ERROR(cudaGetLastError());
cudaDeviceSynchronize();

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, e1, e2);

std::cout << milliseconds << "ms passed";

// Copy from device (zero_init) to device (d_bias)
// cudaMemcpy(d_bias, zero_init, 5 * sizeof(float), cudaMemcpyDeviceToDevice);

// Copy from device to host for printing
cudaMemcpy(h_bias, d_target, 256 * sizeof(float), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

// Check for errors
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA error");
}

// Print values from host memory
for (int i = 0; i < 256; i++) {
    std::cout << h_bias[i] << " ";
}

// Cleanup
delete[] h_bias;
cudaFree(d_bias);
cudaFree(d_bias2);
}*/