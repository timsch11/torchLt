#include <iostream>
#include <stdexcept>
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

__global__ void initZero(float* d_memorySection) {
    d_memorySection[blockIdx.x * blockDim.x + threadIdx.x] = -2.0f;
}

// init array on device with zeros
float* zeros(unsigned int size) {
    // returns a pointer to (first element of) an array (interpretation of dimension is up to the caller) of specified size filled with zeros; array lives in unified memory (on cpu and gpu)

    // calc block/thread allocation scheme
    std::pair<unsigned int, unsigned int> blockThreadAllocation = computeBlockThreadAllocation(size);

    // reserve memory
    float* d_memoryAllocation;
    CHECK_CUDA_ERROR(cudaMalloc(&d_memoryAllocation, blockThreadAllocation.first * blockThreadAllocation.second * sizeof(float)));

    // launch kernel
    initZero<<<blockThreadAllocation.first, blockThreadAllocation.second, 0, 0>>>(d_memoryAllocation);
    CHECK_CUDA_ERROR(cudaGetLastError());

    return d_memoryAllocation;
}


int main() {
    // Allocate host memory for results
float* h_bias = new float[256];  // Host memory

// Device allocation
float* d_bias = zeros(5);
// cudaMalloc(&d_bias, 5 * sizeof(float));

// Initialize with zeros


float* d_bias2 = zeros(5);

// Initialize with zeros
// weight_init(d_bias, 5, 1, 10.75f, 1234567);

vecsub(d_bias, d_bias, 5, d_bias2, 5);

// Copy from device (zero_init) to device (d_bias)
// cudaMemcpy(d_bias, zero_init, 5 * sizeof(float), cudaMemcpyDeviceToDevice);

// Copy from device to host for printing
cudaMemcpy(h_bias, d_bias, 256 * sizeof(float), cudaMemcpyDeviceToHost);
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
}