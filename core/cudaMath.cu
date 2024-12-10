#include <iostream>
#include <stdexcept>
#include <string>
#include "util.cu"


// MATH FUNCTIONS

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