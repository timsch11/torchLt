#include "util.cu"
#include "curand_kernel.h"


// MEMORY INITIALIZATION FUNCTIONS

__global__ void initZero(float* d_memorySection) {
    d_memorySection[blockIdx.x * blockDim.x + threadIdx.x] = 0;
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

// allocates memory of 
float* reserveMemoryOnDevice(unsigned int size) {
    // declare pointer
    float* memoryAlloc;

    // reserve actual space in memory, add some padding for thread efficiency
    CHECK_CUDA_ERROR(cudaMalloc(&memoryAlloc, (size + (size % BLOCK_SIZE)) * sizeof(float)));

    // return pointer 
    return memoryAlloc;
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


int main() {
    // Allocate host memory for results
float* h_bias = new float[5];  // Host memory

// Device allocation
float* d_bias;
cudaMalloc(&d_bias, 5 * sizeof(float));

// Initialize with zeros
weight_init(d_bias, 5, 1, 10.75f, 1234567);

// Copy from device (zero_init) to device (d_bias)
// cudaMemcpy(d_bias, zero_init, 5 * sizeof(float), cudaMemcpyDeviceToDevice);

// Copy from device to host for printing
cudaMemcpy(h_bias, d_bias, 5 * sizeof(float), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

// Check for errors
cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    throw std::runtime_error("CUDA error");
}

// Print values from host memory
for (int i = 0; i < 5; i++) {
    std::cout << h_bias[i] << " ";
}

// Cleanup
delete[] h_bias;
cudaFree(d_bias);
}