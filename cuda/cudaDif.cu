//#include "cudaMath.cu"
#include "util.cu"


__global__ void reluGrad_kernel(float* targetMemorySpace, float* vector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (vector[i] > 0) {
        targetMemorySpace[i] = 1;
    } else {
        targetMemorySpace[i] = 0;
    }
}

cudaError_t reluGrad(float* targetMemorySpace, float* vector, unsigned int size) {
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);
    reluGrad_kernel<<<blocksThreads.first, blocksThreads.second, 0, 0>>>(targetMemorySpace, vector);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return cudaSuccess;
}


/*int main() {
    float h_bias[3] = {1.0f, -2.0f, 3.0f};
    float *bias;
    cudaMalloc(&bias, 3 * sizeof(float));
    cudaMemcpy(bias, h_bias, 3 * sizeof(float), cudaMemcpyHostToDevice);

    reluGrad(bias, bias, 3);

    cudaMemcpy(h_bias, bias, 3*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<3; i++) {
        std::cout << h_bias[i] << " ";
    }
}*/