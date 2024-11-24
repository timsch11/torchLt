#include <iostream>


cudaError_t matvecmul(float* d_matix, unsigned int numRows, unsigned int numCols, float* d_vector, unsigned int vectorSize, float* d_targetMemorySpace) {
    if (numCols != vectorSize) {
        printf("vectors to be added have different shapes\n");
        return cudaErrorInvalidValue;
    }
    
}


int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
        std::cout << deviceProp.concurrentKernels;
    }
    return 0;
}