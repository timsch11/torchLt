// CUDA implementations for computing derivatives 
#include "cudaDif.cuh"

// ReLU gradient computation using element-wise operations
__global__ void __reluGrad(float* d_targetMemorySpace, float* d_vector, float* d_droot_dthis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_vector[i] > 0) {
        d_targetMemorySpace[i] = d_droot_dthis[i];
    } else {
        d_targetMemorySpace[i] = 0;
    }
}

__global__ void __reluGrad(float* d_targetMemorySpace, float* d_vector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_vector[i] > 0) {
        d_targetMemorySpace[i] = 1;
    } else {
        d_targetMemorySpace[i] = 0;
    }
}

cudaError_t reluGrad(float* d_targetMemorySpace, float* d_vector, float* d_droot_dthis, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    if (d_droot_dthis == nullptr) {
        // execute computation
        __reluGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_vector);
    } else {
        // execute computation with prior gradient
        __reluGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_vector, d_droot_dthis);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __sigmoidGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = d_tensor[i];
    d_targetMemorySpace[i] = val * (1 - val) * d_droot_dthis[i];
}

__global__ void __sigmoidGrad(float* d_targetMemorySpace, float* d_tensor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = d_tensor[i];
    d_targetMemorySpace[i] = val * (1 - val);
}

cudaError_t sigmoidGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    if (d_droot_dthis == nullptr) {
        // execute computation
        __sigmoidGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor);
    } else {
        // execute computation with prior gradient
        __sigmoidGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor, d_droot_dthis);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __tanhGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = d_tensor[i];
    d_targetMemorySpace[i] = (1 - (val*val)) * d_droot_dthis[i];
}

__global__ void __tanhGrad(float* d_targetMemorySpace, float* d_tensor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = d_tensor[i];
    d_targetMemorySpace[i] = 1 - (val*val);
}

cudaError_t tanhGrad(float* d_targetMemorySpace, float* d_tensor, float* d_droot_dthis, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    if (d_droot_dthis == nullptr) {
        // execute computation
        __tanhGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor);
    } else {
        // execute computation with prior gradient
        __tanhGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_tensor, d_droot_dthis);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

// Loss function gradient computations
__global__ void __l2LossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual, float* d_droot_dthis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // result of L2 loss is scalar -> d_droot_dthis is just a single value
    d_targetMemorySpace[i] = 2 * *d_droot_dthis * (d_predicted[i] - d_actual[i]);
}

__global__ void __l2LossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[i] = 2 * (d_predicted[i] - d_actual[i]);
}

cudaError_t l2LossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual, float* d_droot_dthis, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    if (d_droot_dthis == nullptr) {
        // execute computation
        __l2LossGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_predicted, d_actual);
    } else {
        // execute computation with prior gradient
        __l2LossGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_predicted, d_actual, d_droot_dthis);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __crossEntropyLossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual, float* d_droot_dthis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // result of loss is scalar -> d_droot_dthis is just a single value
    d_targetMemorySpace[i] = *d_droot_dthis * (d_predicted[i] - d_actual[i]);
}

__global__ void __crossEntropyLossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_targetMemorySpace[i] = d_predicted[i] - d_actual[i];
}

cudaError_t crossEntropyLossGrad(float* d_targetMemorySpace, float* d_predicted, float* d_actual, float* d_droot_dthis, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    if (d_droot_dthis == nullptr) {
        // execute computation
        __crossEntropyLossGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_predicted, d_actual);
    } else {
        // execute computation with prior gradient
        __crossEntropyLossGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_predicted, d_actual, d_droot_dthis);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

__global__ void __softmaxGrad(float* d_targetMemorySpace, float* d_softmax_output, float* d_droot_dthis) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Si = d_softmax_output[i];
    
    // Compute sum for Jacobian-vector product
    __shared__ float sum;
    if (threadIdx.x == 0) {
        sum = 0.0f;
    }
    __syncthreads();
    
    atomicAdd(&sum, Si * d_droot_dthis[i]);
    __syncthreads();
    
    // Compute final gradient: Si * (dL/dSi - sum)
    d_targetMemorySpace[i] = Si * (d_droot_dthis[i] - sum);
}

__global__ void __softmaxGrad(float* d_targetMemorySpace, float* d_softmax_output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Si = d_softmax_output[i];
    d_targetMemorySpace[i] = Si * (1.0f - Si);
}

cudaError_t softmaxGrad(float* d_targetMemorySpace, float* d_softmax_output, float* d_droot_dthis, unsigned int size) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    if (d_droot_dthis == nullptr) {
        // execute computation without chain rule
        __softmaxGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_softmax_output);
    } else {
        // execute computation with chain rule
        __softmaxGrad<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_softmax_output, d_droot_dthis);
    }

    // check for errors
    cudaError_t err = cudaGetLastError();

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return err;
}

