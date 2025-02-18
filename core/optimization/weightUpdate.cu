#include "weightUpdate.cuh"


__global__ void __momentumUpdate(float* d_targetMemorySpace, float* d_pastGradients, float* d_gradient, float lr, float beta, float ibeta) {
    // calculate index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // compute momentum and update d_pastGradients
    float mGrad = beta * d_pastGradients[i] + ibeta * d_gradient[i];
    d_pastGradients[i] = mGrad;

    // perform parameter update
    d_targetMemorySpace[i] -= lr*mGrad;
}

cudaError_t momentumUpdate(float* d_targetMemorySpace, float* d_pastGradients, float* d_gradient, unsigned int size, float lr, float beta, bool async) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    __momentumUpdate<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_pastGradients, d_gradient, lr, beta, 1 - beta);

    // check for errors
    cudaError_t err = cudaGetLastError();

    if (!async) {
        // synchronize before continuing with host code
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    return err;
}

__global__ void __rmspropUpdate(float* d_targetMemorySpace, float* d_pastSquaredGradients, float* d_gradient, float lr, float alpha, float ialpha, float eps) {
    // calculate index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // cache current gradient
    float d_gradi = d_gradient[i];

    // compute AdaptiveLearningRate and store it back in 
    float adl = alpha * d_pastSquaredGradients[i] + d_gradi * d_gradi * ialpha;
    d_pastSquaredGradients[i] = adl;
    d_targetMemorySpace[i] -= (lr * d_gradi) / sqrtf(adl + eps);
}

cudaError_t rmspropUpdate(float* d_targetMemorySpace, float* d_pastSquaredGradients, float* d_gradient, unsigned int size, float lr, float alpha, float eps, bool async) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    __rmspropUpdate<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_pastSquaredGradients, d_gradient, lr, alpha, 1 - alpha, eps);

    // check for errors
    cudaError_t err = cudaGetLastError();

    if (!async) {
        // synchronize before continuing with host code
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    return err;
}

__global__ void __adamUpdate(float* d_targetMemorySpace, float* d_pastGradients, float* d_pastSquaredGradients, float* d_gradient, float lr, float alpha, float ialpha, float momentum, float imomentum, float eps) {
    // calculate index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // cache current gradient value
    float d_gradi = d_gradient[i];

    // calculate AdaptiveLearningRate and store adl back in d_pastSquaredGradients
    float adl = alpha * d_pastSquaredGradients[i] + d_gradi * d_gradi * ialpha;
    d_pastSquaredGradients[i] = adl;

    // compute momentum accelerated gradient and store back in d_pastGradients
    float mGrad = momentum * d_pastGradients[i] + imomentum * d_gradi;
    d_pastGradients[i] = mGrad;

    // perform parameter update
    d_targetMemorySpace[i] -= (lr * mGrad) / sqrtf(adl + eps);
}

cudaError_t adamUpdate(float* d_targetMemorySpace, float* d_pastGradients, float* d_pastSquaredGradients, float* d_gradient, unsigned int size, float lr, float alpha, float momentum, float eps, bool async) {
    // compute optimal block/thread distribution
    std::pair<unsigned int, unsigned int> blocksThreads = computeBlockThreadAllocation(size);

    __adamUpdate<<<blocksThreads.first, blocksThreads.second>>>(d_targetMemorySpace, d_pastGradients, d_pastSquaredGradients, d_gradient, lr, alpha, 1 - alpha, momentum, 1 - momentum, eps);

    // check for errors
    cudaError_t err = cudaGetLastError();

    if (!async) {
        // synchronize before continuing with host code
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    return err;
}