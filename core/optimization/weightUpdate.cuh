#include "../cuda/util.h"


/**
 * @brief CUDA kernel function to perform momentum update on the target memory space.
 *
 * @param d_targetMemorySpace Pointer to the target memory space where the updated weights will be stored.
 * @param d_pastGradients Pointer to the memory space storing the exponentially weighted average of past gradients.
 * @param d_gradient Pointer to the memory space storing the current gradient.
 * @param lr Learning rate for the update.
 * @param beta Momentum factor for the update.
 * @param ibeta Complement of the momentum factor (1 - beta).
 */
__global__ void __momentumUpdate(float* d_targetMemorySpace, float* d_pastGradients, float* d_gradient, float lr, float beta, float ibeta);

/**
 * @brief Function to launch the CUDA kernel for momentum update.
 *
 * @param d_targetMemorySpace Pointer to the target memory space where the updated weights will be stored.
 * @param d_pastGradients Pointer to the memory space storing the exponentially weighted average of past gradients.
 * @param d_gradient Pointer to the memory space storing the current gradient.
 * @param size Size of the memory spaces (number of elements).
 * @param lr Learning rate for the update.
 * @param beta Momentum factor for the update.
 * @return cudaError_t CUDA error code indicating the success or failure of the kernel launch.
 * @note beta is supposed to but not checked to be in [0, 1]
 */
cudaError_t momentumUpdate(float* d_targetMemorySpace, float* d_pastGradients, float* d_gradient, unsigned int size, float lr, float beta, bool async);