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

/**
 * @brief CUDA kernel function to perform RMSProp update on the target memory space.
 *
 * @param d_targetMemorySpace Pointer to the target memory space where the updated weights will be stored.
 * @param d_pastSquaredGradients Pointer to the memory space storing the exponentially weighted average of squared past gradients.
 * @param d_gradient Pointer to the memory space storing the current gradient.
 * @param lr Learning rate for the update.
 * @param alpha Exponential weight factor for the update.
 * @param ialpha 1 - alpha
 */
__global__ void __rmspropUpdate(float* d_targetMemorySpace, float* d_pastSquaredGradients, float* d_gradient, float lr, float alpha, float ialpha, float eps);

/**
 * @brief Function to launch the CUDA kernel for momentum update.
 *
 * @param d_targetMemorySpace Pointer to the target memory space where the updated weights will be stored.
 * @param d_pastSquaredGradients Pointer to the memory space storing the exponentially weighted average of sqared past gradients.
 * @param d_gradient Pointer to the memory space storing the current gradient.
 * @param size Size of the memory spaces (number of elements).
 * @param lr Learning rate for the update.
 * @param alpha Exponential weight factor for the update.
 * @return cudaError_t CUDA error code indicating the success or failure of the kernel launch.
 * @note alpha is supposed to but not checked to be in [0, 1]
 */
cudaError_t rmspropUpdate(float* d_targetMemorySpace, float* d_pastSquaredGradients, float* d_gradient, unsigned int size, float lr, float alpha, float eps, bool async);

/** 
 * @brief CUDA kernel function to perform Adam update on the target memory space.
 *
 * @param d_targetMemorySpace Pointer to the target memory space where the updated weights will be stored.
 * @param d_pastGradients Pointer to the memory space storing the exponentially weighted average of past gradients.
 * @param d_pastSquaredGradients Pointer to the memory space storing the exponentially weighted average of squared past gradients.
 * @param d_gradient Pointer to the memory space storing the current gradient.
 * @param lr Learning rate for the update.
 * @param alpha Exponential weight factor for the RMSProp part.
 * @param ialpha 1 - alpha
 * @param momentum Exponential weight factor for the Momentum part
 * @param imomentum 1 - momentum
 * @param eps Epsilon to use to avoid division by zero 
 */
__global__ void __adamUpdate(float* d_targetMemorySpace, float* d_pastGradients, float* d_pastSquaredGradients, float* d_gradient, float lr, float alpha, float ialpha, float momentum, float imomentum, float eps);

/**
 * @brief Function to launch the CUDA kernel for Adam update.
 *
 * @param d_targetMemorySpace Pointer to the target memory space where the updated weights will be stored.
 * @param d_pastGradients Pointer to the memory space storing the exponentially weighted average of past gradients.
 * @param d_pastSquaredGradients Pointer to the memory space storing the exponentially weighted average of squared past gradients.
 * @param d_gradient Pointer to the memory space storing the current gradient.
 * @param size Size of param 
 * @param lr Learning rate for the update.
 * @param alpha Exponential weight factor for the RMSProp part.
 * @param ialpha 1 - alpha
 * @param momentum Exponential weight factor for the Momentum part
 * @param imomentum 1 - momentum
 * @param eps Epsilon to use to avoid division by zero 
 * @param async Whether to wait for kernel completion
 */
cudaError_t adamUpdate(float* d_targetMemorySpace, float* d_pastGradients, float* d_pastSquaredGradients, float* d_gradient, unsigned int size, float lr, float alpha, float momentum, float eps, bool async);