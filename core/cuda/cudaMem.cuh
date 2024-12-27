#ifndef CUDA_MEM
#define CUDA_MEM


#include <cuda_runtime.h>

#include "curand_kernel.h"
#include "util.h"


/**
 * @brief returns pointer to allocated but uninitalized device float array of <size> (+ padding)
 * @param size size of memory section
 */
float* reserveMemoryOnDevice(unsigned int size);

/**
 * @brief returns pointer to newly created device array of <size> (+ padding) inialized with 0
 * @param size size of array to be initalized
 */
float* zeros(unsigned int size);

/**
 * @brief returns pointer to newly created device array of <size> (+ padding) inialized with <value>
 * @param size size of array to be initalized
 * @param value value to fill in
 */
float* constants(unsigned int size, float constant);

/**
 * @brief Fills already existing array with a constant value
 * @param size Size of array 
 * @param d_value Pointer to array to be filled with constants
 * @param constant Value to fill in
 */
void constants(float* d_value, unsigned int size, float constant);

/**
 * @brief Duplicates an array on device memory
 * @param d_source Pointer to source array (that should be duplicated)
 * @param d_destination Pointer to destination array (that should hold result)
 * @param size Size of array to be duplicated
 * @param transpose Whether destination array should be the transposed version of source
 */
void cudaMemDup(float* d_source, float* d_destination, unsigned int size, bool transpose = false);

/**
 * @brief fills tensor of specified size with values sampled from a scaled random normal distribution: N~(0, sqrt(<scalingFactor>))
 * @param d_targetMemorySpace pointer to (padded) memory section that is to be initalized
 * @param size size of tensor (=number of total elements)
 * @param scaling_factor variance of the normal distribution
 * @param seed determines the seed for the curand normal dist. function
 */
void weight_init(float* d_targetMemorySpace, unsigned int size, float scaling_factor, int seed);

#endif