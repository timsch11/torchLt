#pragma once
#include "cuda_runtime.h"

__global__ void initZero(float* d_memorySection);
float* zeros(unsigned int size);
float* reserveMemoryOnDevice(unsigned int size);
