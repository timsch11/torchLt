#ifndef UTIL
#define UTIL


#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>


// set preferred block size
#define BLOCK_SIZE 256

// error checking
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)


std::pair<unsigned int, unsigned int> computeBlockThreadAllocation(unsigned int size);

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

#endif
