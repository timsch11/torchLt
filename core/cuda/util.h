#ifndef UTIL
#define UTIL


#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>


// set preferred block size
#define BLOCK_SIZE 512


std::pair<unsigned int, unsigned int> computeBlockThreadAllocation(unsigned int size);


/*Thanks to @Infatoshi for this error handling template, saved me some time when debugging: https://github.com/Infatoshi/cuda-course/blob/9bce6b12e6373c2bdd73563d410c390b9ae694c8/05_Writing_your_First_Kernels/05%20Streams/01_stream_basics.cu#L4*/

// error checking
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

#endif
