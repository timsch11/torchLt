#ifndef UTIL_GUARD
#define UTIL_GUARD

#include <iostream>
#include <stdexcept>


// set preferred block size
#define BLOCK_SIZE 256


std::pair<unsigned int, unsigned int> computeBlockThreadAllocation(unsigned int size) {
    unsigned int blockNum = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int threadNum = BLOCK_SIZE;
    return {blockNum, threadNum};
}


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
