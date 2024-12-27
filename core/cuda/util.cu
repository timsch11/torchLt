#include "util.h"


std::pair<unsigned int, unsigned int> computeBlockThreadAllocation(unsigned int size) {
    unsigned int blockNum = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int threadNum = BLOCK_SIZE;
    return {blockNum, threadNum};
}
