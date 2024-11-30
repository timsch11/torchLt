#pragma once
#include <utility>

std::pair<unsigned int, unsigned int> computeBlockThreadAllocation(unsigned int size);

template<typename T>
void check(T err, const char* const func, const char* const file, const int line);

