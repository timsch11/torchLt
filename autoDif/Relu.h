#include <iostream>
#include <stdexcept>
#include "Function.h"
#include "../cudaNN/Tensor.h"


class Relu: public Function {
    public:
        Tensor* forward();
        void backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape);
};