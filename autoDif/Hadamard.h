#include <iostream>
#include <stdexcept>
#include "Operation.h"
#include "../cuda/Tensor.h"


class Hadamard: public Operation {
    public:
        Hadamard(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1, Tensor* arg2, std::pair<unsigned int, unsigned int> shapeArg2);
        Tensor* forward();
        void backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape);
};