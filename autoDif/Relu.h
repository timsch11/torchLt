#include <iostream>
#include <stdexcept>
#include "Function.h"
#include "../cuda/Tensor.h"


class Relu: public Function {
    public:
        Relu(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1);
        Tensor* forward();
        void backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape);
};