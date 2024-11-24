#include <iostream>
#include <stdexcept>
#include "Function.h"
#include "../cudaNN/Tensor.h"


class Operation: public Function {
    protected:
        Tensor* arg1;
        std::pair<unsigned int, unsigned int> shapeArg1;

        Tensor* arg2;
        std::pair<unsigned int, unsigned int> shapeArg2;

    public:
        virtual void backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape) = 0;
};