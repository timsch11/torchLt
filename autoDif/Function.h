#include <iostream>
#include <stdexcept>
#pragma once
#include "../cuda/Tensor.h"


class Function {
    protected:
        Tensor* arg1;
        std::pair<unsigned int, unsigned int> shapeArg1;

    public:
        Function(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1);
        Tensor* getArg1();
        std::pair<unsigned int, unsigned int> getShapeArg1();
        virtual Tensor* forward() = 0;
        virtual void backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape) = 0;
};