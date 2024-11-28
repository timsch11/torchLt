#include <iostream>
#include <stdexcept>
#include "Function.h"
#include "../cuda/Tensor.h"


class Operation: public Function {
    protected:
        Tensor* arg1;
        std::pair<unsigned int, unsigned int> shapeArg1;

        Tensor* arg2;
        std::pair<unsigned int, unsigned int> shapeArg2;

    public:
        Operation(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1, Tensor* arg2, std::pair<unsigned int, unsigned int> shapeArg2);
        Tensor* getArg2();
        std::pair<unsigned int, unsigned int> getShapeArg2();
        virtual Tensor* forward() = 0;
        virtual void backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape) = 0;
};