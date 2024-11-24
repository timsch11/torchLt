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
        Operation(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1, Tensor* arg2, std::pair<unsigned int, unsigned int> shapeArg2): Function(arg1, shapeArg1) {
            this->arg2 = arg2;
            this->shapeArg2 = shapeArg2;
        }

        Tensor* getArg2() {
            return this-> arg2;
        }

        std::pair<unsigned int, unsigned int> getShapeArg2() {
            return this->shapeArg2;
        }

        virtual void backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape) = 0;
};