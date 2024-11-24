#include <iostream>
#include <stdexcept>
#include "../cudaNN/Tensor.h"

class Function {
    protected:
        Tensor* arg1;
        std::pair<unsigned int, unsigned int> shapeArg1;

    public:
        Function(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1) {
            this->arg1 = arg1;
            this->shapeArg1 = shapeArg1;
        }

        virtual void backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape) = 0;

        Tensor* getArg1() {
            return this-> arg1;
        }

        std::pair<unsigned int, unsigned int> getShapeArg1() {
            return this->shapeArg1;
        }
};