#include "Function.h"
#include <iostream>
#include <stdexcept>
#include "../cuda/Tensor.h"

    Function::Function(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1) {
        this->arg1 = arg1;
        this->shapeArg1 = shapeArg1;
    }

    Tensor* Function::getArg1() {
        return this-> arg1;
    }

    std::pair<unsigned int, unsigned int> Function::getShapeArg1() {
        return this->shapeArg1;
    }