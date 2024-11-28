#include "Operation.h"
#include <iostream>
#include <stdexcept>
#include "Function.h"
#include "../cuda/Tensor.h"


    Operation::Operation(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1, Tensor* arg2, std::pair<unsigned int, unsigned int> shapeArg2): Function(arg1, shapeArg1) {
        this->arg2 = arg2;
        this->shapeArg2 = shapeArg2;
    }

    Tensor* Operation::getArg2() {
        return this-> arg2;
    }

    std::pair<unsigned int, unsigned int> Operation::getShapeArg2() {
        return this->shapeArg2;
    }