#include "Relu.h"
#include "Function.h"
#include <iostream>
#include <stdexcept>
#include "../cuda/Tensor.h"

#include "../cuda/cudaNN.cu"
#include "../cuda/cudaDif.cu"
#include "../cuda/cudaMem.cu"



    Relu::Relu(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1): Function(arg1, shapeArg1) {
        if (shapeArg1.second) == 0) {
            throw std::runtime_error;
        }
    }

    Tensor* Relu::forward() {
        float* tensorValue = reserveMemoryOnDevice(this->getShapeArg1().first);
        relu(tensorValue, this->getArg1()->getValue(), this->getShapeArg1().first);
        Tensor t = Tensor(tensorValue, this->getShapeArg1().first, this->getShapeArg1().second, true, this);
        return &t;
    }
    void Relu::backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape) {
        if (!arg1->getTrackGradient()) {
            // think about it
            return;
        }
        reluGrad(this->arg1->getGradient(), this->arg1->getValue(), this->shapeArg1.first)
        this->arg1->getPrecedingFunction()->backward();
    }