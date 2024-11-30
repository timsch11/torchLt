#include "Relu.h"
#include "../cuda/cudaNN.cu"
#include <stdexcept>




    Relu::Relu(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1): Function(arg1, shapeArg1) {
        if (shapeArg1.second > 1) {
            throw std::runtime_error("Invalid shape for Relu");
        }
    }

    Tensor* Relu::forward() {
        float* tensorValue = reserveMemoryOnDevice(this->getShapeArg1().first);
        relu(tensorValue, this->getArg1()->getValue(), this->getShapeArg1().first);
        Tensor* t = new Tensor(tensorValue, this->getShapeArg1().first, this->getShapeArg1().second, true, this);
        return t;
    }
    void Relu::backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape) {
        if (!arg1->getTrackGradient()) {
            // think about it
            return;
        }
        //reluGrad(this->arg1->getGradient(), this->arg1->getValue(), this->shapeArg1.first);
        //this->arg1->getPrecedingFunction()->backward();
    }