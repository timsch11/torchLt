#include "Hadamard.h"
#include "../cuda/cudaNN.cu"


Hadamard::Hadamard(Tensor* arg1, std::pair<unsigned int, unsigned int> shapeArg1, Tensor* arg2, std::pair<unsigned int, unsigned int> shapeArg2): Operation(arg1, shapeArg1, arg2, shapeArg2) {}

Tensor* Hadamard::forward() {
    float* d_result = hadamardAlloc(this->arg1->getValue(), this->arg1->getShape(), this->arg2->getValue(), this->arg2->getShape());
    return Tensor(d_result, this->arg1->getShapeX(), this->arg1->getShapeY(), true, this);
}

void Hadamard::backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape) {

}