#include <iostream>
#include <stdexcept>
#include "Function.h"
#include "../cudaNN/Tensor.h"


class Relu: public Function {
    public:
        void backward(float* gradient, std::pair<unsigned int, unsigned int> gradientShape) {
            if (!arg1->getTrackGradient()) {
                // think about it
                return;
            }
            this->arg1->getValue
            // todo
        }
};