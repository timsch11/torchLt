#include "MomentumWrapper.h"


MomentumWrapper::MomentumWrapper(Tensor &tensor, float lr, float beta) : param(tensor), lr(lr), beta(beta) {

    // allocate memory for exponentially weighed average of past gradients
    this->d_pastGradients = reserveMemoryOnDevice(tensor.getSize());

    if (this->d_pastGradients == nullptr) {
        printf("Error: Memory allocation for past gradients failed");
        delete this;
    }
}

void MomentumWrapper::step(bool async) {
    if (!this->param.isGradientSet()) {
        printf("Optimization failed: Gradient is not set");
        delete this;
    }

    cudaError_t err = momentumUpdate(this->param.getValue(), this->d_pastGradients, this->param.getGradient(), this->param.getSize(), this->lr, this->beta, async);

    if (err != cudaSuccess) {
        std::cout << "Error, optimization step with momentum failed: " << std::string(cudaGetErrorString(err));
        delete this;
    }
}

MomentumWrapper::~MomentumWrapper() {
    if (this->d_pastGradients) {
        cudaFree(this->d_pastGradients);
    }
}