#include "RMSPropWrapper.h"


RMSPropWrapper::RMSPropWrapper(Tensor &tensor, float lr, float alpha, float eps) : param(tensor), lr(lr), alpha(alpha), eps(eps) {

    // allocate memory for exponentially weighed average of past gradients
    this->d_pastSquaredGradients = reserveMemoryOnDevice(tensor.getSize());

    if (this->d_pastSquaredGradients == nullptr) {
        printf("Error: Memory allocation for past gradients failed");
        delete this;
    }
}

void RMSPropWrapper::step(bool async) {
    if (!this->param.isGradientSet()) {
        printf("Optimization failed: Gradient is not set");
        delete this;
    }

    cudaError_t err = rmspropUpdate(this->param.getValue(), this->d_pastSquaredGradients, this->param.getGradient(), this->param.getSize(), this->lr, this->alpha, this->eps, async);

    if (err != cudaSuccess) {
        std::cout << "Error, optimization step with momentum failed: " << std::string(cudaGetErrorString(err));
        delete this;
    }
}

RMSPropWrapper::~RMSPropWrapper() {
    printf("deleting...");
    if (this->d_pastSquaredGradients) {
        cudaFree(this->d_pastSquaredGradients);
    }
}