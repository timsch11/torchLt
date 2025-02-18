#include "AdamWrapper.h"


AdamWrapper::AdamWrapper(Tensor &tensor, float lr, float alpha, float momentum, float eps) : param(tensor), lr(lr), alpha(alpha), momentum(momentum), eps(eps) {

    // allocate memory for exponentially weighed average of past gradients (=Momentum)
    this->d_pastGradients = reserveMemoryOnDevice(tensor.getSize());

    if (this->d_pastGradients == nullptr) {
        printf("Error: Memory allocation for past gradients failed");
        delete this;
    }

    // allocate memory for exponentially weighed average of squared past gradients (=RMSProp)
    this->d_pastSquaredGradients = reserveMemoryOnDevice(tensor.getSize());

    if (this->d_pastSquaredGradients == nullptr) {
        printf("Error: Memory allocation for past squared gradients failed");
        delete this;
    }
}

void AdamWrapper::step(bool async) {
    if (!this->param.isGradientSet()) {
        printf("Optimization failed: Gradient is not set");
        delete this;
    }

    // update weights and d_pastGradients and d_pastSquaredGradients
    cudaError_t err = adamUpdate(this->param.getValue(), this->d_pastGradients, this->d_pastSquaredGradients, this->param.getGradient(), this->param.getSize(), this->lr, this->alpha, this->momentum, this->eps, async);

    if (err != cudaSuccess) {
        std::cout << "Error, Adam optimization step failed: " << std::string(cudaGetErrorString(err));
        delete this;
    }
}

AdamWrapper::~AdamWrapper() {
    printf("deleting...");
    
    // free Momentum collector
    if (this->d_pastGradients) {
        cudaFree(this->d_pastGradients);
    }

    // free learning rate collector
    if (this->d_pastSquaredGradients) {
        cudaFree(this->d_pastSquaredGradients);
    }
}