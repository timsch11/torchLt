#include "LinearLayer.h"
#include "../cuda/Tensor.h"
#include <cstdlib>

// define seed 
#define SEED 12345

// d_ pointers refer to device pointers (=exclusively reside on gpu)
// d<something>_d<something> variables refer to partial derivatives


    LinearLayer::LinearLayer(unsigned int in_features, unsigned int out_features) : Layer(in_features, out_features) {

        initialize_weights();
        // initialze bias to 0
        this->d_bias = zeros(out_features);
    }

     void LinearLayer::initialize_weights() {
        this->d_weights = xavier(this->in_features, this->out_features, SEED);
    }

    float* LinearLayer::forward(float* input) {
        return forward_layer(this->d_weights, this->d_bias, input, this->in_features, this->in_features, this->out_features);
    }

    float* LinearLayer::backward(float* dloss_dthis) {
        // backward pass of linear layer
        // to be implemented with cuda
        return dloss_dthis;
    }