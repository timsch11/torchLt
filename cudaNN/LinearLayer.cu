#include "LinearLayer.h"
#include "cudaOperations.cu"
#include <cstdlib>

// define seed 
#define SEED 12345

// d_ pointers refer to device pointers (=exclusively reside on gpu)
// d<something>_d<something> variables refer to partial derivatives

class LinearLayer: public Layer {
    private:
        float* d_weights;
        float* d_bias;

    public:
        LinearLayer(int in_features, int out_features) : Layer(in_features, out_features) {

            initialize_weights();
            // initialze bias to 0
            this->d_bias = zeros(out_features);
        }

        void initialize_weights() {
            this->d_weights = xavier(this->in_features, this->out_features, SEED);
        }

        float* forward(float* input) {
            return forward_layer(this->d_weights, this->d_bias, input, this->in_features, this->in_features, this->out_features);
        }

        float* backward(float* dloss_dthis) {
            // backward pass of linear layer
            // to be implemented with cuda
            return dloss_dthis;
        }
};