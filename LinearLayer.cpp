#include "LinearLayer.h"
#include "cudaOperations.cu"
#include <cstdlib>

class LinearLayer: public Layer {
    private:
        float* weights;
        float* bias;

    public:
        LinearLayer(int in_features, int out_features, float (*initialization_function) (int, int)) : Layer(in_features, out_features) {

            initialize_weights(initialization_function);
            // initialze bias to 0
            this->bias = new float[out_features];
        }

        void initialize_weights(float (*initialization_function) (int, int)) {
            this->weights = (float*)malloc(this->in_features * this->out_features * sizeof(float));
            // initialize weights with initialization function using cuda
        }

        float* forward(float* input) {
            // forward pass of linear layer
            // to be implemented with cuda
            return input;
        }

        float* backward(float* dloss_dthis) {
            // backward pass of linear layer
            // to be implemented with cuda
            return dloss_dthis;
        }


};