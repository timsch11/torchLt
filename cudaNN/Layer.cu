#include "Layer.h"
#include "../cuda/cudaOperations.cu"

class Layer {
    private:
        int in_features;
        int out_features;
        float* last_input;

    public:
        Layer(int in_features, int out_features) {
            this->in_features = in_features;
            this->out_features = out_features;
            this->last_input = reserveMemoryOnDevice(in_features);
        }

        virtual float* forward(float* input) = 0;

        virtual float* backward(float* dloss_dlayer) = 0;
};