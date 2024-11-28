#include "Layer.h"
#include "../cuda/Tensor.h"


    Layer::Layer(unsigned int in_features, unsigned int out_features) {
        this->in_features = in_features;
        this->out_features = out_features;
        this->last_input = reserveMemoryOnDevice(in_features);
    }