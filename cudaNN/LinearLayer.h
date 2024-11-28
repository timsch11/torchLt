#include "Layer.h"
#include "../cuda/Tensor.h"

class LinearLayer : public Layer {
    private:
        Tensor* weights;
        Tensor* bias;
    public:
        LinearLayer(unsigned int in_features, unsigned int out_features);
        void initialize_weights();

        float* forward(float* input);
        float* backward(float* dloss_dlayer);
};