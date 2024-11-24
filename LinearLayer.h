#include "Layer.h"

class LinearLayer : public Layer {
    private:
        float* d_weights;
        float* d_bias;
    public:
        LinearLayer(int in_features, int out_features);
        void initialize_weights();

        float* forward(float* input);
        float* backward(float* dloss_dlayer);
};