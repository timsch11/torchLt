#include "Layer.h"
#include "Activation.h"

class Activation : public Layer {
    public:
        Activation(int in_features);
        void forward(float* input, float* output);
        void backward(float* input, float* output);
        void update(float* input, float* output);
};