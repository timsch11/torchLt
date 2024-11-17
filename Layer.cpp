#include "Layer.h"

class Layer {
    private:
        int in_features;
        int out_features;
        float* last_input;

    public:
        Layer(int in_features, int out_features) {
            this->in_features = in_features;
            this->out_features = out_features;
        }

        void forward(float* input, float* output) {

        }

        void backward(float* input, float* output) {

        }
};