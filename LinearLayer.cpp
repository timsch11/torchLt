#include "LinearLayer.h"

class LinearLayer: public Layer {
    private:
        float* weights;
        float* bias;
    public:
        LinearLayer(int in_features, int out_features, float (*initialization_function) (int, int)) {

            // initialze bias to 0
            this->bias = new float[out_features];
        }

        void initialize_weights(float (*initialization_function) (int, int)) {
            // uses weight initialization function to initialize weights
            // to be implemented with cuda
        }
};