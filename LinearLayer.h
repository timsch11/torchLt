#include "Layer.h"

class LinearLayer : public Layer {
    private:
        float* weights;
        float* bias;
    public:
        LinearLayer(int in_features, int out_features, float (*initialization_function) (int, int));
        void initialize_weights(float (*initialization_function) (int, int));
};