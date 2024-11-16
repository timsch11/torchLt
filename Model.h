#include "Layer.h"

class Model {
    private:
        Layer* layers;

    public:
        void predict(float* input);
};