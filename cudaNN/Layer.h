class Layer {
    protected:
        unsigned int in_features;
        unsigned int out_features;
        float* last_input;

    public:
        Layer(unsigned int in_features, unsigned int out_features);
        virtual float* forward(float* input) = 0;
        virtual float* backward(float* dloss_dlayer) = 0;
};