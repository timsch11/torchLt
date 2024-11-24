class Layer {
    protected:
        int in_features;
        int out_features;
        float* last_input;

    public:
        Layer(int in_features, int out_features);
        virtual float* forward(float* input) = 0;
        virtual float* backward(float* dloss_dlayer) = 0;
};