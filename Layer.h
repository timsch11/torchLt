class Layer {
    protected:
        int in_features;
        int out_features;
        float* last_input;

    public:
        Layer(int in_features, int out_features);
        float* forward(float* input);
        float* backward(float* input);
};