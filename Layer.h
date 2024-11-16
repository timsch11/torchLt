class Layer {
    private:
        int in_features;
        int out_features;

    public:
        Layer(int in_features, int out_features);
        void forward(float* input, float* output);
        void backward(float* input, float* output);
        void update(float* input, float* output);
};