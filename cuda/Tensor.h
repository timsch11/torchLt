class Tensor {
    private:
        // tensor value and its gradient 
        float* d_value;
        float* d_gradient;

        // tensor shape
        std::pair<unsigned int, unsigned int> shape;

        bool track_gradient;  // whether gradient should be tracked for this tensor
        bool leaf;  // displays whether this tensor is a leaf on its computational graph (=if true there will be no gradient function or args)
        bool gradientSet;  // displays if gradient has been set

        // arguments for gradient function, if gradient function only needs one d_funcArg2 = nullptr
        Tensor* d_funcArg1;  
        Tensor* d_funcArg2;

        std::pair<unsigned int, unsigned int> shapeFuncArg1;
        std::pair<unsigned int, unsigned int> shapeFuncArg2;

        void (*gradFunction)(Tensor*);

    public:
        Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1);
        Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2);
        Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient);

        // getter 
        float* getValue();
        float* getGradient();
        unsigned int getShapeX();
        unsigned int getShapeY();
        Tensor* getArg1();
        Tensor* getArg2();
        std::pair<unsigned int, unsigned int> getShapeArg1();
        std::pair<unsigned int, unsigned int> getShapeArg2();
        std::pair<unsigned int, unsigned int> getShape();
        bool getTrackGradient();
        bool isLeaf();
        bool isGradientSet();

        // setter, only sets the gradient if trackGradient evaluates to true
        void setGradient(float* _d_grad);

        // gradient propagation
        void backward();

        // shape comparison
        bool sameShape(Tensor other);
        bool matVecMulCompatible(Tensor other);

        // operator overloading
        /*Tensor* operator+(Tensor &other);
        Tensor* operator-(Tensor &other);
        Tensor* operator*(Tensor &other);
        Tensor* operator%(Tensor &other);*/

        // activation functions
        Tensor* relu();
};