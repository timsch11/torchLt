#include "cublas_v2.h"

class Tensor {
    private:
        // tensor value and its gradient 

        float* d_value;
        float* d_gradient;

        // tensor shape

        std::pair<unsigned int, unsigned int> shape;

        bool track_gradient;  // whether gradient should be tracked for this tensor
        bool leaf;            // displays whether this tensor is a leaf on its computational graph (=if true there will be no gradient function or args)
        bool gradientSet;     // displays if gradient has been set

        // arguments for gradient function
        // if gradient function only needs one argument: d_funcArg2 = nullptr

        Tensor* d_funcArg1;  
        Tensor* d_funcArg2;

        // streams for synchronized/concurrent execution

        cudaStream_t* graphStream;  // used for operations inside of the computational graph that require synchronization
                                   // all nodes of a computational graph share the same graphStream
        cudaStream_t nodeStream;   // used for operations inside of the computational graph that do not require synchronization
                                   // every Tensor has a unique nodeStream

        // size of the computational graph lower than this node (including this node)

        unsigned int lowerGraphSize;

        // shapes of function arguments

        std::pair<unsigned int, unsigned int> shapeFuncArg1;
        std::pair<unsigned int, unsigned int> shapeFuncArg2;

        // gradient function of operation this node resulted from

        void (*gradFunction)(Tensor*);

    public:
        Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1);
        Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2);
        Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient);

        ~Tensor();

        // getter 
        float* getValue();
        float* getValueCPU();
        float* getGradient();

        unsigned int getShapeX();
        unsigned int getShapeY();
        std::pair<unsigned int, unsigned int> getShape();
        unsigned int getSize();

        Tensor* getArg1();
        Tensor* getArg2();

        std::pair<unsigned int, unsigned int> getShapeArg1();
        std::pair<unsigned int, unsigned int> getShapeArg2();

        cudaStream_t* getGraphStream();

        bool getTrackGradient();
        bool isLeaf();
        bool isGradientSet();

        unsigned int getLowerGraphSize();

        // setter, only sets the gradient if trackGradient evaluates to true
        void changeGradientSet(bool _gradientSet);
        void setGraphStreamForSubgraph(cudaStream_t* graphStream);  // sets the graphStream this and all preceding nodes

        // gradient propagation
        void backward();

        // update value without gradient tracking (i.e. for updating the weights of a neural network)
        // void addNoGrad() # TODO

        // shape comparison
        bool sameShape(Tensor other);
        bool matMulCompatible(Tensor other);

        // operator overloading
        Tensor* operator+(Tensor &other);
        Tensor* operator-(Tensor &other);
        Tensor* operator*(Tensor &other);
        Tensor* operator%(Tensor &other);

        // activation functions
        Tensor* relu();
};