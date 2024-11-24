#include "../autoDif/Function.h"

class Tensor {
    private:
        float* d_value;
        float* d_gradient;

        unsigned int shape_x;
        unsigned int shape_y;

        bool track_gradient;

        Function* precedingFunction;

    public:
        Tensor(float* _value, unsigned int _shape_x, unsigned int _shape_y, bool _track_gradient);

        // getter 
        float* getValue();
        float* getGradient();
        unsigned int getShapeX();
        unsigned int getShapeY();
        bool getTrackGradient();
        Funtion* getPrecedingFunction();

        // shape comparison
        bool sameShape(Tensor other);
        bool matVecMulCompatible(Tensor other);

        // operator overloading
        Tensor* operator+(Tensor &other);
        Tensor* operator*(Tensor &other);
        Tensor* operator%(Tensor &other);
};