#include "../Tensor.h"
#include "weightUpdate.cuh"


class MomentumWrapper {
    private:
        Tensor &param;
        float* d_pastGradients;

        float beta;
        float lr;

    public:
        MomentumWrapper(Tensor &tensor, float lr, float beta);
        void step(bool async);
        ~MomentumWrapper();
};