#include "../Tensor.h"
#include "weightUpdate.cuh"


class AdamWrapper {
    private:
        Tensor &param;

        float* d_pastGradients;
        float* d_pastSquaredGradients;

        float alpha;
        float momentum;
        float lr;
        float eps;

    public:
        AdamWrapper(Tensor &tensor, float lr, float alpha, float momentum, float eps);
        void step(bool async);
        ~AdamWrapper();
};