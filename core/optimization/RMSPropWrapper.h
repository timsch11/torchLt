#include "../Tensor.h"
#include "weightUpdate.cuh"


class RMSPropWrapper {
    private:
        Tensor &param;
        float* d_pastSquaredGradients;

        float alpha;
        float lr;
        float eps;

    public:
        RMSPropWrapper(Tensor &tensor, float lr, float alpha, float eps);
        void step(bool async);
        ~RMSPropWrapper();
};