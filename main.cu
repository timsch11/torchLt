#include "core/Factory.h"


int main() {
    init();
    
    float val1[6] = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
    float val2[6] = {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f};
    float val3[9] = {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 1.0f, 1.0f};

    // 3. Create tensors with error checking
    Tensor* t1 = nullptr;
    Tensor* t2 = nullptr;
    Tensor* t3 = nullptr;
    Tensor* t4 = nullptr;
    //Tensor* t5 = nullptr;

    try {
        t1 = createTensorFromHost(val1, {2, 2}, true);
        
        t2 = createTensorFromHost(val2, {2, 2}, true);
        t4 = createTensorFromHost(val3, {2, 2}, true);

        t3 = t1->matmul(*t2);

        //t4 = *t3 + *t5;

        // t3->printValue();

        // t4->backward();

        // t1->printGradient();

        t3->printValue();
        t2->printValue();
        t1->printValue();

        std::cout << "\n\n\n\n";
        
        // 4. Clean up in reverse order
        //delete t4;
        std::cout << *t3;
        delete t3;
        std::cout << *t2;
        delete t2;
        std::cout << *t1;
        delete t1;

    } catch (const std::runtime_error& e) {
        // 5. Clean up on error
        if (t3) delete t3;
        if (t2) delete t2;
        if (t1) delete t1;
        throw;
    }

    return 0;
}
