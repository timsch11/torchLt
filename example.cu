#include "core/Factory.h"

// Entry point of the program
int main() {
    // Initialize the library or framework
    init();
    
    // Define some sample data for the tensors
    float val1[6] = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
    float val2[6] = {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f};
    float val3[6] = {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f};

    // Create pointers for the tensors, initialized to nullptr
    Tensor* t1 = nullptr;
    Tensor* t2 = nullptr;
    Tensor* t3 = nullptr;
    Tensor* t4 = nullptr;
    Tensor* t5 = nullptr;

    try {
        // Create tensors from the host data with shape {3, 2} and enable gradient tracking
        t1 = createTensorFromHost(val1, {3, 2}, true);
        t2 = createTensorFromHost(val2, {3, 2}, true);
        t5 = createTensorFromHost(val3, {3, 2}, true);

        // Perform tensor operations: subtraction and addition
        t3 = *t1 - *t2;  // t3 = t1 - t2
        t4 = *t3 + *t5;  // t4 = t1 - t2 + t5

        // Print the values of the resulting tensors
        t3->printValue();
        t4->printValue();

        // Perform backpropagation to compute gradients
        t4->backward();

        // Print the gradients of t4 with respect to t1, t2, and t5
        std::cout << "Gradient of t4 with respect to t1";
        t1->printGradient();

        std::cout << "Gradient of t4 with respect to t2";
        t2->printGradient();

        std::cout << "Gradient of t4 with respect to t5";
        t5->printGradient();
        
        // Clean up and free the allocated memory for tensors
        delete t5;
        delete t4;
        delete t3;
        delete t2;
        delete t1;

    } catch (const std::runtime_error& e) {
        // Clean up in case of an error
        if (t5) delete t5;
        if (t4) delete t4;
        if (t3) delete t3;
        if (t2) delete t2;
        if (t1) delete t1;
        throw;
    }

    return 0;
}