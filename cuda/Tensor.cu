#include "Tensor.h"
#include "cudaMem.cu"
#include "../autoDif/Function.h"


// only covers vectors and matrices
// prefix d_ marks values residing on gpu memory

    Tensor::Tensor(float* _value, unsigned int _shape_x, unsigned int _shape_y, bool _track_gradient, Function* _precedingFunction) {
        // shape_x is #rows and shape_y is #columns, 0 = no actual row/column

        // check for zero configuration
        if (_shape_x == 0) {
            throw std::runtime_error("Cannot initialize tensor that has 0 dimension");
        }
            
        this->d_value = _value;
        this->shape_x = _shape_x;
        this->shape_y = _shape_y;
        this->track_gradient = _track_gradient;
        this->precedingFunction = _precedingFunction;
        if (track_gradient) {
            d_gradient = reserveMemoryOnDevice(_shape_x * _shape_y);
        }
    }

    // basic functions

    float* Tensor::getValue() {
        return this->d_value;
    }

    float* Tensor::getGradient() {
        return this->d_gradient;
    }

    unsigned int Tensor::getShapeX() {
        return this->shape_x;
    }

    unsigned int Tensor::getShapeY() {
        return this->shape_y;
    }

    bool Tensor::getTrackGradient() {
        return this->track_gradient;
    }

    Function* Tensor::getPrecedingFunction() {
        return this->precedingFunction;
    }

    bool Tensor::sameShape(Tensor other) {
        // returns true if tensors are of same shape
        return (this->getShapeX() == other.getShapeX()) && (this->getShapeY() == other.getShapeY());
    }

    bool Tensor::matVecMulCompatible(Tensor other) {
        // returns true if thisTensor (matrix) x otherTensor (vector) can be performed
        return (this->getShapeY() == other.getShapeX()) && (other.getShapeY() == 0);
    }

    // operator overloading, Note: this class is specifically optimized for neural networks running on the gpu, therefore result of operation is stored in second tensor
    // => a + b updates value of b and returns pointer to tensor b

    Tensor* Tensor::operator+(Tensor &other) {
         // adds tensor values up and stores result in new Tensor, returns pointer to Tensor that holds result of addition if shapes match, otherwise prints error message and returns nullpointer
        // check if shapes match
        if (this->sameShape(other) && this->getShapeY() == 0) {
            /*CHECK_CUDA_ERROR(vecadd(this->getValue(), this->getShapeX(), other.getValue(), other.getShapeX(), other.getValue()));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            return &other;*/
        }
        // error handling if shapes do not match
        printf("Error: Tensors must have same shape and be vectors (shape_y = 0)\n");
        return nullptr;
    }

    Tensor* Tensor::operator-(Tensor &other) {
        return nullptr;
    }

    Tensor* Tensor::operator*(Tensor &other) {
        // Remark: only supports Matrix-Vector multiplication yet
        // Matrix-Vector multiplication
        if (matVecMulCompatible(other)) {
            
        }

        return nullptr;
    }

    Tensor* Tensor::operator%(Tensor &other) {
        // performs hadamard product
        return nullptr;
    }

    // activation functions
    // TODO

int main() {
    float h_bias[3] = {1.0f, 2.0f, 3.0f};
    float *bias;
    cudaMalloc(&bias, 3 * sizeof(float));
    cudaMemcpy(bias, h_bias, 3 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor t1 = Tensor(bias, 3, 0, false, nullptr);
    /*float *bias, *inp;
    cudaError_t err;
    
    err = cudaMalloc(&bias, 3 * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaMalloc(&inp, 3 * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFree(bias);
        return 1;
    }

    // Initialize with some values
    float h_bias[3] = {1.0f, 2.0f, 3.0f};
    float h_inp[3] = {4.0f, 5.0f, 6.0f};
    
    cudaMemcpy(bias, h_bias, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inp, h_inp, 3 * sizeof(float), cudaMemcpyHostToDevice);

    printf("Memory allocated and initialized\n");
    fflush(stdout);  // Force print

    Tensor t1 = Tensor(bias, 3, 0, false);
    Tensor t2 = Tensor(inp, 3, 0, false);
    
    printf("Tensors created\n");  // Debug print
    
    Tensor* t3 = t1 + t2;
    if (t3 == nullptr) {
        printf("Addition failed\n");
        cudaFree(bias);
        cudaFree(inp);
        return 1;
    }
    
    printf("Addition completed\n");  // Debug print
    
    float host_result[3];
    err = cudaMemcpy(host_result, t3->getValue(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(bias);
        cudaFree(inp);
        return 1;
    }
    
    printf("Results:\n");  // Debug print
    for (int i = 0; i < 3; i++) {
        printf("%f ", host_result[i]);
    }
    printf("\n");
    
    cudaFree(bias);
    cudaFree(inp);
    
    return 0;
    */
}