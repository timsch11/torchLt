#include "Tensor.h"
#include <stdexcept>
#include "cudaNN.cu"


// only covers vectors and matrices
// prefix d_ marks values residing on gpu memory

// initalize leaf
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient) {
    // shape_x is #rows and shape_y is #columns, 0 = no actual row/column, vector has ONE column!!!

    // check for zero configuration
    if (_shape.first == 0 || _shape.second == 0) {
        throw std::runtime_error("Cannot initialize zero tensor");
    }
            
    this->d_value = _d_value;
    this->shape = _shape;
    this->track_gradient = _track_gradient;

    this->leaf = true;

    if (track_gradient) {
        d_gradient = reserveMemoryOnDevice(_shape.first * _shape.second);
    }
}

// initalize result of single tensor operation
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1): Tensor(_d_value, _shape, _track_gradient) {
    
    this-> gradFunction = _gradFunction;
    this->d_funcArg1 = _d_funcArg1;
    this->shapeFuncArg1 = _shapeFuncArg1;

    this->leaf = false;
}

// initalize result of dual tensor operation
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2): Tensor(_d_value, _shape, _track_gradient, _gradFunction, _d_funcArg1, _shapeFuncArg1) {
    this->d_funcArg2 = _d_funcArg2;
    this->shapeFuncArg2 = _shapeFuncArg2;
}

// basic functions

float* Tensor::getValue() {
    return this->d_value;
}

float* Tensor::getGradient() {
    return this->d_gradient;
}

unsigned int Tensor::getShapeX() {
    return this->shape.first;
}

unsigned int Tensor::getShapeY() {
    return this->shape.second;
}

std::pair<unsigned int, unsigned int> Tensor::getShape() {
    return this->shape;
}

// returns number of entries of this tensor (product of shapes with respect to each dimension)
unsigned int Tensor::getSize() {
    return this->getShape().first * this->getShape().second;
}

bool Tensor::getTrackGradient() {
    return this->track_gradient;
}

Tensor* Tensor::getArg1() {
    return this->d_funcArg1;
}

Tensor* Tensor::getArg2() {
    return this->d_funcArg2;
}

std::pair<unsigned int, unsigned int> Tensor::getShapeArg1() {
    return this->shapeFuncArg1;
}

std::pair<unsigned int, unsigned int> Tensor::getShapeArg2() {
    return this->shapeFuncArg2;
}

bool Tensor::isLeaf() {
    return this->leaf;
}
        
bool Tensor::isGradientSet() {
    return this->gradientSet;
}

void Tensor::changeGradientSet(bool _gradientSet) {
    this->gradientSet = _gradientSet;
}

void Tensor::backward() {
    if (this->getTrackGradient()) {
        this->gradFunction(this);
    }
}

bool Tensor::sameShape(Tensor other) {
    // returns true if tensors are of same shape
    return (this->getShapeX() == other.getShapeX()) && (this->getShapeY() == other.getShapeY());
}

bool Tensor::matVecMulCompatible(Tensor other) {
    // returns true if thisTensor (matrix) x otherTensor (vector) can be performed
    return (this->getShapeY() == other.getShapeX()) && (other.getShapeY() == 1);
}

static void additionGradient(Tensor* currentTensor) {

}

// adds tensor values up and stores result in new Tensor, returns pointer to Tensor that holds result of addition if shapes match, otherwise prints error message and returns nullpointer
Tensor* Tensor::operator+(Tensor &other) {
    float* d_result = vecaddAlloc(this->getValue(), this->getSize(), other.getValue(), other.getSize());
    return new Tensor(d_result, this->getShape(), true, additionGradient, this, this->getShape(), &other, other.getShape());
}

static void subtractionGradient(Tensor* currentTensor) {

}

Tensor* Tensor::operator-(Tensor &other) {
    float* d_result = vecsubAlloc(this->getValue(), this->getSize(), other.getValue(), other.getSize());
    return new Tensor(d_result, this->getShape(), true, subtractionGradient, this, this->getShape(), &other, other.getShape());
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

static void reluGradient(Tensor* currentTensor) {
    Tensor* tensorGrad = currentTensor->getArg1();
    std::pair<unsigned int, unsigned int> shape = tensorGrad->getShape();
    reluGrad(tensorGrad->getGradient(), tensorGrad->getValue(), shape.first * shape.second);
    tensorGrad->changeGradientSet(true);
}

Tensor* Tensor::relu() {
    unsigned int size = this->getShapeX() * this->getShapeY();
    float* d_tensorValue = reluAlloc(this->getValue(), size);
    return new Tensor(d_tensorValue, this->getShape(), true, reluGradient, this, this->getShape());
}


int main() {
    float* mem = zeros(5);
    Tensor* t1 = new Tensor(mem, {5, 1}, true);
    float* mem2 = zeros(5);
    Tensor* t2 = new Tensor(mem2, {5, 1}, true);
    Tensor* t3 = *t1 - *t2;
    // t2->backward();
    float host_result[5];
    cudaMemcpy(host_result, t3->getValue(), 5 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<5; i++) {
        std::cout << host_result[i];
    }

    /*float h_bias[3] = {1.0f, -2.0f, 3.0f};
    float *bias;
    cudaMalloc(&bias, 3 * sizeof(float));
    cudaMemcpy(bias, h_bias, 3 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor t1 = Tensor(bias, 3, 1, false, nullptr);
    Tensor* t2 = t1.relu();

    cudaMemcpy(h_bias, t2->getValue(), 3 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<3; i++) {
        std::cout << h_bias[i];
    }

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
    
    return 0;*/
}