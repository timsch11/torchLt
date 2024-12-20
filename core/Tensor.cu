#include "Tensor.h"
#include <stdexcept>
#include "cudaNN.cu"


cublasHandle_t* handle = nullptr;
unsigned int activeTensors = 0;

/**
 * @brief creates a global cuBlas handle if it does not already exist
 * @note throws error if creation failed
 * 
 */
void init_cuBlas() {
    if (handle == nullptr) {

        // allocate memory for handle object
        handle = new cublasHandle_t;

        // create handle, store status type
        cublasStatus_t createStatus = cublasCreate_v2(handle);

        // error handling
        if (createStatus != CUBLAS_STATUS_SUCCESS) {

            // free allocated memory and reset pointer
            delete handle;
            handle = nullptr;

            throw std::runtime_error("cuBLAS initialization failed" + (std::string) cublasGetStatusString(createStatus));
        }
    }
}

/**
 * @brief destroys the global cuBlas handle and frees associated memory
 * @note to run a cuBlas function you need to first call init_cuBlas again
 */
void destroy_cuBlas() {
    if (handle != nullptr) {

        // destroy handle, store status type
        cublasStatus_t destroyStatus = cublasDestroy_v2(*handle);

        // no error handling, only one error case: CUBLAS_STATUS_NOT_INITIALIZED (if condition ensures that this does not happen)

        // reset pointer to cuBlas handle
        delete handle;
        handle = nullptr;
    }
}

// initalize leaf
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient) {
    // shape_x is #rows and shape_y is #columns, 0 = no actual row/column, vector has ONE column!!!

    // init cuBlas (if not done yet)
    init_cuBlas();

    // keep track of active tensors
    activeTensors++;

    // check for zero configuration
    if (_shape.first == 0 || _shape.second == 0) {
        throw std::runtime_error("Cannot initialize zero tensor");
    }
            
    this->d_value = _d_value;
    this->shape = _shape;
    this->track_gradient = _track_gradient;

    this->leaf = true;
    this->lowerGraphSize = 1;

    this-> gradFunction = nullptr;
    this->d_funcArg1 = nullptr;
    this->d_funcArg2 = nullptr;

    // cudaStream_t graphStream
    // cudaStream_t nodeStream = 

    if (track_gradient) {
        d_gradient = reserveMemoryOnDevice(_shape.first * _shape.second);
    }
}

// initalize result of single tensor operation
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1): Tensor(_d_value, _shape, _track_gradient) {
    
    this-> gradFunction = _gradFunction;
    this->d_funcArg1 = _d_funcArg1;
    this->shapeFuncArg1 = _shapeFuncArg1;

    this->lowerGraphSize += _d_funcArg1->getLowerGraphSize();

    this->graphStream = _d_funcArg1->getGraphStream();

    this->leaf = false;
}

// initalize result of dual tensor operation
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2): Tensor(_d_value, _shape, _track_gradient, _gradFunction, _d_funcArg1, _shapeFuncArg1) {
    this->d_funcArg2 = _d_funcArg2;
    this->shapeFuncArg2 = _shapeFuncArg2;
    this->lowerGraphSize += _d_funcArg2->getLowerGraphSize();

    // merge graphStreams of subgraphs

    cudaStream_t* unifiedGraphStream;

    if (_d_funcArg1->getLowerGraphSize() >= _d_funcArg2->getLowerGraphSize()) {  // use graphStream of d_funcArg1
        unifiedGraphStream = _d_funcArg1->getGraphStream();

        // modify graphStreams of other subgraph
        _d_funcArg2->setGraphStreamForSubgraph(unifiedGraphStream);

    } else {  // use graphStream of d_funcArg2

        unifiedGraphStream = _d_funcArg2->getGraphStream();
        
        // modify graphStreams of other subgraph
        _d_funcArg1->setGraphStreamForSubgraph(unifiedGraphStream);
    }

    // update this node
    this->graphStream = unifiedGraphStream;
}

// basic functions

float* Tensor::getValue() {
    return this->d_value;
}

float* Tensor::getValueCPU() {
    // initalize float array on host 
    float* host_value = (float*) malloc(this->getSize() * sizeof(float));

    // copy data, check for errors
    CHECK_CUDA_ERROR(cudaMemcpy(host_value, this->getValue(), this->getSize() * sizeof(float), cudaMemcpyDeviceToHost));
    
    return host_value;
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

cudaStream_t* Tensor::getGraphStream() {
    return this->graphStream;
}

unsigned int Tensor::getLowerGraphSize() {
    return this->lowerGraphSize;
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

// updates graphStream for the subgraph and itself, requires initalized stream
void Tensor::setGraphStreamForSubgraph(cudaStream_t* _graphStream) {

    this->graphStream = _graphStream;

    // stop recursion if we reached a leaf
    if (this->isLeaf()) {
        return;
    }

    if (this->d_funcArg1 != nullptr) {
        // call setGraphStreamForSubgraph in d_funcArg1
        this->d_funcArg1->setGraphStreamForSubgraph(_graphStream);
    }

    if (this->d_funcArg2 != nullptr) {
        // call setGraphStreamForSubgraph in d_funcArg2
        this->d_funcArg2->setGraphStreamForSubgraph(_graphStream);
    }
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

bool Tensor::matMulCompatible(Tensor other) {
    // returns true if matrices are compatible for matmul
    return this->getShapeY() == other.getShapeX();
}

static void additionGradient(Tensor* currentTensor) {

}

// adds tensor values up and stores result in new Tensor, returns pointer to Tensor that holds result of addition if shapes match, otherwise prints error message and returns nullpointer
Tensor* Tensor::add(Tensor &other) {
    float* d_result = tensoraddAlloc(this->getValue(), this->getSize(), other.getValue(), other.getSize());
    return new Tensor(d_result, this->getShape(), true, additionGradient, this, this->getShape(), &other, other.getShape());
}

Tensor* Tensor::operator+(Tensor &other) {
    return this->add(other);
}

static void subtractionGradient(Tensor* currentTensor) {

}

Tensor* Tensor::sub(Tensor &other) {
    float* d_result = tensorsubAlloc(this->getValue(), this->getSize(), other.getValue(), other.getSize());
    return new Tensor(d_result, this->getShape(), true, subtractionGradient, this, this->getShape(), &other, other.getShape());
}

Tensor* Tensor::operator-(Tensor &other) {
    return this->sub(other);
}

static void multiplicationGradient(Tensor* currentTensor) {

}

Tensor* Tensor::matmul(Tensor &other) {
    if (matMulCompatible(other)) {

        // allocate memory for result
        float* d_result = reserveMemoryOnDevice(this->getShapeX() * other.getShapeY());

        // define some parameters
        float alpha = 1.0f;
        float beta = 0.0f;
        int tx = this->getShapeX();

        // matmul
        cublasStatus_t matmulStatus = cublasSgemm_v2(*handle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    other.getShapeY(),
                                                    this->getShapeX(),
                                                    other.getShapeX(),
                                                    &alpha,
                                                    other.getValue(),
                                                    other.getShapeY(),
                                                    this->getValue(),
                                                    this->getShapeY(),
                                                    &beta,
                                                    d_result,
                                                    other.getShapeY()
        );
        
        if (matmulStatus != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("matrix multiplication failed: " + (std::string) cublasGetStatusString(matmulStatus));
        }

        // return result as a new Tensor
        return new Tensor(d_result, {this->getShapeX(), other.getShapeY()}, true, multiplicationGradient, this, this->getShape(), &other, other.getShape());
    }
    throw std::runtime_error("incompatible shapes for matrix multiplication");
}

Tensor* Tensor::operator*(Tensor &other) {
    return matmul(other);
}

static void hadamardGradient(Tensor* currentTensor) {

}

Tensor* Tensor::hadamardProduct(Tensor &other) {
    // error checking is done is hadamardAlloc
    float* d_result = hadamardAlloc(this->getValue(), this->getShape(), other.getValue(), other.getShape());
    
    // return new Tensor
    return new Tensor(d_result, this->getShape(), true, hadamardGradient, this, this->getShape(), &other, other.getShape());
}

Tensor* Tensor::operator%(Tensor &other) {
    return hadamardProduct(other);
}

// activation functions

static void reluGradient(Tensor* currentTensor) {
    Tensor* tensorGrad = currentTensor->getArg1();
    std::pair<unsigned int, unsigned int> shape = tensorGrad->getShape();
    reluGrad(tensorGrad->getGradient(), tensorGrad->getValue(), shape.first * shape.second);
    tensorGrad->changeGradientSet(true);
}

Tensor* Tensor::relu() {
    float* d_tensorValue = reluAlloc(this->getValue(), this->getSize());
    return new Tensor(d_tensorValue, this->getShape(), true, reluGradient, this, this->getShape());
}

// frees memory associated with this tensor and manages the cuBlas handle, be aware that this impacts the gradient calculation of preceding operations
Tensor::~Tensor() {

    // free occupied memory
    cudaFree(d_value);
    cudaFree(d_gradient);

    // cuBlas handle
    activeTensors--;
    if (activeTensors == 0) {
        destroy_cuBlas();
    } 
}


int main() {
    float* mem = constants(4, 2);
    Tensor* t1 = new Tensor(mem, {2, 2}, true);
    float mem2[6] = {0.0f, -2.0f, 3.0f, -4.0f};
    float* d_mem2;
    cudaMalloc(&d_mem2, 6*sizeof(float));
    cudaMemcpy(d_mem2, &mem2, 6 * sizeof(float), cudaMemcpyHostToDevice);
    Tensor* t2 = new Tensor(d_mem2, {2, 2}, true);
    Tensor* t3 = t2->relu();
    // t2->backward();
    float host_result[6];
    cudaMemcpy(host_result, t3->getValue(), 6 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<6; i++) {
        std::cout << host_result[i] << " ";
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
