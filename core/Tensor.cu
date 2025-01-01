#include "Tensor.h"


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

            throw std::runtime_error(std::string("cuBLAS initialization failed: ") + cublasGetStatusString(createStatus));
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

// initalize leaf with values from a custom function
Tensor::Tensor(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed, void(*initalization_function)(float*, unsigned int, unsigned int, int))
: Tensor(nullptr, _shape, _track_gradient) {
    this->d_value = reserveMemoryOnDevice( _shape.first * _shape.second);
    initalization_function(this->getValue(), _shape.first, _shape.second, seed);
}

// initalize leaf
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient) {
    // shape_x is #rows and shape_y is #columns, 0 = no actual row/column, vector has ONE column!!!

    // init cuBlas (if not done yet)
    init_cuBlas();

    // initalize reference counter
    refCount = new int(1);

    // keep track of active tensors
    activeTensors++;

    // check for zero configuration
    if (_shape.first == 0 || _shape.second == 0) {
        throw std::runtime_error("Cannot initialize zero tensor");
    }
            
    this->d_value = _d_value;
    this->shape = _shape;

    this->gradientSet = false;
    this->track_gradient = _track_gradient;

    this->leaf = true;
    this->lowerGraphSize = 1;

    this-> gradFunction = nullptr;
    this->d_funcArg1 = nullptr;
    this->d_funcArg2 = nullptr;

    // cudaStream_t graphStream
    // cudaStream_t nodeStream = 

    if (_track_gradient) {
        this->d_gradient = reserveMemoryOnDevice(_shape.first * _shape.second);
    }
}

// initalize result of single tensor operation
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1)
: Tensor(_d_value, _shape, _track_gradient) {
    
    this-> gradFunction = _gradFunction;
    this->d_funcArg1 = _d_funcArg1;
    this->shapeFuncArg1 = _shapeFuncArg1;

    this->lowerGraphSize += _d_funcArg1->getLowerGraphSize();

    this->graphStream = _d_funcArg1->getGraphStream();

    this->leaf = false;
}

// initalize result of dual tensor operation
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2)
: Tensor(_d_value, _shape, _track_gradient, _gradFunction, _d_funcArg1, _shapeFuncArg1) {
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

float* Tensor::getValue() const {
    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    return this->d_value;
}

float* Tensor::getValueCPU() const {
    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // initalize float array on host 
    float* host_value = (float*) malloc(this->getSize() * sizeof(float));

    // copy data, check for errors
    CHECK_CUDA_ERROR(cudaMemcpy(host_value, this->getValue(), this->getSize() * sizeof(float), cudaMemcpyDeviceToHost));
    
    return host_value;
}

float* Tensor::getGradient() const {
    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return this->d_gradient;
}

float* Tensor::getGradientCPU() const {
    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // initalize float array on host 
    float* host_gradient = (float*) malloc(this->getSize() * sizeof(float));

    // copy data, check for errors
    CHECK_CUDA_ERROR(cudaMemcpy(host_gradient, this->getGradient(), this->getSize() * sizeof(float), cudaMemcpyDeviceToHost));
    
    return host_gradient;
}

unsigned int Tensor::getShapeX() const {
    return this->shape.first;
}

unsigned int Tensor::getShapeY() const {
    return this->shape.second;
}

std::pair<unsigned int, unsigned int> Tensor::getShape() const {
    return this->shape;
}

// returns number of entries of this tensor (product of shapes with respect to each dimension)
unsigned int Tensor::getSize() const {
    return this->getShape().first * this->getShape().second;
}

bool Tensor::getTrackGradient() const {
    return this->track_gradient;
}

Tensor* Tensor::getArg1() const {
    return this->d_funcArg1;
}

Tensor* Tensor::getArg2() const {
    return this->d_funcArg2;
}

std::pair<unsigned int, unsigned int> Tensor::getShapeArg1() const {
    return this->shapeFuncArg1;
}

std::pair<unsigned int, unsigned int> Tensor::getShapeArg2() const {
    return this->shapeFuncArg2;
}

cudaStream_t* Tensor::getGraphStream() const {
    return this->graphStream;
}

unsigned int Tensor::getLowerGraphSize() const {
    return this->lowerGraphSize;
}

bool Tensor::isLeaf() const {
    return this->leaf;
}
        
bool Tensor::isGradientSet() const {
    return this->gradientSet;
}

void Tensor::changeGradientSet(bool _gradientSet) {
    this->gradientSet = _gradientSet;
}

void Tensor::addReference() {
    (*refCount)++;
}

void Tensor::removeReference() {
    (*refCount)--;
    if (*refCount <= 0) {
        delete this;
    }
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
    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // skip if either gradient tracking is disabled or there are no preceding calculations
    if (this->getTrackGradient() && !this->isLeaf()) {
        this->gradFunction(this);

        // recursively calculate gradients of preceding operations
        if (this->getArg1() != nullptr) {
            this->getArg1()->backward();
        }

        if (this->getArg2() != nullptr) {
            this->getArg2()->backward();
        }
    }
}

bool Tensor::sameShape(Tensor other) const {
    // returns true if tensors are of same shape
    return (this->getShapeX() == other.getShapeX()) && (this->getShapeY() == other.getShapeY());
}

bool Tensor::matMulCompatible(Tensor other) const {
    // returns true if matrices are compatible for matmul
    return this->getShapeY() == other.getShapeX();
}

/**
 * @brief Computes the gradient for the addition operation in a computational graph.
 *
 * This function calculates the gradient of the current tensor with respect to its arguments.
 * If the gradient for the current tensor is already set, it duplicates this gradient to the
 * gradients of the arguments. Otherwise, it initializes the gradients of the arguments to 1.0.
 *
 * @param currentTensor Pointer to the current tensor whose gradient is to be computed.
 *
 * The function performs the following steps:
 * 1. Retrieves the arguments (arg1 and arg2) of the current tensor.
 * 2. Checks if the gradient for the current tensor is already set.
 *    - If set, duplicates the gradient to the gradients of arg1 and arg2 if they track gradients.
 *    - If not set, initializes the gradients of arg1 and arg2 to 1.0 if they track gradients.
 */
static void additionGradient(Tensor* currentTensor) {
    // cache args
    Tensor* arg1 = currentTensor->getArg1();
    Tensor* arg2 = currentTensor->getArg2();

    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // copy gradient of this Tensor (and multply by I, neglected)
    if (currentTensor->isGradientSet()) {

        // the gradient of the last computation in this computational graph wrt this Tensor
        float* partialGradient = currentTensor->getGradient();

        if (arg1->getTrackGradient()) {
            cudaMemDup(partialGradient, arg1->getGradient(), arg1->getSize(), false);
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            cudaMemDup(partialGradient, arg2->getGradient(), arg1->getSize(), false);
            arg2->changeGradientSet(true);
        }
    } else {

        if (arg1->getTrackGradient()) {
            constants(arg1->getGradient(), arg1->getSize(), 1.0f);
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            constants(arg2->getGradient(), arg1->getSize(), 1.0f);
            arg2->changeGradientSet(true);
        }
    }
}

// adds tensor values up and stores result in new Tensor, returns pointer to Tensor that holds result of addition if shapes match, otherwise prints error message and returns nullpointer
Tensor* Tensor::add(Tensor &other) {
    // increment reference count of results args (dependencies)
    this->addReference();
    other.addReference();

    float* d_result = tensoraddAlloc(this->getValue(), this->getSize(), other.getValue(), other.getSize());
    return new Tensor(d_result, this->getShape(), true, &additionGradient, this, this->getShape(), &other, other.getShape());
}

Tensor* Tensor::operator+(Tensor &other) {
    return this->add(other);
}

/**
 * @brief Computes the gradient for the subtraction operation in a computational graph.
 *
 * This function calculates the gradient of the current tensor with respect to its arguments
 * (arg1 and arg2) and updates their gradients accordingly. If the gradient for the current
 * tensor is already set, it duplicates this gradient to the arguments' gradients. If the
 * gradient is not set, it initializes the arguments' gradients to a constant value of 1.0.
 *
 * @param currentTensor Pointer to the current tensor whose gradient is being computed.
 */
static void subtractionGradient(Tensor* currentTensor) {
    // cache args
    Tensor* arg1 = currentTensor->getArg1();
    Tensor* arg2 = currentTensor->getArg2();

    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // copy gradient of this Tensor (and multply by I, neglected)
    if (currentTensor->isGradientSet()) {

        // the gradient of the last computation in this computational graph wrt this Tensor
        float* partialGradient = currentTensor->getGradient();

        if (arg1->getTrackGradient()) {
            cudaMemDup(partialGradient, arg1->getGradient(), arg1->getSize(), false);
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            cudaMemDup(partialGradient, arg2->getGradient(), arg1->getSize(), false);
            arg2->changeGradientSet(true);
        }
    } else {
        if (arg1->getTrackGradient()) {
            constants(arg1->getGradient(), arg1->getSize(), 1.0f);
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            constants(arg2->getGradient(), arg1->getSize(), 1.0f);
            arg2->changeGradientSet(true);
        }
    }
}

Tensor* Tensor::sub(Tensor &other) {
    // increment reference count of results args (dependencies)
    this->addReference();
    other.addReference();

    float* d_result = tensorsubAlloc(this->getValue(), this->getSize(), other.getValue(), other.getSize());
    return new Tensor(d_result, this->getShape(), true, &subtractionGradient, this, this->getShape(), &other, other.getShape());
}

Tensor* Tensor::operator-(Tensor &other) {
    return this->sub(other);
}

/**
 * @brief Computes the gradient of a matrix multiplication (C = A * B) wrt to A and B
 * @param Pointer to the current tensor whose gradient is being computed
 */
static void multiplicationGradient(Tensor* currentTensor) {
    // set scalars for sgemm call
    float alpha = 1.0f, beta = 0.0f;

    // cache tensors
    Tensor* A = currentTensor->getArg1();
    Tensor* B = currentTensor->getArg2();

    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // cache shapes
    int m = A->getShapeX(), n = A->getShapeY(), p = B->getShapeY();

    // gradOut_currTensor = gradient of output of this computational graph wrt to this Tensor
    float* gradOut_currTensor = nullptr;

    if (!currentTensor->isGradientSet()) {
        /* this tensors gradient is not set => we started to differentiate from here so no need to apply the chain rule (saves a multiplication with Identity matrix) */
        // dA = I * B^T = B^T
        if (A->getTrackGradient()) {
            cudaMemDup(B->getValue(), A->getGradient(), B->getSize(), true);  // size of B = size of B^T (size=#entries)
            A->changeGradientSet(true);
        }

        // dB = A^T * I = A^T
        if (B->getTrackGradient()) {
            cudaMemDup(A->getValue(), B->getGradient(), A->getSize(), true);
            A->changeGradientSet(true);
        }

    } else {
        // gradOut_currTensor is gradient of current tensor
        gradOut_currTensor = currentTensor->getGradient();

        // dA = gradOut_currTensor * B^T
        if (A->getTrackGradient()) {
            cublasSgemm_v2(
                *handle, CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, p,
                &alpha,
                gradOut_currTensor, m,
                B->getValue(), B->getShapeX(),
                &beta,
                A->getGradient(), m
            );
            A->changeGradientSet(true);
        }

        // synchronize before continuing with host code
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // dB = A^T * gradOut_currTensor
        if (B->getTrackGradient()) {
            cublasSgemm_v2(
                *handle, CUBLAS_OP_T, CUBLAS_OP_N,
                n, p, m,
                &alpha,
                A->getValue(), m,
                gradOut_currTensor, m,
                &beta,
                B->getGradient(), n
            );
            B->changeGradientSet(true);
        }
        // synchronize before continuing with host code
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}

Tensor* Tensor::matmul(Tensor &other) {

    // increment reference count of results args (dependencies)
    this->addReference();
    other.addReference();

    if (this->matMulCompatible(other)) {

        try {
            float* d_resultMatrix = matmulAlloc(handle, this->getShapeX(), this->getShapeY(), other.getShapeX(), other.getShapeY(), this->getValue(), other.getValue());
            
            // Create new tensor with its own memory space
            return new Tensor(d_resultMatrix, 
                         {this->getShapeX(), other.getShapeY()},
                         true, 
                         multiplicationGradient,
                         this,
                         this->getShape(),
                         &other,
                         other.getShape());
        } catch (...) {
            std::cout << "error in matmul";
            throw;
        }
    }
    throw std::runtime_error("incompatible shapes for matrix multiplication");
}

Tensor* Tensor::operator*(Tensor &other) {
    return matmul(other);
}

/**
 * @brief Computes and stores the derivatives of the hadamard product operation
 * 
 * Sets gradient for arg1 and arg2 of the current Tensor (if they track their gradient)
 * Uses the chain rule to calculate the gradient
 * @param currentTensor The Tensor this function is called for
 */
static void hadamardGradient(Tensor* currentTensor) {
    // cache args
    Tensor* arg1 = currentTensor->getArg1();
    Tensor* arg2 = currentTensor->getArg2();

    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // if a gradient is set the gradient calculations needs to factor in the gradient wrt to this tensor (chain rule)
    if (currentTensor->isGradientSet()) {
        // the gradient of the last computation in this computational graph wrt this Tensor
        float* partialGradient = currentTensor->getGradient();

        // calculate gradient, multiply with partialGradient, if gradient is tracked at all
        if (arg1->getTrackGradient()) {
            hadamard(arg1->getGradient(), arg2->getValue(), partialGradient, arg1->getShape());
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            hadamard(arg2->getGradient(), arg1->getValue(), partialGradient, arg1->getShape());
            arg2->changeGradientSet(true);
        }

    } else {
        if (arg1->getTrackGradient()) {
            cudaMemDup(arg2->getValue(), arg1->getGradient(), arg1->getSize(), false);
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            cudaMemDup(arg1->getValue(), arg2->getGradient(), arg1->getSize(), false);
            arg2->changeGradientSet(true);
        }
    }
}

Tensor* Tensor::hadamardProduct(Tensor &other) {
    // increment reference count of results args (dependencies)
    this->addReference();
    other.addReference();

    // error checking is done is hadamardAlloc
    float* d_result = hadamardAlloc(this->getValue(), this->getShape(), other.getValue(), other.getShape());
    
    // return new Tensor
    return new Tensor(d_result, this->getShape(), true, hadamardGradient, this, this->getShape(), &other, other.getShape());
}

Tensor* Tensor::operator%(Tensor &other) {
    return hadamardProduct(other);
}

// activation functions

/**
 * @brief Computes the gradient of the ReLU activation function for the given tensor.
 *
 * This function calculates the gradient of the ReLU activation function and stores it
 * in the gradient attribute of the tensor. It also sets the gradientSet flag to true
 * to indicate that the gradient has been computed and stored.
 *
 * @param currentTensor Pointer to the tensor for which the ReLU gradient is to be computed.
 */
static void reluGradient(Tensor* currentTensor) {
    // cache arg1
    Tensor* tensorGrad = currentTensor->getArg1();

    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // compute gradient store in tensor's gradient attribute
    reluGrad(tensorGrad->getGradient(), tensorGrad->getValue(), tensorGrad->getSize());

    // set gradientSet flag
    tensorGrad->changeGradientSet(true);
}

Tensor* Tensor::relu() {
    // increment reference count of results args (dependencies)
    this->addReference();

    float* d_tensorValue = reluAlloc(this->getValue(), this->getSize());
    return new Tensor(d_tensorValue, this->getShape(), true, reluGradient, this, this->getShape());
}

/**
 * @brief Computes the gradient of the sigmoid activation function for the given tensor.
 *
 * This function calculates the gradient of the sigmoid activation function and stores it
 * in the gradient attribute of the tensor. It also sets the gradientSet flag to true
 * to indicate that the gradient has been computed and stored.
 *
 * @param currentTensor Pointer to the tensor for which the sigmoid gradient is to be computed.
 */
static void sigmoidGradient(Tensor* currentTensor) {
    // cache arg1
    Tensor* tensorGrad = currentTensor->getArg1();

    // make sure that concurrent operations finished before accessing value
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // compute gradient store in tensor's gradient attribute, pass currentTensor's value for simplifying computation
    sigmoidGrad(tensorGrad->getGradient(), currentTensor->getValue(), tensorGrad->getSize());

    // set gradientSet flag
    tensorGrad->changeGradientSet(true);
}

Tensor* Tensor::sigmoid() {
    // increment reference count of results args (dependencies)
    this->addReference();

    // compute sigmoid func store result in newly allocated memory section and return pointer to it
    float* d_sigmoidValue = sigmoidAlloc(this->getValue(), this->getSize());

    // return new tensor that has holds result of the sigmoid function as a value and corresponding shape and gradient function
    return new Tensor(d_sigmoidValue, this->getShape(), true, sigmoidGradient, this, this->getShape());
}

/**
 * @brief Computes the gradient of the tanh activation function for the given tensor.
 *
 * This function calculates the gradient of the tanh activation function and stores it
 * in the gradient attribute of the tensor. It also sets the gradientSet flag to true
 * to indicate that the gradient has been computed and stored.
 *
 * @param currentTensor Pointer to the tensor for which the tanh gradient is to be computed.
 */
static void tanhGradient(Tensor* currentTensor) {
    // cache arg1
    Tensor* tensorGrad = currentTensor->getArg1();

    // compute gradient store in tensor's gradient attribute, pass currentTensor's value for simplifying computation
    tanhGrad(tensorGrad->getGradient(), currentTensor->getValue(), tensorGrad->getSize());

    // set gradientSet flag
    tensorGrad->changeGradientSet(true);
}

Tensor* Tensor::tanh() {
    // increment reference count of results args (dependencies)
    this->addReference();

    // compute tanh func store result in newly allocated memory section and return pointer to it
    float* d_tanhValue = tanhAlloc(this->getValue(), this->getSize());

    // return new tensor that holds the result of the tanh function as a value and corresponding shape and gradient function
    return new Tensor(d_tanhValue, this->getShape(), true, tanhGradient, this, this->getShape());
}

// PRINTING

std::ostream& operator<<(std::ostream &s, const Tensor &tensor) {

    // example
    /* shape=(3, 3)
        1 2 3
        4 5 6
        7 8 9 
    */

    // add shape tuple to stream: first line
    s << std::endl << "shape=(" << tensor.getShapeX() << "," << tensor.getShapeY() << ")";

    // cache frequently used attributes as local variables
    unsigned int size = tensor.getSize();
    unsigned int rowSize = tensor.getShapeY();

    // load copy of value (residing on GPU) to CPU to properly access values
    float* val = tensor.getValueCPU();

    // print matrix in row major format
    for (unsigned int ind=0; ind<size; ind++) {

        // add linebreak before printing next row
        if (ind % rowSize == 0) {
            s << std::endl;
        }

        // append current entry
        s << val[ind] << " ";
    }

    // free CPU copy of values
    free(val);

    // return appended stream
    return s << std::endl;
}

void Tensor::printValue() const {
    std::cout << *this;
}

void Tensor::printGradient() const {
    // example
    /* shape=(3, 3)
        1 2 3
        4 5 6
        7 8 9 
    */

    // add shape tuple to stream: first line
    std::cout << std::endl << "shape=(" << this->getShapeX() << "," << this->getShapeY() << ")";

    // cache frequently used attributes as local variables
    unsigned int size = this->getSize();
    unsigned int rowSize = this->getShapeY();

    // load copy of value (residing on GPU) to CPU to properly access values
    float* grad = this->getGradientCPU();

    // print matrix in row major format
    for (unsigned int ind=0; ind<size; ind++) {

        // add linebreak before printing next row
        if (ind % rowSize == 0) {
            std::cout << std::endl;
        }

        // append current entry
        std::cout << grad[ind] << " ";
    }

    // free CPU copy of values
    free(grad);
}

// frees memory associated with this tensor and manages the cuBlas handle, be aware that this impacts the gradient calculation of preceding operations
Tensor::~Tensor() {

    std::cout << " \nGarbage collector\n";

    // Safety check for null refCount
    if (!refCount) {
        return;
    }

    std::cout << "garbage collector actually deletes";

    // Decrement reference count
    (*refCount)--;

    // if other Tensors that reference this Tensor still exist: abort
    if (*(this->refCount) > 0) {
        return;
    }

    // clean up
    if (this->refCount != nullptr) {
        delete refCount;
    }

    // decrement (and delete if no other references) reference counter of args
    if (this->getArg1() != nullptr) {
        this->getArg1()->removeReference();
        this->d_funcArg1 = nullptr;
    }
    if (this->getArg2() != nullptr) {
        this->getArg2()->removeReference();
        this->d_funcArg2 = nullptr;
    }

    // free occupied memory
    if (this->d_value) {
        CHECK_CUDA_ERROR(cudaFree(this->d_value));  // no error checking because when the shit already hit the fan this will already be freed
        this->d_value = nullptr;
    }

    if (this->getTrackGradient() && this->d_gradient) {
        CHECK_CUDA_ERROR(cudaFree(this->d_gradient));
        this->d_gradient = nullptr;
    }

    // decrement activeTensors counter and manages cuBlas handle
    activeTensors--;
    if (activeTensors == 0) {
        destroy_cuBlas();
    } 
}
