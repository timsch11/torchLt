#include "Tensor.h"


// ##########################
// initalize global variables
// ##########################

cublasHandle_t* handle = nullptr;       // cublas handle
cublasLtHandle_t* handleLt = nullptr;   // cublasLt handle:

unsigned int activeTensors = 0;         // Tensor counter (used to keep track of when to initalize a new cublas handle and when to delete)


// ###########################################
// cublas/cublasLt handle lifecycle management
// ###########################################

// ### Creation
/**
 * @brief creates a global cuBlas handle if it does not already exist
 * @return returns either CUBLAS_STATUS_SUCCESS or error from initalization of either cublas or cublaslt handle creation
 * 
 */
cublasStatus_t init_cuBlas() {
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

            std::cout << "cuBLAS initialization failed: " << cublasGetStatusString(createStatus);
            return createStatus;
        }

        if (handleLt == nullptr) {

            // allocate memory for handle object
            handleLt = new cublasLtHandle_t;

            // create handle, store status type
            cublasStatus_t ltcreateStatus = cublasLtCreate(handleLt);

            // error handling
            if (ltcreateStatus != CUBLAS_STATUS_SUCCESS) {

                // free allocated memory and reset pointer
                delete handleLt;
                handleLt = nullptr;

                std::cout << "cuBLAS initialization failed: " << cublasGetStatusString(ltcreateStatus);

                return ltcreateStatus;
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ### Deletion
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

        // do the same for the cublasLt handle
        cublasLtDestroy(*handleLt);

        delete handleLt;
        handleLt = nullptr;
    }
}


// ###################
// Tensor constructors
// ###################

// initalize leaf with values from a custom function
Tensor::Tensor(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed, cudaError_t(*initalization_function)(float*, unsigned int, unsigned int, int))
: Tensor(nullptr, _shape, _track_gradient) {
    this->d_value = reserveMemoryOnDevice( _shape.first * _shape.second);

    // error check
    this->handleError(this->d_value, "Error: Memory allocation failed");

    // fill with values from initalization function
    this->handleError(initalization_function(this->d_value, _shape.second, _shape.first, seed), "Error: Tensor initalization with values from custom function failed");

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// initalize with a constant value (no gradient tracking of previous operation)
Tensor::Tensor(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, float constant) 
: Tensor(nullptr, _shape, _track_gradient) {
    unsigned int size = _shape.first * _shape.second;
    this->d_value = reserveMemoryOnDevice(size);

    // error check
    this->handleError(this->d_value, "Error: Memory allocation failed");

    // fill with constant
    this->handleError(constants(this->d_value, size, constant), "Error: Tensor initalization with constant values failed");
}

// initalize leaf
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient): 
    d_value(_d_value), shape(_shape), gradientSet(false), track_gradient(_track_gradient), leaf(true), 
    gradFunction(nullptr), d_funcArg1(nullptr), d_funcArg2(nullptr), d_funcArg3(nullptr) {
    // shape_x is #rows and shape_y is #columns, 0 = no actual row/column, vector has ONE column!!!

    // init cuBlas (if not done yet)
    this->handleError(init_cuBlas(), "Error: cuBlas initalization failed");

    // initalize reference counter
    refCount = new int(1);

    // keep track of active tensors
    activeTensors++;

    // check for zero configuration
    if (_shape.first == 0 || _shape.second == 0) {
        printf("Error: Cannot initialize zero tensor: exit\n");
        delete this;
        exit(EXIT_FAILURE);
    }
    
    if (_track_gradient) {
        this->d_gradient = reserveMemoryOnDevice(_shape.first * _shape.second);

        // check error
        this->handleError(this->d_gradient, "Error: Memory allocation for gradient failed");
    }
}

// initalize result of single tensor operation (one arg)
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1)
: Tensor(_d_value, _shape, _track_gradient) {
    
    this->gradFunction = _gradFunction;
    this->d_funcArg1 = _d_funcArg1;
    this->shapeFuncArg1 = _shapeFuncArg1;

    this->leaf = false;
}

// initalize result of dual tensor operation (two args)
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2)
: Tensor(_d_value, _shape, _track_gradient, _gradFunction, _d_funcArg1, _shapeFuncArg1) {
    this->d_funcArg2 = _d_funcArg2;
    this->shapeFuncArg2 = _shapeFuncArg2;
}

// initalize result of triple tensor operation (three args =sfpass)
Tensor::Tensor(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2, Tensor* _d_funcArg3, std::pair<unsigned int, unsigned int> _shapeFuncArg3)
: Tensor(_d_value, _shape, _track_gradient, _gradFunction, _d_funcArg1, _shapeFuncArg1, _d_funcArg2, _shapeFuncArg2) {
    this->d_funcArg3 = _d_funcArg3;
    this->shapeFuncArg3 = _shapeFuncArg3;
}

// ##############
// Getter methods
// ##############

float* Tensor::getValue() const {
    return this->d_value;
}

float* Tensor::getValueCPU() const {
    // synchronize before accessing data
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // validate device pointer
    if (this->getValue() == nullptr) {
        this->handleError(nullptr, "Error: Device pointer is null");
        return nullptr;
    }

    // initalize float array on host 
    float* host_value = (float*) malloc(this->getSize() * sizeof(float));

    this->handleError(host_value, "Error: Memory allocation on host failed");

    // copy data, check for errors
    this->handleError(cudaMemcpy(host_value, this->getValue(), this->getSize() * sizeof(float), cudaMemcpyDeviceToHost), "Error: Memcpy from device to host failed");
    
    return host_value;
}

float* Tensor::getGradient() const {
    return this->d_gradient;
}

float* Tensor::getGradientCPU() const {
    // synchronize before accessing data
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // initalize float array on host 
    float* host_gradient = (float*) malloc(this->getSize() * sizeof(float));

    this->handleError(host_gradient, "Error: Memory allocation on host failed");

    // copy data, check for errors
    this->handleError(cudaMemcpy(host_gradient, this->getGradient(), this->getSize() * sizeof(float), cudaMemcpyDeviceToHost), "Error: Memcpy from device to host failed");
    
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

void Tensor::enableGradientTracking() {
    this->track_gradient = true;
}

void Tensor::disableGradientTracking() {
    this->track_gradient = false;
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

Tensor* Tensor::getArg3() const {
    return this->d_funcArg3;
}

std::pair<unsigned int, unsigned int> Tensor::getShapeArg1() const {
    return this->shapeFuncArg1;
}

std::pair<unsigned int, unsigned int> Tensor::getShapeArg2() const {
    return this->shapeFuncArg2;
}

std::pair<unsigned int, unsigned int> Tensor::getShapeArg3() const {
    return this->shapeFuncArg3;
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

// ###############################
// Custom garbage collection utils
// ###############################

void Tensor::removeReference() {
    (*refCount)--;
    if (*refCount <= 0) {
        delete this;
    }
}

void Tensor::addReference() {
    (*refCount)++;
}

int Tensor::getReferenceCount() {
    return *(this->refCount);
}

// ###############################
// row/column accesses and slicing
// ###############################

Tensor* Tensor::getRows(unsigned int fromRow, unsigned int toRow) {
    // check bounds
    if (fromRow > toRow || fromRow > this->getShapeX() || toRow > this->getShapeX()) {
        printf("Error: row indexing is wrong");
        delete this;
        return nullptr;
    }

    // calculate new shape
    std::pair<unsigned int, unsigned int> resultShape = {toRow - fromRow, this->getShapeY()};

    // calculate required size
    unsigned int size = resultShape.first * resultShape.second;

    // Throw an error if size is zero
    if (size == 0) {
        printf("Error: Cannot initalize zero Tensor: Keep in mind that upper bounds are excluded");
        delete this;
        return nullptr;
    }

    // allocate memory
    float* d_value_copy = reserveMemoryOnDevice(size);

    // check for errors during allocation
    this->handleError(d_value_copy, "Error: Memory allocation for Tensor deepcopy failed");

    // calculate position
    unsigned int from = fromRow * this->getShapeY();

    // synchronize before accessing data
    //CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // clone values
    cudaMemDup(this->getValue() + from, d_value_copy, resultShape.first, resultShape.second, false);

    return new Tensor(d_value_copy, resultShape, true);
}

Tensor* Tensor::getCols(unsigned int fromCol, unsigned int toCol) {
    // check bounds
    if (fromCol > toCol || fromCol > this->getShapeY() || toCol > this->getShapeY()) {
        printf("Error: column indexing is wrong");
        delete this;
        return nullptr;
    }

    // calculate new shape
    std::pair<unsigned int, unsigned int> resultShape = {this->getShapeX(), toCol - fromCol};

    // calculate required size
    unsigned int size = resultShape.first * resultShape.second;

    // Throw an error if size is zero
    if (size == 0) {
        printf("Error: Cannot initialize zero Tensor: Keep in mind that upper bounds are excluded");
        delete this;
        return nullptr;
    }

    // allocate memory
    float* d_value_copy = reserveMemoryOnDevice(size);

    // check for errors during allocation
    this->handleError(d_value_copy, "Error: Memory allocation for Tensor deepcopy failed");

    // calculate position
    unsigned int from = fromCol;

    // clone values
    cudaError_t err = cudaMemcpy2D(d_value_copy, resultShape.second * sizeof(float),
                                   this->getValue() + from, this->getShapeY() * sizeof(float),
                                   resultShape.second * sizeof(float), resultShape.first,
                                   cudaMemcpyDeviceToDevice);

    this->handleError(err, "Error: Memory copy for slicing tensor failed");

    return new Tensor(d_value_copy, resultShape, true);
}

Tensor* Tensor::getVal(unsigned int row, unsigned int col) {
    // check bounds
    if (row > this->getShapeX() || col > this->getShapeY()) {
        printf("Error: index out of bounds");
        delete this;
        return nullptr;
    }
    // allocate memory
    float* d_value_copy = reserveMemoryOnDevice(1);

    // check for errors during allocation
    this->handleError(d_value_copy, "Error: Memory allocation for Tensor deepcopy failed");

    // calculate position of value that is to be retrieved
    unsigned int position = row * this->getShapeY() + col;

    // clone values
    cudaMemDup(this->getValue() + position, d_value_copy, 1, 1, false);

    // check for copy errors
    this->handleError(d_value_copy, "Error: Memory copy failed");

    return new Tensor(d_value_copy, {1, 1}, true);
}

Tensor* Tensor::get(unsigned int fromRow, unsigned int toRow, unsigned int fromCol, unsigned int toCol) {
    // check bounds (rows)
    if (fromRow > toRow || fromRow > this->getShapeX() || toRow > this->getShapeX()) {
        printf("Error: row indexing is wrong");
        delete this;
        return nullptr;
    }

    // check bounds (columns)
    if (fromCol > toCol || fromCol > this->getShapeY() || toCol > this->getShapeY()) {
        printf("Error: column indexing is wrong");
        delete this;
        return nullptr;
    }

    // calculate new shape
    std::pair<unsigned int, unsigned int> resultShape = {toRow - fromRow, toCol - fromCol};

    // calculate required size
    unsigned int size = resultShape.first * resultShape.second;

    // Throw an error if size is zero
    if (size == 0) {
        printf("Error: Cannot initialize zero Tensor: Keep in mind that upper bounds are excluded");
        delete this;
        return nullptr;
    }

    // allocate memory for the slice
    float* d_value_copy = reserveMemoryOnDevice(size);
    this->handleError(d_value_copy, "Error: Memory allocation for Tensor deepcopy failed");

    // calculate source offset and original column count
    unsigned int originalCols = this->getShapeY();
    float* srcPtr = this->getValue() + (fromRow * originalCols + fromCol);

    // use cudaMemcpy2D to copy the submatrix row by row
    cudaError_t err = cudaMemcpy2D(d_value_copy, resultShape.second * sizeof(float),
                                   srcPtr, originalCols * sizeof(float),
                                   resultShape.second * sizeof(float), resultShape.first,
                                   cudaMemcpyDeviceToDevice);

    this->handleError(err, "Error: Memory copy for slicing tensor failed");

    return new Tensor(d_value_copy, resultShape, true);
}

// #######
// Copying
// #######

Tensor* Tensor::deepcopy() {
    // cache size
    unsigned int size = this->getSize();

    // allocate memory
    float* d_value_copy = reserveMemoryOnDevice(size);

    // check for errors during allocation
    this->handleError(d_value_copy, "Error: Memory allocation for Tensor deepcopy failed");

    // clone values
    cudaMemDup(this->getValue(), d_value_copy, this->getShapeX(), this->getShapeY(), false);

    return new Tensor(d_value_copy, this->getShape(), this->getTrackGradient());
}

// ###############
// Backpropagation
// ###############

void Tensor::backward() {
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

void Tensor::asyncbackpropsgd(float lr) {
    // TODO
    // skip if either gradient tracking is disabled or there are no preceding calculations
    if (this->getTrackGradient() && !this->isLeaf()) {
        this->gradFunction(this);

        // recursively calculate gradients of preceding operations and optimize
        if (this->getArg1() != nullptr) {
            // SGD step
            /* asynchronous */
            this->getArg1()->asyncsgd(lr);

            this->getArg1()->asyncbackpropsgd(lr);
        }

        if (this->getArg2() != nullptr) {
            // SGD step
            /* asynchronous */
            this->getArg2()->asyncsgd(lr); 

            this->getArg2()->asyncbackpropsgd(lr);
        }
    }
}

// ######################
// Shape comparison utils
// ######################

bool Tensor::sameShape(Tensor other) const {
    // returns true if tensors are of same shape
    return (this->getShapeX() == other.getShapeX()) && (this->getShapeY() == other.getShapeY());
}

bool Tensor::matMulCompatible(Tensor other) const {
    // returns true if matrices are compatible for matmul
    return this->getShapeY() == other.getShapeX();
}

// #########################
// Tensor level optimization
// #########################

void Tensor::sgd(float lr) {
    if (!this->isGradientSet() || this->getGradient() == nullptr) {
        printf("Error: Optimization step not possible because gradient is not set");
        delete this;
        exit(EXIT_FAILURE);
    }

    this->handleError(scaledSubtraction(this->getValue(), this->getValue(), this->getSize(), this->getGradient(), this->getSize(), lr, false), "Error: scaledSubtraction failed");
}

void Tensor::asyncsgd(float lr) {
    if (!this->isGradientSet() || this->getGradient() == nullptr) {
        printf("Error: Optimization step not possible because gradient is not set");
        delete this;
        exit(EXIT_FAILURE);
    }

    this->handleError(scaledSubtraction(this->getValue(), this->getValue(), this->getSize(), this->getGradient(), this->getSize(), lr, true), "Error: scaledSubtraction failed");
}

// ################################################
// Addition (calculation, gradient and op overload)
// ################################################

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

    // copy gradient of this Tensor (and multply by I, neglected)
    if (currentTensor->isGradientSet()) {

        // the gradient of the last computation in this computational graph wrt this Tensor
        float* partialGradient = currentTensor->getGradient();

        if (arg1->getTrackGradient()) {
            cudaError_t err = cudaMemDup(partialGradient, arg1->getGradient(), arg1->getShapeX(), arg1->getShapeY(), false);
            
            // check for errors
            currentTensor->handleError(err, "Error in addition gradient, when duplicating memory cells: exit");

            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            cudaError_t err = cudaMemDup(partialGradient, arg2->getGradient(), arg2->getShapeX(), arg2->getShapeY(), false);

            // check for errors
            currentTensor->handleError(err, "Error in addition gradient, when duplicating memory cells: exit");

            arg2->changeGradientSet(true);
        }
    } else {

        if (arg1->getTrackGradient()) {
            cudaError_t err = constants(arg1->getGradient(), arg1->getSize(), 1.0f);

            // check for errors
            currentTensor->handleError(err, "Error in addition gradient, when initalizing constant memory cells: exit\n");

            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            cudaError_t err = constants(arg2->getGradient(), arg2->getSize(), 1.0f);

            // check for errors
            currentTensor->handleError(err, "Error in addition gradient, when initalizing constant memory cells: exit\n");

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

    // check error
    this->handleError(d_result, "Error: Addition failed");

    // disable gradientSet flag
    this->changeGradientSet(false);
    other.changeGradientSet(false);

    return new Tensor(d_result, this->getShape(), true, &additionGradient, this, this->getShape(), &other, other.getShape());
}

Tensor* Tensor::operator+(Tensor &other) {
    return this->add(other);
}

// ###################################################
// Subtraction (calculation, gradient and op overload)
// ###################################################

/**
 * @brief Computes the gradient for the subtraction operation in a computational graph.
 *
 * This function calculates the gradient of the current tensor with respect to its arguments
 * (arg1 and arg2) and updates their gradients accordingly. If the gradient for the current
 * tensor is already set, it duplicates this gradient to the arguments' gradients (positive for arg1, negative for arg2). If the
 * gradient is not set, it initializes the arguments' gradients to a constant value of positive and negative 1.0, respectively.
 *
 * @param currentTensor Pointer to the current tensor whose gradient is being computed.
 */
static void subtractionGradient(Tensor* currentTensor) {
    // cache args
    Tensor* arg1 = currentTensor->getArg1();
    Tensor* arg2 = currentTensor->getArg2();

    // copy gradient of this Tensor (and multply by I, neglected)
    if (currentTensor->isGradientSet()) {

        // the gradient of the last computation in this computational graph wrt this Tensor
        float* partialGradient = currentTensor->getGradient();

        if (arg1->getTrackGradient()) {
            currentTensor->handleError(cudaMemDup(partialGradient, arg1->getGradient(), arg1->getShapeX(), arg1->getShapeY(), false), "Error: Gradient calculation (operation: subtraction) failed");
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            currentTensor->handleError(cudaMemDupScaled(partialGradient, arg2->getGradient(), -1.0f, arg2->getSize()), "Error: Gradient calculation (operation: subtraction) failed");
            arg2->changeGradientSet(true);
        }
    } else {
        if (arg1->getTrackGradient()) {
            currentTensor->handleError(constants(arg1->getGradient(), arg1->getSize(), 1.0f), "Error: Gradient calculation (operation: subtraction) failed");
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            currentTensor->handleError(constants(arg2->getGradient(), arg2->getSize(), -1.0f), "Error: Gradient calculation (operation: subtraction) failed");
            arg2->changeGradientSet(true);
        }
    }
}

Tensor* Tensor::sub(Tensor &other) {
    // increment reference count of results args (dependencies)
    this->addReference();
    other.addReference();

    float* d_result = tensorsubAlloc(this->getValue(), this->getSize(), other.getValue(), other.getSize());
    
    // check error
    this->handleError(d_result, "Error: Subtraction failed");

    // disable gradientSet flag
    this->changeGradientSet(false);
    other.changeGradientSet(false);

    return new Tensor(d_result, this->getShape(), true, &subtractionGradient, this, this->getShape(), &other, other.getShape());
}

Tensor* Tensor::operator-(Tensor &other) {
    return this->sub(other);
}

// #############################################################
// Matrix multiplication (calculation, gradient and op overload)
// #############################################################

/**
 * @brief Computes the gradient of a matrix multiplication (C = A * B) wrt to A and B
 * @param Pointer to the current tensor whose gradient is being computed
 */
static void matmulGradient(Tensor* currentTensor) {
    // cache tensors
    Tensor* A = currentTensor->getArg1();
    Tensor* B = currentTensor->getArg2();

    // gradOut_currTensor = gradient of output of this computational graph wrt to this Tensor
    float* gradOut_currTensor = nullptr;

    if (!currentTensor->isGradientSet()) {
        // IMPORTANT: THIS CASE IS ONLY DEFINED FOR MATMUL OPERATION THAT RESULT IN A SCALAR (SHAPE=(1, 1))
        if (currentTensor->getShapeX() != 1 || currentTensor->getShapeY() != 1) {
            printf("Error: Backward cannot be called on a non-scalar result of a matrix multiplication");
            delete currentTensor;
        }

        /* this tensors gradient is not set => we started to differentiate from here so no need to apply the chain rule (saves a multiplication with Identity matrix) */
        // dA = I * B^T = B^T
        // IMPORTANT: THIS CASE IS ONLY DEFINED FOR MATMUL OPERATION THAT RESULT IN A SCALAR (SHAPE=(1, 1)) -> therefore multiplication with I is neglectable (shape would be 1x1)
        if (A->getTrackGradient()) {
            currentTensor->handleError(cudaMemDup(B->getValue(), A->getGradient(), B->getShapeX(), B->getShapeY(), true), "Error: Duplicating memory in matmulGradient failed");  // size of B = size of B^T (size=#entries)
            A->changeGradientSet(true);
        }
        // dB = A^T * I = A^T
        if (B->getTrackGradient()) {
            currentTensor->handleError(cudaMemDup(A->getValue(), B->getGradient(), A->getShapeX(), A->getShapeY(), true), "Error: Duplicating memory in matmulGradient failed");
            B->changeGradientSet(true);
        }

    } else {
        // gradOut_currTensor is gradient of current tensor
        gradOut_currTensor = currentTensor->getGradient();

        // cache shapes of gradOut_currTensor and original matrices
        int m = A->getShapeX();    // rows of A
        int k = A->getShapeY();    // cols of A (= rows of B)
        int n = B->getShapeY();    // cols of B

        // dA = gradOut_currTensor * B^T
        currentTensor->handleError(gemm(handleLt, gradOut_currTensor, B->getValue(), A->getGradient(), 
            m,      // rows of result (dA)
            k,      // cols of result (dA)
            n,      // common dimension
            CUBLAS_OP_N, CUBLAS_OP_T), 
            "Error: Gradient calculation of arg1 of matmul failed");
        A->changeGradientSet(true);

        // dB = A^T * gradOut_currTensor
        currentTensor->handleError(gemm(handleLt, A->getValue(), gradOut_currTensor, B->getGradient(), 
            k,      // rows of result (dB)
            n,      // cols of result (dB)
            m,      // common dimension
            CUBLAS_OP_T, CUBLAS_OP_N), 
            "Error: Gradient calculation of arg2 of matmul failed");
        B->changeGradientSet(true);    
    }
}

Tensor* Tensor::matmul(Tensor &other) {

    other.addReference();  // other's references are incremented twice (solve garbage collector fight)

    if (this->matMulCompatible(other)) {

        // increment reference count of results args (dependencies)
        this->addReference();
        other.addReference();

        // float* d_resultMatrix = matmulAlloc(handle, this->getShapeX(), this->getShapeY(), other.getShapeX(), other.getShapeY(), this->getValue(), other.getValue());
        float* d_resultMatrix = gemmA(handleLt, this->getValue(), other.getValue(), (int) this->getShapeX(), (int) other.getShapeY(), (int) this->getShapeY(), CUBLAS_OP_N, CUBLAS_OP_N);

        // check error
        this->handleError(d_resultMatrix, "Error: Matrix multiplication failed");

        // disable gradientSet flag
        this->changeGradientSet(false);
        other.changeGradientSet(false);

        // Create new tensor with its own memory space
        return new Tensor(d_resultMatrix, 
                    {this->getShapeX(), other.getShapeY()},
                    true, 
                    matmulGradient,
                    this,
                    this->getShape(),
                    &other,
                    other.getShape());
    }
    printf("Error: Incompatible shapes for matrix multiplication\n");
    delete this;
    exit(EXIT_FAILURE);
}

Tensor* Tensor::operator*(Tensor &other) {
    return matmul(other);
}

/*static void scaleGradient(Tensor* currentTensor) {
    // cache arg1
    Tensor* tensorGrad = currentTensor->getArg1();

    // check if gradient is tracked
    if (tensorGrad->getTrackGradient()) {

        // compute gradient store in tensor's gradient attribute
        if (currentTensor->isGradientSet()) {
            // get partial gradient 
            float* partialGrad = currentTensor->getGradient();

            // multiply gradient with preceding partial gradient (chain rule)
            // compute gradient store in tensor's gradient attribute, pass currentTensor's value for simplifying computation
            currentTensor->handleError(tanhGrad(tensorGrad->getGradient(), currentTensor->getValue(), partialGrad, tensorGrad->getSize()), "Error: gradient calculation (operation: scalar multiplication) failed");
        } else {
            // no previous partial gradient
            // compute gradient store in tensor's gradient attribute, pass currentTensor's value for simplifying computation
            currentTensor->handleError(constants(tensorGrad->getGradient(), tensorGrad->getSize(), ), "Error: gradient calculation (operation: scalar multiplication) failed");
        }

        // set gradientSet flag
        tensorGrad->changeGradientSet(true);
    }
}

Tensor* Tensor::scale(float scalar) {
    // increment reference count of results args (dependencies)
    this->addReference();

    // allocate memory
    float* d_result = reserveMemoryOnDevice(this->getSize());

    // check for errors during allocation
    this->handleError(d_result, "Error: Memory allocation for scaling failed");

    // error checking is done is hadamardAlloc
    this->handleError(cudaMemDupScaled(this->getValue(), d_result, scalar, this->getSize(), false), "Error: Scaled memory copy for Tensor-scaling failed");

    // disable gradientSet flag
    this->changeGradientSet(false);

    // return new Tensor
    return new Tensor(d_result, this->getShape(), true, scaleGradient, this, this->getShape());
}*/

// ###################################################
// Dot product (calculation, gradient and op overload)
// ###################################################

/**
 * @brief Computes and stores the derivatives of the dot product operation
 * 
 * Sets gradient for arg1 and arg2 of the current Tensor (if they track their gradient)
 * Uses the chain rule to calculate the gradient
 * @param currentTensor The Tensor this function is called for
 */
static void dotGradient(Tensor* currentTensor) {
    // cache args
    Tensor* arg1 = currentTensor->getArg1();
    Tensor* arg2 = currentTensor->getArg2();

    // if a gradient is set the gradient calculations needs to factor in the gradient wrt to this tensor (chain rule)
    if (currentTensor->isGradientSet()) {
        // partialGradient.shape = (1, 1) = scalar -> darg1 = partialGradient * arg2 and vice versa
        // the gradient of the last computation in this computational graph wrt this Tensor
        float* partialGradient = currentTensor->getGradient();

        // calculate gradient, multiply with partialGradient, if gradient is tracked at all
        if (arg1->getTrackGradient()) {
            currentTensor->handleError(cudaMemDupScaled(arg2->getValue(), arg1->getGradient(), partialGradient, arg1->getSize()), "Error: Gradient calculation (operation: dot product) failed");
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            currentTensor->handleError(cudaMemDupScaled(arg1->getValue(), arg2->getGradient(), partialGradient, arg2->getSize()), "Error: Gradient calculation (operation: dot product) failed");
            arg2->changeGradientSet(true);
        }

    } else {
        // no need to apply chain rule since this is the root of the computational graph
        if (arg1->getTrackGradient()) {
            currentTensor->handleError(cudaMemDup(arg2->getValue(), arg1->getGradient(), arg1->getShapeX(), arg1->getShapeY(), false), "Error: Gradient calculation (operation: dot product) failed");
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            currentTensor->handleError(cudaMemDup(arg1->getValue(), arg2->getGradient(), arg2->getShapeX(), arg2->getShapeY(), false), "Error: Gradient calculation (operation: dot product) failed");
            arg2->changeGradientSet(true);
        }
    }
}

Tensor* Tensor::dot(Tensor &other) {
    // check vector
    if (this->getShapeY() != 1 || other.getShapeY() != 1) {
        printf("Error: Incompatible shapes for dot product\n");
        delete this;
        exit(EXIT_FAILURE);
    }
    // increment reference count of results args (dependencies)
    this->addReference();
    other.addReference();

    // error checking is done is hadamardAlloc
    float* d_result = dotAlloc(handle, this->getValue(), this->getShapeX(), other.getValue(), other.getShapeX());

    // check error
    this->handleError(d_result, "Error: Dot product failed");

    // disable gradientSet flag
    this->changeGradientSet(false);
    other.changeGradientSet(false);

    // return new Tensor
    return new Tensor(d_result, {1, 1}, true, dotGradient, this, this->getShape(), &other, other.getShape());
}

// ########################################################
// Hadamard product (calculation, gradient and op overload)
// ########################################################

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

    // if a gradient is set the gradient calculations needs to factor in the gradient wrt to this tensor (chain rule)
    if (currentTensor->isGradientSet()) {
        // the gradient of the last computation in this computational graph wrt this Tensor
        float* partialGradient = currentTensor->getGradient();

        // calculate gradient, multiply with partialGradient, if gradient is tracked at all
        if (arg1->getTrackGradient()) {
            currentTensor->handleError(hadamard(arg1->getGradient(), arg2->getValue(), partialGradient, arg2->getShape()), "Error: Gradient calculation (operation: hadamard product) failed");
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            currentTensor->handleError(hadamard(arg2->getGradient(), arg1->getValue(), partialGradient, arg1->getShape()), "Error: Gradient calculation (operation: hadamard product) failed");
            arg2->changeGradientSet(true);
        }

    } else {
        // no need to apply chain rule since this is the root of the computational graph
        if (arg1->getTrackGradient()) {
            currentTensor->handleError(cudaMemDup(arg2->getValue(), arg1->getGradient(), arg1->getShapeX(), arg1->getShapeY(), false), "Error: Gradient calculation (operation: hadamard product) failed");
            arg1->changeGradientSet(true);
        }

        if (arg2->getTrackGradient()) {
            currentTensor->handleError(cudaMemDup(arg1->getValue(), arg2->getGradient(), arg2->getShapeX(), arg2->getShapeY(), false), "Error: Gradient calculation (operation: hadamard product) failed");
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

    // check error
    this->handleError(d_result, "Error: Hadamard product failed");

    // disable gradientSet flag
    this->changeGradientSet(false);
    other.changeGradientSet(false);

    // return new Tensor
    return new Tensor(d_result, this->getShape(), true, hadamardGradient, this, this->getShape(), &other, other.getShape());
}

Tensor* Tensor::operator%(Tensor &other) {
    return hadamardProduct(other);
}

// ###############################
// ReLU (calculation and gradient)
// ###############################

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

    // check if gradient is tracked
    if (tensorGrad->getTrackGradient()) {

        // compute gradient store in tensor's gradient attribute
        if (currentTensor->isGradientSet()) {
            // get partial gradient 
            float* partialGrad = currentTensor->getGradient();

            // multiply gradient with preceding partial gradient (chain rule)
            currentTensor->handleError(reluGrad(tensorGrad->getGradient(), tensorGrad->getValue(), partialGrad, tensorGrad->getSize()), "Error: gradient calculation (operation: ReLU) failed");
        } else {
            // no previous partial gradient
            currentTensor->handleError(reluGrad(tensorGrad->getGradient(), tensorGrad->getValue(), nullptr, tensorGrad->getSize()), "Error: gradient calculation (operation: ReLU) failed");
        }
        // set gradientSet flag
        tensorGrad->changeGradientSet(true);
    }
}

Tensor* Tensor::relu() {
    // increment reference count of results args (dependencies)
    this->addReference();

    float* d_tensorValue = reluAlloc(this->getValue(), this->getSize());

    // check error
    this->handleError(d_tensorValue, "Error: ReLU failed");

    // disable gradientSet flag
    this->changeGradientSet(false);

    return new Tensor(d_tensorValue, this->getShape(), true, reluGradient, this, this->getShape());
}

// ##################################
// Sigmoid (calculation and gradient)
// ##################################

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

    // check if gradient is tracked
    if (tensorGrad->getTrackGradient()) {

        // compute gradient store in tensor's gradient attribute
        if (currentTensor->isGradientSet()) {
            // get partial gradient 
            float* partialGrad = currentTensor->getGradient();

            // multiply gradient with preceding partial gradient (chain rule)
            currentTensor->handleError(sigmoidGrad(tensorGrad->getGradient(), currentTensor->getValue(), partialGrad, tensorGrad->getSize()), "Error: gradient calculation (operation: sigmoid) failed");
        } else {
            // no previous partial gradient
            currentTensor->handleError(sigmoidGrad(tensorGrad->getGradient(), currentTensor->getValue(), nullptr, tensorGrad->getSize()), "Error: gradient calculation (operation: sigmoid) failed");
        }

        // set gradientSet flag
        tensorGrad->changeGradientSet(true);
    }
}

Tensor* Tensor::sigmoid() {
    // increment reference count of results args (dependencies)
    addReference();

    // compute sigmoid func store result in newly allocated memory section and return pointer to it
    float* d_sigmoidValue = sigmoidAlloc(this->getValue(), this->getSize());

    // check error
    this->handleError(d_sigmoidValue, "Error: Sigmoid failed");

    // disable gradientSet flag
    this->changeGradientSet(false);

    // return new tensor that has holds result of the sigmoid function as a value and corresponding shape and gradient function
    return new Tensor(d_sigmoidValue, this->getShape(), true, sigmoidGradient, this, this->getShape());
}

// ###############################
// Tanh (calculation and gradient)
// ###############################

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

    // check if gradient is tracked
    if (tensorGrad->getTrackGradient()) {

        // compute gradient store in tensor's gradient attribute
        if (currentTensor->isGradientSet()) {
            // get partial gradient 
            float* partialGrad = currentTensor->getGradient();

            // multiply gradient with preceding partial gradient (chain rule)
            // compute gradient store in tensor's gradient attribute, pass currentTensor's value for simplifying computation
            currentTensor->handleError(tanhGrad(tensorGrad->getGradient(), currentTensor->getValue(), partialGrad, tensorGrad->getSize()), "Error: gradient calculation (operation: tanh) failed");
        } else {
            // no previous partial gradient
            // compute gradient store in tensor's gradient attribute, pass currentTensor's value for simplifying computation
            currentTensor->handleError(tanhGrad(tensorGrad->getGradient(), currentTensor->getValue(), nullptr, tensorGrad->getSize()), "Error: gradient calculation (operation: tanh) failed");
        }

        // set gradientSet flag
        tensorGrad->changeGradientSet(true);
    }
}

Tensor* Tensor::tanh() {
    // increment reference count of results args (dependencies)
    this->addReference();

    // compute tanh func store result in newly allocated memory section and return pointer to it
    float* d_tanhValue = tanhAlloc(this->getValue(), this->getSize());

    // check error
    this->handleError(d_tanhValue, "Error: tanh failed");

    // disable gradientSet flag
    this->changeGradientSet(false);

    // return new tensor that holds the result of the tanh function as a value and corresponding shape and gradient function
    return new Tensor(d_tanhValue, this->getShape(), true, tanhGradient, this, this->getShape());
}

// ##################################
// Softmax (calculation and gradient)
// ##################################

/**
 * @brief Computes the gradient of the softmax function for the given tensor.
 *
 * This function calculates the gradient of the softmax function and stores it
 * in the gradient attribute of the tensor. It also sets the gradientSet flag to true
 * to indicate that the gradient has been computed and stored.
 *
 * @param currentTensor Pointer to the tensor for which the softmax gradient is to be computed.
 */
static void softmaxGradient(Tensor* currentTensor) {
    // cache arg1
    Tensor* tensorGrad = currentTensor->getArg1();

    // check if gradient is tracked
    if (tensorGrad->getTrackGradient()) {
        // compute gradient store in tensor's gradient attribute
        if (currentTensor->isGradientSet()) {
            // get partial gradient 
            float* partialGrad = currentTensor->getGradient();

            // multiply gradient with preceding partial gradient (chain rule)
            currentTensor->handleError(softmaxGrad(tensorGrad->getGradient(), currentTensor->getValue(), partialGrad, tensorGrad->getSize()), 
                "Error: gradient calculation (operation: softmax) failed");
        } else {
            // no previous partial gradient
            currentTensor->handleError(softmaxGrad(tensorGrad->getGradient(), currentTensor->getValue(), nullptr, tensorGrad->getSize()),
                "Error: gradient calculation (operation: softmax) failed");
        }

        // set gradientSet flag
        tensorGrad->changeGradientSet(true);
    }
}

Tensor* Tensor::softmax() {
    // increment reference count of results args (dependencies)
    this->addReference();

    // compute tanh func store result in newly allocated memory section and return pointer to it
    float* d_result = softmaxAlloc(this->getValue(), this->getSize());

    // check error
    this->handleError(d_result, "Error: Softmax failed");

    // disable gradientSet flag
    this->changeGradientSet(false);

    // return new tensor that holds the result of the tanh function as a value and corresponding shape and gradient function
    return new Tensor(d_result, this->getShape(), true, softmaxGradient, this, this->getShape());
}

// ##################################
// L2 Loss (calculation and gradient)
// ##################################

/**
 * @brief Calculates the gradient for the L2 Loss operation applied to a tensor.
 *
 * This function retrieves the prediction tensor (arg1) and, if gradient tracking is enabled,
 * fetches the corresponding label tensor (arg2). It then computes the L2 Loss gradient using
 * the prediction's gradient and value along with the label values. The resulting gradient is
 * stored in the tensor's gradient attribute. In case of an error during the gradient computation,
 * the function handles it by invoking the error handler on the tensor.
 *
 * @param currentTensor Pointer to the Tensor structure for which the L2 Loss gradient is computed.
 *
 * @note The gradient is only computed wrt to the prediction values (gradient wrt to the labels makes no sense).
 */
static void l2Gradient(Tensor* currentTensor) {
    // cache arg1 (prediction)
    Tensor* predicted = currentTensor->getArg1();   // predictions of L2 Loss operation

    // check if gradient is tracked
    if (predicted->getTrackGradient()) {
        Tensor* actual = currentTensor->getArg2();      // labels of L2 Loss operation

        // compute gradient store in tensor's gradient attribute
        if (currentTensor->isGradientSet()) {
            // get partial gradient 
            float* partialGrad = currentTensor->getGradient();

            // multiply gradient with preceding partial gradient (chain rule)
            // compute gradient store in tensor's gradient attribute, pass currentTensor's value for simplifying computation
            currentTensor->handleError(l2LossGrad(predicted->getGradient(), predicted->getValue(), actual->getValue(), partialGrad, predicted->getSize()), "Error: gradient calculation (operation: L2 Loss) failed");
        } else {
            // no previous partial gradient
            // compute gradient store in tensor's gradient attribute, pass currentTensor's value for simplifying computation
            currentTensor->handleError(l2LossGrad(predicted->getGradient(), predicted->getValue(), actual->getValue(), nullptr, predicted->getSize()), "Error: gradient calculation (operation: L2 Loss) failed");
        }

        // set gradientSet flag
        predicted->changeGradientSet(true);
    }
}

Tensor* Tensor::l2(Tensor &other) {
    // increment reference count of results args (dependencies)
    this->addReference();
    other.addReference();

    float* d_result = l2LossAlloc(this->getValue(), other.getValue(), this->getShape(), other.getShape());

    // check error
    this->handleError(d_result, "Error: An error occured during the computation of the L2 Loss");

    // disable gradientSet flag
    this->changeGradientSet(false);
    other.changeGradientSet(false);

    // return new Tensor
    return new Tensor(d_result, {1, 1}, true, l2Gradient, this, this->getShape(), &other, other.getShape());
}

// wrapper for backpropagation: allows backpropagation without calculating loss
Tensor* Tensor::l2NoVal(Tensor &other) {
    // increment reference count of results args (dependencies)
    this->addReference();
    other.addReference();

    float* d_result = reserveMemoryOnDevice(1);

    // disable gradientSet flag
    this->changeGradientSet(false);
    other.changeGradientSet(false);

    // return new Tensor
    return new Tensor(d_result, {1, 1}, true, l2Gradient, this, this->getShape(), &other, other.getShape());
}

// #########################################################
// Categorical cross entropy Loss (calculation and gradient)
// #########################################################

/**
 * @brief Calculates the gradient for the categorical cross entropy Loss operation applied to a tensor.
 *
 * This function retrieves the prediction tensor (arg1) and, if gradient tracking is enabled,
 * fetches the corresponding label tensor (arg2). It then computes the cross entropy Loss gradient using
 * the prediction's gradient and value along with the label values. The resulting gradient is
 * stored in the tensor's gradient attribute. In case of an error during the gradient computation,
 * the function handles it by invoking the error handler on the tensor.
 *
 * @param currentTensor Pointer to the Tensor structure for which the cross entropy Loss gradient is computed.
 *
 * @note The gradient is only computed wrt to the prediction values (gradient wrt to the labels makes no sense).
 */
static void crossEntropyLossGradient(Tensor* currentTensor) {
    // cache arg1 (prediction)
    Tensor* predicted = currentTensor->getArg1();   // predictions of L2 Loss operation

    // check if gradient is tracked
    if (predicted->getTrackGradient()) {
        Tensor* actual = currentTensor->getArg2();      // labels of L2 Loss operation

        // compute gradient store in tensor's gradient attribute
        if (currentTensor->isGradientSet()) {
            // get partial gradient 
            float* partialGrad = currentTensor->getGradient();

            // multiply gradient with preceding partial gradient (chain rule)
            // compute gradient store in tensor's gradient attribute, pass currentTensor's value for simplifying computation
            currentTensor->handleError(crossEntropyLossGrad(predicted->getGradient(), predicted->getValue(), actual->getValue(), partialGrad, predicted->getSize()), "Error: gradient calculation (operation: Cross Entropy Loss) failed");
        } else {
            // no previous partial gradient
            // compute gradient store in tensor's gradient attribute, pass currentTensor's value for simplifying computation
            currentTensor->handleError(crossEntropyLossGrad(predicted->getGradient(), predicted->getValue(), actual->getValue(), nullptr, predicted->getSize()), "Error: gradient calculation (operation: Cross EntropyL2 Loss) failed");
        }

        // set gradientSet flag
        predicted->changeGradientSet(true);
    }
}

Tensor* Tensor::categoricalCrossEntropy(Tensor &other) {
    if (!this->sameShape(other) || this->getShapeY() != 1 || other.getShapeY() != 1) {
        printf("Tensors must be vectors and have the same shape to comnpute the cross entropy loss");
        delete this;
        return nullptr;
    }

    // increment reference count of results args (dependencies)
    this->addReference();
    other.addReference();

    float* d_result = categoricalCrossEntropyLossAlloc(this->getValue(), other.getValue(), this->getSize());

    // check error
    this->handleError(d_result, "Error: An error occured during the computation of the cross entropy loss");

    // disable gradientSet flag
    this->changeGradientSet(false);
    other.changeGradientSet(false);

    // return new Tensor
    return new Tensor(d_result, {1, 1}, true, crossEntropyLossGradient, this, this->getShape(), &other, other.getShape());
}

// ##########################################################
// Optimized standard forward pass (calculation and gradient)
// ##########################################################

static void sfpassGradient(Tensor* currentTensor) {
    // cache tensors
    Tensor* weight = currentTensor->getArg1();
    Tensor* input = currentTensor->getArg2();
    Tensor* bias = currentTensor->getArg3();

    // gradOut_currTensor = gradient of output of this computational graph wrt to this Tensor
    float* gradOut_currTensor = nullptr;

    if (!currentTensor->isGradientSet()) {
        // IMPORTANT: THIS CASE IS ONLY DEFINED FOR MATMUL OPERATION THAT RESULT IN A SCALAR (SHAPE=(1, 1))
        if (currentTensor->getShapeX() != 1 || currentTensor->getShapeY() != 1) {
            printf("Error: Backward cannot be called on a non-scalar result of a matrix multiplication");
            delete currentTensor;
        }

        if (bias->getTrackGradient()) {
            currentTensor->handleError(constants(bias->getGradient(), bias->getSize(), 1.0f), "Error: Duplicating memory in sfpass failed");
            bias->changeGradientSet(true);
        }

        /* this tensors gradient is not set => we started to differentiate from here so no need to apply the chain rule (saves a multiplication with Identity matrix) */
        // dA = I * B^T = B^T
        // IMPORTANT: THIS CASE IS ONLY DEFINED FOR MATMUL OPERATION THAT RESULT IN A SCALAR (SHAPE=(1, 1)) -> therefore multiplication with I is neglectable (shape would be 1x1)
        if (weight->getTrackGradient()) {
            currentTensor->handleError(cudaMemDup(input->getValue(), weight->getGradient(), input->getShapeX(), input->getShapeY(), true), "Error: Duplicating memory in sfpass failed");  // size of B = size of B^T (size=#entries)
            weight->changeGradientSet(true);
        }
        // dB = A^T * I = A^T
        if (input->getTrackGradient()) {
            currentTensor->handleError(cudaMemDup(weight->getValue(), input->getGradient(), weight->getShapeX(), weight->getShapeY(), true), "Error: Duplicating memory in sfpass failed");
            input->changeGradientSet(true);
        }

    } else {
        // gradOut_currTensor is gradient of current tensor
        gradOut_currTensor = currentTensor->getGradient();

        // cache shapes of gradOut_currTensor and original matrices
        int m = weight->getShapeX();    // rows of A
        int k = weight->getShapeY();    // cols of A (= rows of B)
        int n = input->getShapeY();    // cols of B

        if (bias->getTrackGradient()) {
            currentTensor->handleError(cudaMemDup(gradOut_currTensor, bias->getGradient(), bias->getShapeX(), bias->getShapeY(), false), "Error: Duplicating memory in sfpass failed");
            bias->changeGradientSet(true);
        }

        // dA = gradOut_currTensor * B^T
        currentTensor->handleError(gemm(handleLt, gradOut_currTensor, input->getValue(), weight->getGradient(), 
            m,      // rows of result (dA)
            k,      // cols of result (dA)
            n,      // common dimension
            CUBLAS_OP_N, CUBLAS_OP_T), 
            "Error: Gradient calculation of arg1 of sfpass failed");
        weight->changeGradientSet(true);

        // dB = A^T * gradOut_currTensor
        currentTensor->handleError(gemm(handleLt, weight->getValue(), gradOut_currTensor, input->getGradient(), 
            k,      // rows of result (dB)
            n,      // cols of result (dB)
            m,      // common dimension
            CUBLAS_OP_T, CUBLAS_OP_N), 
            "Error: Gradient calculation of arg2 of sfpass failed");
        input->changeGradientSet(true);    
    }
}

Tensor* Tensor::sfpass(Tensor &weight, Tensor &bias) {
    // increment reference count of results args (dependencies)
    this->addReference();
    weight.addReference();
    bias.addReference();

    int m = weight.getShapeX();
    int n = this->getShapeY();
    int k = this->getShapeX();

    float* d_result = forwardPass(handleLt, weight.getValue(), this->getValue(), bias.getValue(), m, n, k, CUBLAS_OP_N, CUBLAS_OP_N);

    // check error
    this->handleError(d_result, "Error: An error occured during the computation of the forward pass.");

    // disable gradientSet flag
    this->changeGradientSet(false);
    weight.changeGradientSet(false);
    bias.changeGradientSet(false);

    // return new Tensor
    return new Tensor(d_result, {m, n}, true, sfpassGradient, &weight, weight.getShape(), this, this->getShape(), &bias, bias.getShape());
}

// ######################################
// Transposing (calculation and gradient)
// ######################################

static void transposeGradient(Tensor* currentTensor) {
    /* thisGradient = transpose(currentTensorGradient) */

    // cache arg1
    Tensor* destination = currentTensor->getArg1();

    // check if gradient is tracked
    if (destination->getTrackGradient()) {

        // compute gradient store in tensor's gradient attribute
        if (currentTensor->isGradientSet()) {

            // transpose gradient and store in destination
            currentTensor->handleError(cudaMemDup(currentTensor->getGradient(), destination->getGradient(), currentTensor->getShapeX(), currentTensor->getShapeY(), true), "Error: gradient calculation (operation: transpose) failed");
        
        } else {
            // fill with ones
            currentTensor->handleError(constants(destination->getGradient(), destination->getSize(), 1.0f), "Error: gradient calculation (operation: transpose) failed");
        }

        // set gradientSet flag
        destination->changeGradientSet(true);
    }

}

Tensor* Tensor::transpose() {
    // increment reference count of results args (dependencies)
    this->addReference();

    // cache size
    unsigned int size = this->getSize();

    // allocate memory
    float* d_value = reserveMemoryOnDevice(size);

    // check for errors during allocation
    this->handleError(d_value, "Error: Memory allocsation for Tensor transpose failed");

    // copy mem, transpose during copy process
    this->handleError(cudaMemDup(this->getValue(), d_value, this->getShapeX(), this->getShapeY(), true), "Error: An error occured during transposed copying");

    // disable gradientSet flag
    this->changeGradientSet(false);

    // return new Tensor
    return new Tensor(d_value, {this->getShapeY(), this->getShapeX()}, true, &transposeGradient, this, this->getShape());
}

// #############################
// Printing (value and gradient)
// #############################

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

    // check error
    tensor.handleError(val, "Error: Operation to get host copy of values failed");

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

    // check error
    this->handleError(grad, "Error: Operation to get host copy of gradient failed");

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

// ####################
// Error handling utils
// ####################

void Tensor::handleError(cudaError_t err, std::string errorText) const {
    // only handle if error occured, otherwise do nothing
    if (err != cudaSuccess) {
        // error message was already printed in the error function
        std::cout << "\nCUDA error handled in <Tensor>: " << errorText << "\nressources are freed" << std::endl;

        // free ressources 
        delete this; 
        exit(EXIT_FAILURE);
    }
}

void Tensor::handleError(cublasStatus_t err, std::string errorText) const {
    if (err != CUBLAS_STATUS_SUCCESS) {
        // error message was already printed in the error function
        std::cout << "\nCUDA error handled in <Tensor>: " << errorText << "\nressources are freed" << std::endl;

        // free ressources 
        delete this; 
        exit(EXIT_FAILURE);
    }
}

void Tensor::handleError(float* err, std::string errorText) const {
    // only handle if error occured, otherwise do nothing
    if (err == nullptr) {
        std::cout << "\nCUDA error handled in <Tensor>: " << errorText << "\nressources are freed" << std::endl;

        // free ressources
        delete this; 
        exit(EXIT_FAILURE);
    }
}

void Tensor::handleError(std::string errorText) const {

}

// #############################################
// Destructor and graph based garbage collection
// #############################################

Tensor::~Tensor() {

    // Safety check for null refCount
    if (!refCount) {
        return;
    }

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
    if (this->d_value != nullptr) {
        cudaFree(this->d_value);
        this->d_value = nullptr;
    }

    if (this->getTrackGradient() && this->d_gradient != nullptr) {
        cudaFree(this->d_gradient);
        this->d_gradient = nullptr;
    }

    // decrement activeTensors counter and manages cuBlas handle
    activeTensors--;
    if (activeTensors == 0) {
        destroy_cuBlas();
    } 
}
