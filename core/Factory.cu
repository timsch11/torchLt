#include "Tensor.h"


Tensor* createTensorFromInitFunction(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed, cudaError_t(*initalization_function)(float*, unsigned int, unsigned int, int)) {
    Tensor* obj = nullptr;

    try {
        obj = new Tensor(_shape, _track_gradient, seed, initalization_function);
    } catch(std::runtime_error exc) {
        delete obj;
        
        std::cout << "Error when trying to create Tensor: " << std::string(exc.what()) << "\n";
        exit(EXIT_FAILURE);
    }

    return obj;
}

Tensor* createTensorWithXavierInit(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed) {
    Tensor* obj = nullptr;

    try {
        obj = new Tensor(_shape, _track_gradient, seed, &xavier);
    } catch(std::runtime_error exc) {
        delete obj;
        
        std::cout << "Error when trying to create Tensor: " << std::string(exc.what()) << "\n";
        exit(EXIT_FAILURE);
    }

    return obj;
}

Tensor* createTensorWithKaimingHeInit(std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, int seed) {
    Tensor* obj = nullptr;

    try {
        obj = new Tensor(_shape, _track_gradient, seed, &kaiming_he);
    } catch(std::runtime_error exc) {
        delete obj;
        
        std::cout << "Error when trying to create Tensor: " << std::string(exc.what()) << "\n";
        exit(EXIT_FAILURE);
    }

    return obj;
}

Tensor* createTensorFromDevice(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1) {
    Tensor* obj = nullptr;

    try {
        obj = new Tensor(_d_value, _shape, _track_gradient, _gradFunction, _d_funcArg1, _shapeFuncArg1);
    } catch(std::runtime_error exc) {
        delete obj;
        
        std::cout << "Error when trying to create Tensor: " << std::string(exc.what()) << "\n";
        exit(EXIT_FAILURE);
    }

    return obj;
}

Tensor* createTensorFromDevice(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient, void (*_gradFunction)(Tensor*), Tensor* _d_funcArg1, std::pair<unsigned int, unsigned int> _shapeFuncArg1, Tensor* _d_funcArg2, std::pair<unsigned int, unsigned int> _shapeFuncArg2) {
    Tensor* obj = nullptr;

    try {
        obj = new Tensor(_d_value, _shape, _track_gradient, _gradFunction, _d_funcArg1, _shapeFuncArg1, _d_funcArg2, _shapeFuncArg2);
    } catch(std::runtime_error exc) {
        delete obj;

        std::cout << "Error when trying to create Tensor: " << std::string(exc.what()) << "\n";
        exit(EXIT_FAILURE);
    }

    return obj;
}

Tensor* createTensorFromDevice(float* _d_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient) {
    Tensor* obj = nullptr;

    try {
        obj = new Tensor(_d_value, _shape, _track_gradient);
    } catch(std::runtime_error exc) {
        delete obj;

        std::cout << "Error when trying to create Tensor: " << std::string(exc.what()) << "\n";
        exit(EXIT_FAILURE);
    }

    return obj;
}

Tensor* createTensorFromHost(float* _h_value, std::pair<unsigned int, unsigned int> _shape, bool _track_gradient) {
    float* d_value;
    unsigned int size = _shape.first * _shape.second;

    cudaError_t allocationError = cudaMalloc(&d_value, size * sizeof(float));

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    if (allocationError != cudaSuccess) {
        std::string errorString = "cudaMalloc failed: " + std::string(cudaGetErrorString(allocationError)) + "\n";
        std::cout << errorString;
        exit(EXIT_FAILURE);
    }

    cudaError_t copyError = cudaMemcpy(d_value, _h_value, size * sizeof(float), cudaMemcpyHostToDevice);

    // synchronize before continuing with host code
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (copyError != cudaSuccess) {
        std::string errorString = "cudaMemCpy failed: " + std::string(cudaGetErrorString(copyError)) + "\n";
        cudaFree(d_value);
        std::cout << errorString;
        exit(EXIT_FAILURE);
    }

    Tensor* obj = nullptr;
    try {
        obj = new Tensor(d_value, _shape, _track_gradient);

        return obj;
    } catch(std::runtime_error exc) {
        cudaFree(d_value);
        std::cout << "Error when trying to create Tensor: " << std::string(exc.what()) << "\n";
        exit(EXIT_FAILURE);
    }
}


void init() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices available");
        exit(EXIT_FAILURE);
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("Failed to initialize CUDA device");
        exit(EXIT_FAILURE);
    }
}
