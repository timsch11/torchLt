# Core Package Documentation

## Overview
The core package contains the fundamental tensor operations and neural network primitives implemented using CUDA for GPU acceleration. This package provides the backbone for building and training neural networks with efficient parallel computation.

## Key Components

### Tensor Class
The main class implementing tensor operations with GPU support. Located in `Tensor.h` and `Tensor.cu`.

Key features:
- GPU-accelerated tensor operations
- Automatic differentiation support
- Dynamic computational graph construction
- CUDA stream management for concurrent execution

### CUDA Operations
Split across multiple files for better organization:

#### cudaMath.cu
- Basic mathematical operations (addition, subtraction, multiplication)
- Matrix multiplication using cuBLAS
- Element-wise operations (Hadamard product)

#### cudaMem.cu
- Memory management utilities
- Weight initialization functions
- Device memory allocation/deallocation

#### cudaDif.cu
- Gradient computation kernels
- Backpropagation support functions

#### cudaNN.cu
- Neural network specific operations
- Activation functions (ReLU)
- Weight update mechanisms

## Usage Example
```cpp
// Create tensors on GPU
float* mem = constants(4, 2);
Tensor* t1 = new Tensor(mem, {2, 2}, true);

// Perform operations
Tensor* t2 = t1->relu();
t2->backward(); // Compute gradients

// Clean up
delete t1;
delete t2;
```

## Dependencies
- CUDA Toolkit
- cuBLAS library
- C++11 or higher

## Error Handling
All CUDA operations are wrapped with error checking macros. Runtime errors will throw exceptions with detailed error messages.

## Performance Considerations
- Operations are optimized for GPU execution
- Memory transfers between host and device are minimized
- CUDA streams enable concurrent execution where possible

## Contributing
When adding new features:
1. Follow existing error handling patterns
2. Document CUDA kernels and functions
3. Consider memory management implications
4. Add appropriate unit tests
