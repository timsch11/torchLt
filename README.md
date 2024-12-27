# cuTensor Library

A high-performance tensor computation library implemented in CUDA C++ for GPU acceleration.

# Note:

Even though the core library is largely done, there are still some topics to address:
- optimize synchronization
- some more testing

Further feature I want to add in the future:
- cython wrapper
- loss functions

## Overview

This library provides a robust implementation of tensor operations with automatic differentiation support, making it suitable for deep learning applications. The library is built with CUDA to leverage GPU acceleration for compute-intensive operations.

## Key Features

- GPU-accelerated tensor operations
- Automatic differentiation by computational graph tracking
- Dynamic memory management
- Tensor math operations
- Common neural network operations:
    - Activation functions (ReLU, Sigmoid, Tanh)
    - Weight initialization (Kaiming He, Xavier)
 
## Tensor Operations:

- matrix multiplication (cuBLAS sgemm)
- addition/subtraction (custom kernels)
- hadamard product (custom kernel)
- activation functions (custom kernels)
- several memory operations like transpose copying (custom kernels)

## Requirements

- CUDA Toolkit
- C++11 or higher
- cuBLAS library

## Building

Use the provided PowerShell build script:

```powershell
./built.ps1
```

This will:
1. Create necessary directories
2. Compile all CUDA source files
3. Generate static library

## Usage Example

```cpp
// Initialize library
init();

// Create tensors
float data1[6] = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
float data2[6] = {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f};

Tensor* t1 = createTensorFromHost(data1, {3, 2}, true);
Tensor* t2 = createTensorFromHost(data2, {3, 2}, true);

// Perform operations
Tensor* t3 = *t1 - *t2;
t3->backward();  // Compute gradients

// Clean up
delete t1;
delete t2;
delete t3;
```

## How to compile your files

```
nvcc example.cu -o example.exe -lTensor -lcublas
```

## Project Structure

- `core/`: Core implementation files
    - `Tensor.h/cu`: Main tensor class implementation
    - `Factory.h/cu`: Tensor creation utilities
    - `cuda/`: CUDA kernel implementations
        - `cudaMath.cuh/cu`: Mathematical operations
        - `cudaMem.cuh/cu`: Memory management
        - `cudaDif.cuh/cu`: Gradient computation
        - `cudaNN.cuh/cu`: Neural network operations
        - `util.cuh/cu`: Utility functions
