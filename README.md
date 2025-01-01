# cuTensor

cuTensor is a CUDA-accelerated tensor library designed for high-performance numerical computations. It provides a Python interface for tensor operations, leveraging the power of NVIDIA GPUs to perform efficient mathematical operations and automatic differentiation.

# Note: still in development

Even though the core library is largely done, some operations (namely matmul) are still buggy which I will address in the future

Further features I want to add in the future:
- loss functions

## Features

- **High Performance**: Utilizes CUDA for GPU acceleration.
- **Automatic Differentiation**: Supports backpropagation for gradient computation.
- **Accessible from Python**: Cython bindings make Tensor accessible from python
- **Flexible Initialization**: Supports Xavier and Kaiming He initialization.
- **Comprehensive Tensor Operations**: Includes basic arithmetic, activation functions, and more.

## Installation

### Prerequisites

- NVIDIA CUDA Toolkit
- Python 3.x
- Cython
- NumPy

### Building

Use this PowerShell build script for the python wrapper:

```powershell
./pybuilt.ps1
```

*Use this if you want to only use the python wrapper*

This will:
1. Create necessary directories
2. Compile all CUDA source files
3. Create static and dynamic libraries
4. Setup python wrapper

Use this PowerShell build script to build the c++ library:

```powershell
./cppbuilt.ps1
```

*Use this if you want to only use the c++ library*

This will:
1. Create necessary directories
2. Compile all CUDA source files
3. Create static library

## Usage Example

*Take a look at example.cu or example.py*

## How to compile your c++ files

```
nvcc example.cu -o example.exe -lTensor -lcublas
```
(Take care of you include path or store Tensor.lib in the same directory as your file)

## Project Structure

**Cuda**

- `core/`: Core implementation files
    - `Tensor.h/cu`: Main tensor class implementation
    - `Factory.h/cu`: Tensor creation utilities
    - `cuda/`: CUDA kernel implementations
        - `cudaMath.cuh/cu`: Mathematical operations
        - `cudaMem.cuh/cu`: Memory management
        - `cudaDif.cuh/cu`: Gradient computation
        - `cudaNN.cuh/cu`: Neural network operations
        - `util.cuh/cu`: Utility functions

**Python**
- `Tensor`: Interface for using the Tensor API
