# cudaNN - A CUDA-Accelerated Neural Network Library

cudaNN is a high-performance neural network library that leverages CUDA for GPU acceleration. It provides a Python interface while maintaining the computational efficiency of CUDA C++.

## Overview

The library implements a tensor-based computational framework supporting automatic differentiation and various neural network operations. It's designed for both educational purposes and performance demonstrations of GPU-accelerated machine learning.

## Project Structure

### Core CUDA Components (`/core/`)
- `Tensor.h/cu`: Core tensor implementation with GPU support
- `Factory.h/cu`: Factory methods for tensor creation
- `cuda/`: CUDA kernel implementations
    - `cudaNN.cuh/cu`: Neural network operations
    - `cudaMath.cuh/cu`: Mathematical operations
    - `cudaMem.cuh/cu`: Memory management
    - `cudaDif.cuh/cu`: Automatic differentiation 

### Python Interface
- `wrapper.pyx`: Cython wrapper bridging C++ and Python
- `__init__.py`: Python API interface
- `setup.py`: Build configuration for Python extension

### Examples (`/example/`)
- `nn1.py`: Simple regression example
- `nn2.py`: Performance benchmark example 

## Features

- GPU-accelerated tensor operations
- Automatic differentiation
- Common neural network layers and activations
- Python-friendly API
- Xavier and Kaiming He weight initialization
- Multiple optimization algorithms

## Installation

### Prerequisites
- CUDA Toolkit (11.0+)
- Python 3.7+
- Visual Studio (Windows) or GCC (Linux)
- Cython
- Numpy

### Building

1. Set CUDA environment:
```powershell
$env:CUDA_PATH = "path/to/cuda"  # Windows
```

2. Build C++ library:
```powershell
.\buildscripts\cppbuild.ps1  # Windows
```

3. Build Python extension:
```powershell
.\buildscripts\pybuild.ps1   # Windows
```

## Usage Example

```python
from Tensor import PyTensor


inp = PyTensor([1.2, 3.3], (2, 1))

w0 = PyTensor(shape=(32, 2), _track_gradient=True, kaimingHeInit=True)
b0 = PyTensor([[0] * 32], (32, 1), _track_gradient=True)
    
w1 = PyTensor(shape=(4, 32), _track_gradient=True, kaimingHeInit=True)
b1 = PyTensor([[0] * 4], (4, 1), _track_gradient=True)

labels = PyTensor([1, 2, 3, 4], (4, 1))

# example neural network layers
a0 = ((w0 @ inp) + b0).tanh()
a1 = ((w1 @ a0) + b1).relu()

loss = a1.l2(labels)

loss.backward()

w0.sgd(lr=0.01)
b0.sgd(lr=0.01)
w1.sgd(lr=0.01)
b1.sgd(lr=0.01)


```

## Performance Benchmarks

The `nn2.py` example demonstrates the library's performance with a large neural network:
- ~100 million parameters
- GPU-accelerated forward and backward passes

## License

MIT License

## Acknowledgments

- NVIDIA CUDA Team
- Python and Cython communities
