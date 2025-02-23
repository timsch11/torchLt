# torchLt - A CUDA-Accelerated Neural Network Library

torchLt is a high-performance neural network library that leverages CUDA for GPU acceleration. Its main objective is to be a fast neural network framework that outperforms other libraries like pytorch in many standard use cases.

## Overview

The framework's core consists of a Tensor library written in pure cuda c++ that supports automatic differentiation and various neural network operations. A seperate optimization library implements many state of the art optimizers for training neural networks. The cython wrapper utilizes the two mentioned sub-libraries to build and train efficient and flexible neural networks from python.

## Features

- GPU-accelerated tensor operations
- Automatic differentiation
- Common neural network layers and activations
- Python-friendly API
- Xavier and Kaiming He weight initialization
- Multiple optimization algorithms

## Project Structure

### Core CUDA Components (`/core/`)
- `Tensor.h/cu`: Core tensor implementation with GPU support
- `Factory.h/cu`: Factory methods for tensor creation
- `cuda/`: CUDA kernel implementations
    - `cudaNN.cuh/cu`: Neural network operations
    - `cudaMath.cuh/cu`: Basic math operations
    - `cudaMem.cuh/cu`: Memory management utils
    - `cudaDif.cuh/cu`: Automatic differentiation
    - `util.cuh/cu`: Error handling and other utils
  - `optimization/`: CUDA kernel implementations
    - `MomentumWrapper.cuh/cu`: Momentum Optimizer
    - `RMSPropWrapper.cuh/cu`: RMSProp Optimizer
    - `AdamWrapper.cuh/cu`: Adam Optimizer
    - `weightUpdate.cuh/cu`: Kernels used by Optimizers

### Cython wrapper
- `setup.py`: Build configuration for Python extension
- `wrapper.pyx`: Cython wrapper bridging C++ and Python: Implements the neural network framework with calls into the Cuda C++ libraries

### Python Interface: torchLt
The following parts are mainly used for IDE support, the actual functionality is imported from the cython wrapper.
- `__init__.py`: Neccessary Cuda configurations and Tensor API (with IDE descriptions)
- `Model.py`: Model API (with IDE descriptions)
- `Layer.py`: Layer API (with IDE descriptions)
- `Optimizer.py`: Optimizer API (with IDE descriptions)

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

Build C++ library (if you want to use the c++ library):
```powershell
.\buildscripts\cppbuild.ps1  # Windows
```

Build Python extension (if you want to use the python API):
```powershell
.\buildscripts\pybuild.ps1   # Windows
```

## Usage Example

```python
import torchLt
import torchLt.Layer
import torchLt.Model
import torchLt.Optimizer


model = torchLt.Model.Sequential(torchLt.Layer.Linear(1, 10), torchLt.Layer.Relu(), torchLt.Layer.Linear(10, 10), torchLt.Layer.Relu(), torchLt.Layer.Linear(10, 1))

inp = torchLt.PyTensor([1], shape=(1, 1))
label = torchLt.PyTensor([100000], shape=(1, 1))

optimizer = torchLt.Optimizer.Momentum(model.getParams(), lr=0.01, beta=0.5)

for i in range(30):
    y = model(inp)
    loss = y.l2(label)
    loss.printValue()
    loss.backward()

    optimizer.asyncstep()

```

## Performance Benchmarks

![grafik](https://github.com/user-attachments/assets/c3c6c609-1389-4530-80b3-09dc57141909)

Performance differed between optimizers: torchLt outperformed pytorch by almost 100% for Adam optimizer (and by a less but still significant portion for RMSProp and Momentum), however, pytorch training was faster with Stochastic Gradient Descent.

Proceeding: Each framework trained an equal neural network with sizes from 25 to 100 million parameters with own synthetically generated data. Result is averaged training time from three seperate trainings. Benchmark was performed for different optimizers.

Note: To be fair I have to mention that this library does not yet support mini batches yet, therefore benchmarks were conducted without mini batches (=batch size of 1). This is a marginal disadvantage for pytorch which requires zeroing the gradient after each batch. However, this should not significantly change the benchmark results.

Code used for the benchmark: `benchmark-comparison.py`

PyTorch version: 2.6.0 with Cuda 12.6

## Roadmap

- Mini batch support
- Optimize stream usage during backpropagation and optimization

## Developed on:
- Windows 10
- AMD Ryzen 5 3600 x64
- NVIDIA RTX 2070 Super with CUDA 12.6
