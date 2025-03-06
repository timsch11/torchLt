#!/bin/bash
# Save initial location
INITIAL_LOCATION=$(pwd)

cd "$(dirname "$0")"
cd ..
cd ..

# Make bin directory
mkdir -p bin/ubuntu_amd_x64/pylib
mkdir -p bin/ubuntu_amd_x64/pybuild

# Set include path
INCLUDE_PATH="$INITIAL_LOCATION/core"

# Detect CUDA version and set appropriate architecture flags
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

# Set architecture flags based on CUDA version
# This enables JIT compilation for a wider range of architectures
ARCH_FLAGS="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70"

# For newer CUDA versions, include newer architectures
if [ $CUDA_MAJOR -ge 10 ]; then
    ARCH_FLAGS="$ARCH_FLAGS -gencode arch=compute_75,code=sm_75"
fi
if [ $CUDA_MAJOR -ge 11 ]; then
    ARCH_FLAGS="$ARCH_FLAGS -gencode arch=compute_80,code=sm_80"
fi
if [ $CUDA_MAJOR -ge 12 ]; then
    ARCH_FLAGS="$ARCH_FLAGS -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89"
fi

echo "Building with CUDA $CUDA_VERSION using architecture flags: $ARCH_FLAGS"

# Common compiler flags
COMMON_FLAGS="-Xcompiler \"-fPIC\" -I\"$INCLUDE_PATH\" $ARCH_FLAGS"

# Compile source files to objects
nvcc -c core/cuda/util.cu -o bin/ubuntu_amd_x64/pybuild/util.obj $COMMON_FLAGS
nvcc -c core/Factory.cu -o bin/ubuntu_amd_x64/pybuild/factory.obj $COMMON_FLAGS
nvcc -c core/Tensor.cu -o bin/ubuntu_amd_x64/pybuild/tensor.obj $COMMON_FLAGS
nvcc -c core/cuda/cudaDif.cu -o bin/ubuntu_amd_x64/pybuild/cudadif.obj $COMMON_FLAGS
nvcc -c core/cuda/cudaMath.cu -o bin/ubuntu_amd_x64/pybuild/cudamath.obj $COMMON_FLAGS
nvcc -c core/cuda/cudaMem.cu -o bin/ubuntu_amd_x64/pybuild/cudamem.obj $COMMON_FLAGS
nvcc -c core/cuda/cudaNN.cu -o bin/ubuntu_amd_x64/pybuild/cudann.obj $COMMON_FLAGS
nvcc -c core/optimization/MomentumWrapper.cu -o bin/ubuntu_amd_x64/pybuild/momentumwrapper.obj $COMMON_FLAGS
nvcc -c core/optimization/weightUpdate.cu -o bin/ubuntu_amd_x64/pybuild/weightupdate.obj $COMMON_FLAGS
nvcc -c core/optimization/RMSPropWrapper.cu -o bin/ubuntu_amd_x64/pybuild/rmspropwrapper.obj $COMMON_FLAGS
nvcc -c core/optimization/AdamWrapper.cu -o bin/ubuntu_amd_x64/pybuild/adamwrapper.obj $COMMON_FLAGS

# Create static library
nvcc -lib bin/ubuntu_amd_x64/pybuild/util.obj bin/ubuntu_amd_x64/pybuild/factory.obj bin/ubuntu_amd_x64/pybuild/tensor.obj bin/ubuntu_amd_x64/pybuild/cudadif.obj bin/ubuntu_amd_x64/pybuild/cudamath.obj bin/ubuntu_amd_x64/pybuild/cudamem.obj bin/ubuntu_amd_x64/pybuild/cudann.obj bin/ubuntu_amd_x64/pybuild/momentumwrapper.obj bin/ubuntu_amd_x64/pybuild/rmspropwrapper.obj bin/ubuntu_amd_x64/pybuild/weightupdate.obj bin/ubuntu_amd_x64/pybuild/adamwrapper.obj -o bin/ubuntu_amd_x64/pylib/libTensor.a $ARCH_FLAGS

# Create shared library
#nvcc --shared bin/ubuntu_amd_x64/pybuild/util.obj bin/ubuntu_amd_x64/pybuild/factory.obj bin/ubuntu_amd_x64/pybuild/tensor.obj bin/ubuntu_amd_x64/pybuild/cudadif.obj bin/ubuntu_amd_x64/pybuild/cudamath.obj bin/ubuntu_amd_x64/pybuild/cudamem.obj bin/ubuntu_amd_x64/pybuild/cudann.obj bin/ubuntu_amd_x64/pybuild/momentumwrapper.obj bin/ubuntu_amd_x64/pybuild/rmspropwrapper.obj bin/ubuntu_amd_x64/pybuild/weightupdate.obj bin/ubuntu_amd_x64/pybuild/adamwrapper.obj -Xcompiler "-fPIC" -o bin/ubuntu_amd_x64/pylib/libTensor.so
nvcc --shared bin/ubuntu_amd_x64/pybuild/util.obj bin/ubuntu_amd_x64/pybuild/factory.obj bin/ubuntu_amd_x64/pybuild/tensor.obj bin/ubuntu_amd_x64/pybuild/cudadif.obj bin/ubuntu_amd_x64/pybuild/cudamath.obj bin/ubuntu_amd_x64/pybuild/cudamem.obj bin/ubuntu_amd_x64/pybuild/cudann.obj bin/ubuntu_amd_x64/pybuild/momentumwrapper.obj bin/ubuntu_amd_x64/pybuild/rmspropwrapper.obj bin/ubuntu_amd_x64/pybuild/weightupdate.obj bin/ubuntu_amd_x64/pybuild/adamwrapper.obj -Xcompiler "-fPIC" -o bin/ubuntu_amd_x64/pylib/libTensor.so $ARCH_FLAGS -L"/usr/local/cuda-12.8/lib64" -lcudart -lcublas -lcublasLt

venv/bin/python setup.py build_ext --inplace

# go back to inital dir
cd "$INITIAL_LOCATION"
