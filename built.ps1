# change dir but save current dir to go back later
$INITIAL_LOCATION = Get-Location
Set-Location $PSScriptRoot

# Make built directory
mkdir built
Set-Location built 
mkdir Tensorlib
Set-Location ..

# Compile source files to objects
$INCLUDE_PATH = "/core"

nvcc -c core/cuda/util.cu -o built/Tensorlib/util.obj -I"$INCLUDE_PATH"
nvcc -c core/Factory.cu -o built/Tensorlib/factory.obj -I"$INCLUDE_PATH"
nvcc -c core/Tensor.cu -o built/Tensorlib/tensor.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaDif.cu -o built/Tensorlib/cudadif.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMath.cu -o built/Tensorlib/cudamath.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMem.cu -o built/Tensorlib/cudamem.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaNN.cu -o built/Tensorlib/cudann.obj -I"$INCLUDE_PATH"

# Create static library
nvcc -lib built/Tensorlib/util.obj built/Tensorlib/factory.obj built/Tensorlib/tensor.obj built/Tensorlib/cudadif.obj built/Tensorlib/cudamath.obj built/Tensorlib/cudamem.obj built/Tensorlib/cudann.obj -o Tensor.lib

# go back to inital dir
Set-Location $INITIAL_LOCATION