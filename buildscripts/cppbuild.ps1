# change dir but save current dir to go back later
$INITIAL_LOCATION = Get-Location
Set-Location $PSScriptRoot

# go to project folder
Set-Location ..

# Make bin directory
mkdir bin -Force
Set-Location bin 
mkdir cppbuild -Force
mkdir cpplib -Force
Set-Location ..

# Compile source files to objects
$INCLUDE_PATH = "/core"

nvcc -c core/cuda/util.cu -o bin/cppbuild/util.obj -I"$INCLUDE_PATH"
nvcc -c core/Factory.cu -o bin/cppbuild/factory.obj -I"$INCLUDE_PATH"
nvcc -c core/Tensor.cu -o bin/cppbuild/tensor.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaDif.cu -o bin/cppbuild/cudadif.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMath.cu -o bin/cppbuild/cudamath.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMem.cu -o bin/cppbuild/cudamem.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaNN.cu -o bin/cppbuild/cudann.obj -I"$INCLUDE_PATH"
nvcc -c core/optimization/MomentumWrapper.cu -o bin/cppbuild/momentumwrapper.obj -I"$INCLUDE_PATH"
nvcc -c core/optimization/RMSPropWrapper.cu -o bin/cppbuild/rmspropwrapper.obj -I"$INCLUDE_PATH"
nvcc -c core/optimization/AdamWrapper.cu -o bin/cppbuild/adamwrapper.obj -I"$INCLUDE_PATH"
nvcc -c core/optimization/weightUpdate.cu -o bin/cppbuild/weightupdate.obj -I"$INCLUDE_PATH"

# Create static library
nvcc -lib bin/cppbuild/util.obj bin/cppbuild/factory.obj bin/cppbuild/tensor.obj bin/cppbuild/cudadif.obj bin/cppbuild/cudamath.obj bin/cppbuild/cudamem.obj bin/cppbuild/cudann.obj bin/cppbuild/momentumwrapper.obj bin/cppbuild/rmspropwrapper.obj bin/cppbuild/adamwrapper.obj bin/cppbuild/weightupdate.obj -o bin/cpplib/Tensor.lib

# go back to inital dir
Set-Location $INITIAL_LOCATION