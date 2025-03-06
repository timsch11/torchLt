# change dir but save current dir to go back later
$INITIAL_LOCATION = Get-Location
Set-Location $PSScriptRoot

# go to project folder
Set-Location ..
Set-Location ..

# Make bin directory
mkdir bin -Force
Set-Location bin 
mkdir win_amd_x64 -Force
Set-Location win_amd_x64 
mkdir cppbuild -Force
mkdir cpplib -Force
Set-Location ..
Set-Location ..

# Compile source files to objects
$INCLUDE_PATH = "/core"

nvcc -c core/cuda/util.cu -o bin/win_amd_x64/cppbuild/util.obj -I"$INCLUDE_PATH"
nvcc -c core/Factory.cu -o bin/win_amd_x64/cppbuild/factory.obj -I"$INCLUDE_PATH"
nvcc -c core/Tensor.cu -o bin/win_amd_x64/cppbuild/tensor.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaDif.cu -o bin/win_amd_x64/cppbuild/cudadif.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMath.cu -o bin/win_amd_x64/cppbuild/cudamath.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMem.cu -o bin/win_amd_x64/cppbuild/cudamem.obj -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaNN.cu -o bin/win_amd_x64/cppbuild/cudann.obj -I"$INCLUDE_PATH"
nvcc -c core/optimization/MomentumWrapper.cu -o bin/win_amd_x64/cppbuild/momentumwrapper.obj -I"$INCLUDE_PATH"
nvcc -c core/optimization/RMSPropWrapper.cu -o bin/win_amd_x64/cppbuild/rmspropwrapper.obj -I"$INCLUDE_PATH"
nvcc -c core/optimization/AdamWrapper.cu -o bin/win_amd_x64/cppbuild/adamwrapper.obj -I"$INCLUDE_PATH"
nvcc -c core/optimization/weightUpdate.cu -o bin/win_amd_x64/cppbuild/weightupdate.obj -I"$INCLUDE_PATH"

# Create static library
nvcc -lib bin/win_amd_x64/cppbuild/util.obj bin/win_amd_x64/cppbuild/factory.obj bin/win_amd_x64/cppbuild/tensor.obj bin/win_amd_x64/cppbuild/cudadif.obj bin/win_amd_x64/cppbuild/cudamath.obj bin/win_amd_x64/cppbuild/cudamem.obj bin/win_amd_x64/cppbuild/cudann.obj bin/win_amd_x64/cppbuild/momentumwrapper.obj bin/win_amd_x64/cppbuild/rmspropwrapper.obj bin/win_amd_x64/cppbuild/adamwrapper.obj bin/win_amd_x64/cppbuild/weightupdate.obj -o bin/win_amd_x64/cpplib/Tensor.lib

# go back to inital dir
Set-Location $INITIAL_LOCATION