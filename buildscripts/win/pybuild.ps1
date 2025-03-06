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
mkdir pybuild -Force
mkdir pylib -Force
Set-Location ..
Set-Location ..

# Compile source files to objects
$INCLUDE_PATH = "/core"

nvcc -c core/cuda/util.cu -o bin/pybuild/util.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/Factory.cu -o bin/pybuild/factory.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/Tensor.cu -o bin/pybuild/tensor.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaDif.cu -o bin/pybuild/cudadif.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMath.cu -o bin/pybuild/cudamath.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMem.cu -o bin/pybuild/cudamem.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaNN.cu -o bin/pybuild/cudann.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/optimization/MomentumWrapper.cu -o bin/pybuild/momentumwrapper.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/optimization/weightUpdate.cu -o bin/pybuild/weightupdate.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/optimization/RMSPropWrapper.cu -o bin/pybuild/rmspropwrapper.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/optimization/AdamWrapper.cu -o bin/pybuild/adamwrapper.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"

# Create static library
nvcc -lib bin/pybuild/util.obj bin/pybuild/factory.obj bin/pybuild/tensor.obj bin/pybuild/cudadif.obj bin/pybuild/cudamath.obj bin/pybuild/cudamem.obj bin/pybuild/cudann.obj bin/pybuild/momentumwrapper.obj bin/pybuild/rmspropwrapper.obj bin/pybuild/weightupdate.obj bin/pybuild/adamwrapper.obj -o bin/pylib/Tensor.lib

# Create dll
nvcc -lib bin/pybuild/util.obj bin/pybuild/factory.obj bin/pybuild/tensor.obj bin/pybuild/cudadif.obj bin/pybuild/cudamath.obj bin/pybuild/cudamem.obj bin/pybuild/cudann.obj bin/pybuild/momentumwrapper.obj bin/pybuild/rmspropwrapper.obj bin/pybuild/weightupdate.obj bin/pybuild/adamwrapper.obj -Xcompiler "/MD" -o bin/pylib/Tensor.dll

$pythonPath = & python -c "import sys; print(sys.executable)"

& $pythonPath setup.py build_ext --inplace

# go back to inital dir
Set-Location $INITIAL_LOCATION