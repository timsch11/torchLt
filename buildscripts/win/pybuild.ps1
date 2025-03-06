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

nvcc -c core/cuda/util.cu -o bin/win_amd_x64/pybuild/util.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/Factory.cu -o bin/win_amd_x64/pybuild/factory.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/Tensor.cu -o bin/win_amd_x64/pybuild/tensor.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaDif.cu -o bin/win_amd_x64/pybuild/cudadif.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMath.cu -o bin/win_amd_x64/pybuild/cudamath.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMem.cu -o bin/win_amd_x64/pybuild/cudamem.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaNN.cu -o bin/win_amd_x64/pybuild/cudann.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/optimization/MomentumWrapper.cu -o bin/win_amd_x64/pybuild/momentumwrapper.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/optimization/weightUpdate.cu -o bin/win_amd_x64/pybuild/weightupdate.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/optimization/RMSPropWrapper.cu -o bin/win_amd_x64/pybuild/rmspropwrapper.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/optimization/AdamWrapper.cu -o bin/win_amd_x64/pybuild/adamwrapper.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"

# Create static library
nvcc -lib bin/win_amd_x64/pybuild/util.obj bin/win_amd_x64/pybuild/factory.obj bin/win_amd_x64/pybuild/tensor.obj bin/win_amd_x64/pybuild/cudadif.obj bin/win_amd_x64/pybuild/cudamath.obj bin/win_amd_x64/pybuild/cudamem.obj bin/win_amd_x64/pybuild/cudann.obj bin/win_amd_x64/pybuild/momentumwrapper.obj bin/win_amd_x64/pybuild/rmspropwrapper.obj bin/win_amd_x64/pybuild/weightupdate.obj bin/win_amd_x64/pybuild/adamwrapper.obj -o bin/win_amd_x64/pylib/Tensor.lib

# Create dll
nvcc -lib bin/win_amd_x64/pybuild/util.obj bin/win_amd_x64/pybuild/factory.obj bin/win_amd_x64/pybuild/tensor.obj bin/win_amd_x64/pybuild/cudadif.obj bin/win_amd_x64/pybuild/cudamath.obj bin/win_amd_x64/pybuild/cudamem.obj bin/win_amd_x64/pybuild/cudann.obj bin/win_amd_x64/pybuild/momentumwrapper.obj bin/win_amd_x64/pybuild/rmspropwrapper.obj bin/win_amd_x64/pybuild/weightupdate.obj bin/win_amd_x64/pybuild/adamwrapper.obj -Xcompiler "/MD" -o bin/win_amd_x64/pylib/Tensor.dll

$pythonPath = & python -c "import sys; print(sys.executable)"

& $pythonPath setup.py build_ext --inplace

# go back to inital dir
Set-Location $INITIAL_LOCATION