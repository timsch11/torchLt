# change dir but save current dir to go back later
$INITIAL_LOCATION = Get-Location
Set-Location $PSScriptRoot

# go to project folder
Set-Location ..

# Make bin directory
mkdir bin
Set-Location bin 
mkdir pylib
mkdir pybuilt
Set-Location ..

# Compile source files to objects
$INCLUDE_PATH = "/core"

nvcc -c core/cuda/util.cu -o bin/pybuilt/util.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/Factory.cu -o bin/pybuilt/factory.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/Tensor.cu -o bin/pybuilt/tensor.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaDif.cu -o bin/pybuilt/cudadif.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMath.cu -o bin/pybuilt/cudamath.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaMem.cu -o bin/pybuilt/cudamem.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
nvcc -c core/cuda/cudaNN.cu -o bin/pybuilt/cudann.obj -Xcompiler "/MD" -I"$INCLUDE_PATH"

# Create static library
nvcc -lib bin/pybuilt/util.obj bin/pybuilt/factory.obj bin/pybuilt/tensor.obj bin/pybuilt/cudadif.obj bin/pybuilt/cudamath.obj bin/pybuilt/cudamem.obj bin/pybuilt/cudann.obj -o bin/pylib/Tensor.lib

# Create dll
nvcc -lib bin/pybuilt/util.obj bin/pybuilt/factory.obj bin/pybuilt/tensor.obj bin/pybuilt/cudadif.obj bin/pybuilt/cudamath.obj bin/pybuilt/cudamem.obj bin/pybuilt/cudann.obj -Xcompiler "/MD" -o bin/pylib/Tensor.dll


# go back to inital dir
Set-Location $INITIAL_LOCATION

$pythonPath = & python -c "import sys; print(sys.executable)"

& $pythonPath setup.py build_ext --inplace