# Make built directory
mkdir built

# Compile source files to objects
$INCLUDE_PATH = "/core"

nvcc -c cuda/util.cu -o built/util.obj -I"$INCLUDE_PATH"
nvcc -c Factory.cu -o built/factory.obj -I"$INCLUDE_PATH"
nvcc -c Tensor.cu -o built/tensor.obj -I"$INCLUDE_PATH"
nvcc -c cuda/cudaDif.cu -o built/cudadif.obj -I"$INCLUDE_PATH"
nvcc -c cuda/cudaMath.cu -o built/cudamath.obj -I"$INCLUDE_PATH"
nvcc -c cuda/cudaMem.cu -o built/cudamem.obj -I"$INCLUDE_PATH"
nvcc -c cuda/cudaNN.cu -o built/cudann.obj -I"$INCLUDE_PATH"

# Create static library
nvcc -lib built/util.obj built/factory.obj built/tensor.obj built/cudadif.obj built/cudamath.obj built/cudamem.obj built/cudann.obj -o Tensor.lib