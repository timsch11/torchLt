$INITIAL_LOCATION = Get-Location
Set-Location $PSScriptRoot

# Make build directories
mkdir built -Force
Set-Location built
mkdir Tensorlib -Force
mkdir static_obj -Force
mkdir shared_obj -Force
Set-Location ..

$INCLUDE_PATH = "/core"
$SRC_FILES = @(
    "core/cuda/util.cu",
    "core/Factory.cu",
    "core/Tensor.cu",
    "core/cuda/cudaDif.cu",
    "core/cuda/cudaMath.cu",
    "core/cuda/cudaMem.cu",
    "core/cuda/cudaNN.cu"
)

# Compile static objects
foreach ($src in $SRC_FILES) {
    $obj = "built/static_obj/" + [System.IO.Path]::GetFileNameWithoutExtension($src) + ".obj"
    nvcc -c $src -o $obj -I"$INCLUDE_PATH"
}

# Compile dynamic objects
foreach ($src in $SRC_FILES) {
    $obj = "built/shared_obj/" + [System.IO.Path]::GetFileNameWithoutExtension($src) + ".obj"
    nvcc -c $src -o $obj -Xcompiler "/MD" -I"$INCLUDE_PATH"
}

# Create static library
$STATIC_OBJS = Get-ChildItem "built/static_obj/*.obj" | ForEach-Object { $_.FullName }
nvcc -lib $STATIC_OBJS -o built/Tensorlib/Tensor.lib

# Create dll
$SHARED_OBJS = Get-ChildItem "built/shared_obj/*.obj" | ForEach-Object { $_.FullName }
nvcc -lib $SHARED_OBJS -Xcompiler "/MD" -o built/Tensorlib/Tensor.dll

Set-Location $INITIAL_LOCATION

$pythonPath = & python -c "import sys; print(sys.executable)"
& $pythonPath setup.py build_ext --inplace