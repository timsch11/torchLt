from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import numpy
from sys import platform
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent


if platform == "linux" or platform == "linux2":
    # Check for CUDA in standard Linux locations
    CUDA_HOME = Path("/usr/local/cuda")
    if not CUDA_HOME.exists():
        CUDA_HOME = Path("/opt/cuda")
    if not CUDA_HOME.exists():
        CUDA_HOME = Path(os.environ.get('CUDA_HOME', ''))

    ext = Extension('cuTensorCpy',
                sources=['wrapper.pyx'],
                libraries=['Tensor', 'cudart', 'cublas', 'cublasLt'],
                language='c++',
                include_dirs=[
                    os.path.join(CUDA_HOME, 'include'),
                    os.path.join(PROJECT_ROOT, 'core'),
                    numpy.get_include()
                ],
                library_dirs=[
                    os.path.join(CUDA_HOME, 'lib64'),
                    os.path.join(PROJECT_ROOT, 'bin', 'ubuntu_amd_x64', 'pylib'),
                    os.path.join(PROJECT_ROOT, 'core')
                ],
                runtime_library_dirs=[
                    os.path.join(CUDA_HOME, 'lib64'),
                    os.path.join(PROJECT_ROOT, 'bin', 'ubuntu_amd_x64', 'pylib')
                ],
                extra_link_args=[
                    f'-Wl,-rpath,{os.path.join(CUDA_HOME, "lib64")}',
                    f'-Wl,-rpath,{os.path.join(PROJECT_ROOT, "core")}',
                ]
                )

    setup(
        name='cuTensorCpy',
        ext_modules=cythonize([ext])
    )
elif platform == "win32":
    CUDA_HOME = Path(os.environ.get('CUDA_PATH'))

    ext = Extension('cuTensorCpy',
                sources=['wrapper.pyx'],
                libraries=['Tensor', 'cudart', 'cublas', 'cublasLt'],
                language='c++',
                include_dirs=[
                    os.path.join(CUDA_HOME, 'include'),
                    os.path.join(PROJECT_ROOT, 'core'),
                    numpy.get_include()
                ],
                library_dirs=[
                    os.path.join(CUDA_HOME, 'lib', 'x64'),
                    os.path.join(PROJECT_ROOT, 'bin', 'win_amd_x64', 'pylib'),
                    os.path.join(PROJECT_ROOT, 'core')
                ]
                )

    setup(
        name='cuTensorCpy',
        ext_modules=cythonize([ext])
    )
