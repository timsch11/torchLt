from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os
import numpy
from pathlib import Path


CUDA_HOME = Path(os.environ.get('CUDA_PATH'))
PROJECT_ROOT = Path(__file__).parent


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
                    os.path.join(PROJECT_ROOT, 'bin', 'pylib'),
                    os.path.join(PROJECT_ROOT, 'core')
                ]
                )

setup(
    name='cuTensorCpy',
    ext_modules=cythonize([ext])
)
