from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os
import numpy
from pathlib import Path

from setuptools.command.build_ext import build_ext


CUDA_HOME = Path(os.environ.get('CUDA_PATH')) #, 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'))
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).parent))


ext = Extension('cuTensorCpy',
                sources=['wrapper.pyx'],
                libraries=['Tensor', 'cudart', 'cublas'],
                language='c++',
                include_dirs=[
                    os.path.join(CUDA_HOME, "include"),
                    os.path.join(PROJECT_ROOT, "core"),
                    numpy.get_include()
                ],
                library_dirs=[
                    os.path.join(CUDA_HOME, "lib/x64"),
                    os.path.join(PROJECT_ROOT),
                    os.path.join(PROJECT_ROOT, "core")
                ]
                )

setup(
    name='cuTensorCpy',
    ext_modules=cythonize([ext])
)


"""
# Custom build command for CUDA files
ext = Extension('cudaext',
                sources=['wrapper.pyx'],
                libraries=['Tensor', 'cudart'],
                language='c++',
                include_dirs=[CUDA_HOME + "\\include"],
                library_dirs=[CUDA_HOME + "\\lib\\x64", "C:\\Users\\Tim\\Desktop\\projects\\cupyTensor2\\cudaNN", "C:\\Users\\Tim\\Desktop\\projects\\cupyTensor2\\cudaNN\\core"],
                extra_compile_args=['/openmp'],
                runtime_library_dirs=[
                    os.path.join("C:\\Users\\Tim\\Desktop\\projects\\cupyTensor2\\cudaNN")  # Path where .dll will be at runtime
                ]
                )

setup(
    name='py_wrap',
    include_dirs=[CUDA_HOME + "\\include", "C:\\Users\\Tim\\Desktop\\projects\\cupyTensor2\\cudaNN\\core"],
    ext_modules=[ext],
    cmdclass={'build_ext': build_ext},
)


extra_compile_args = {
    'cl': ['/O2'],
    'nvcc': ['-O3', '--ptxas-options=-v', '-g', '-G'],
}

ext_modules = cythonize(
    Extension(
        "cuda_wrapper",
        sources=["wrapper.pyx"],
        libraries=["cudart"],
        extra_compile_args=extra_compile_args,
        language="c++",
        include_dirs=[os.path.join(CUDA_HOME, "include"),
                      numpy.get_include()],
        library_dirs=[os.path.join(CUDA_HOME, "lib\\x64")],
    )
)

setup(
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    # cmdclass={'build_ext': CUDA_build_ext}
)"""