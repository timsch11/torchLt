import os
import sys

from pathlib import Path
import numpy as np


# Add DLL search paths
cuda_path = Path(os.environ.get('CUDA_PATH')) #, 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6'))
project_root = Path(os.environ.get('PROJECT_ROOT', Path(__file__).parent))

if sys.platform == 'win32':
    os.add_dll_directory(str(cuda_path / 'bin'))
    os.add_dll_directory(str(project_root))


"""interface for Tensor API"""
class PyTensor:
    def __init__(self, values: np.ndarray, shape: tuple):
        pass

    def __dealloc__(self):
        pass
        
    def backward(self):
        pass

    def getShapeX(self):
        pass

    def getShapeY(self):
        pass

    def printValue(self):
        pass

    def printGradient(self):
        pass


# overwrite interface with actual functions
from cuTensorCpy import *