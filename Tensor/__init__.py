import os
import sys

from pathlib import Path
import numpy as np


# Add DLL search paths
cuda_path = Path(os.environ.get('CUDA_PATH'))
project_root = Path(__file__).parent

if sys.platform == 'win32':
    os.add_dll_directory(str(cuda_path / 'bin'))
    os.add_dll_directory(str(project_root))


"""interface for PyTensor API"""
class PyTensor:
    def __init__(self, values: np.ndarray = [], shape: tuple = (), _track_gradient: bool = True, xavierInit: bool = False, kaimingHeInit: bool = False):
        """
        Initialize a PyTensor object.

        Args:
            values (np.ndarray, optional): Initial values for the PyTensor. Defaults to empty list.
            shape (tuple, optional): Shape of the PyTensor (rows, columns). Defaults to empty tuple.
            _track_gradient (bool, optional): Whether to track gradients for backpropagation. Defaults to True.
            xavierInit (bool, optional): Use Xavier initialization for weights. Defaults to False.
            kaimingHeInit (bool, optional): Use Kaiming He initialization for weights. Defaults to False.

        Raises:
            ValueError: If shape is invalid or initialization parameters conflict
        """
        pass

    def __dealloc__(self):
        pass
        
    def backward(self):
        """
        Compute gradients through backpropagation.
        """
        pass

    def getShapeX(self) -> int:
        """
        Get the number of columns in the PyTensor.

        Returns:
            int: Number of columns
        """
        pass

    def getShapeY(self) -> int:
        """
        Get the number of rows in the PyTensor.

        Returns:
            int: Number of rows
        """
        pass

    def printValue(self):
        """
        Print the PyTensor values to console.
        """
        pass

    def printGradient(self):
        """
        Print the gradient values to console.
        """
        pass

    def getTrackGradient(self) -> bool:
        """
        Check if PyTensor is tracking gradients.

        Returns:
            bool: True if PyTensor is tracking gradients, False otherwise
        """
        pass

    def isLeaf(self) -> bool:
        """
        Check if PyTensor is a leaf node in computational graph.

        Returns:
            bool: True if PyTensor is a leaf node, False otherwise
        """
        pass

    def isGradientSet(self) -> bool:
        """
        Check if gradient has been computed for this PyTensor.

        Returns:
            bool: True if gradient exists, False otherwise
        """
        pass

    def getLowerGraphSize(self) -> int:
        """
        Get the number of nodes in the computational graph below this PyTensor.

        Returns:
            int: Number of dependent nodes
        """
        pass

    def getValue(self) -> np.ndarray:
        """
        Get the PyTensor values as a numpy array.

        Returns:
            np.ndarray: PyTensor values in CPU memory
        """
        pass

    def getGradient(self) -> np.ndarray:
        """
        Get the gradient values as a numpy array.

        Returns:
            np.ndarray: Gradient values in CPU memory
        """
        pass

    def add(self, other: 'PyTensor') -> 'PyTensor':
        """
        Element-wise addition with another PyTensor.

        Args:
            other (Tensor): PyTensor to add

        Returns:
            PyTensor: New PyTensor containing element-wise sum

        Raises:
            TypeError: If other is not a PyTensor
        """
        pass

    def sub(self, other: 'PyTensor') -> 'PyTensor':
        """
        Element-wise subtraction with another PyTensor.

        Args:
            other (Tensor): PyTensor to subtract

        Returns:
            PyTensor: New PyTensor containing element-wise difference

        Raises:
            TypeError: If other is not a PyTensor
        """
        pass

    def hadamard(self, other: 'PyTensor') -> 'PyTensor':
        """
        Element-wise multiplication with another PyTensor.

        Args:
            other (Tensor): PyTensor to multiply element-wise

        Returns:
            PyTensor: New PyTensor containing Hadamard product

        Raises:
            TypeError: If other is not a PyTensor
        """
        pass

    def matmul(self, other: 'PyTensor') -> 'PyTensor':
        """
        Matrix multiplication with another PyTensor.

        Args:
            other (Tensor): PyTensor to multiply with

        Returns:
            PyTensor: New PyTensor containing matrix product

        Raises:
            TypeError: If other is not a PyTensor
        """
        pass

    def l2(self, other: 'PyTensor') -> 'PyTensor':
        """
        L2 Loss: <self> represents prediction, <other> represents actual values

        Args:
            other (Tensor): PyTensor to represent truth-value

        Returns:
            PyTensor: New PyTensor containing L2 Loss

        Raises:
            TypeError: If other is not a PyTensor
        """
        pass

    def relu(self) -> 'PyTensor':
        """
        Apply ReLU activation function element-wise.

        Returns:
            PyTensor: New PyTensor with ReLU activation applied
        """
        pass

    def sigmoid(self) -> 'PyTensor':
        """
        Apply sigmoid activation function element-wise.

        Returns:
            PyTensor: New PyTensor with sigmoid activation applied
        """
        pass

    def tanh(self) -> 'PyTensor':
        """
        Apply tanh activation function element-wise.

        Returns:
            PyTensor: New PyTensor with tanh activation applied
        """
        pass

    def transpose(self):
        """
        Transposes tensor in-place
        """

    def get(self, fromRow: int, toRow: int, fromCol: int, toCol: int) -> 'PyTensor':
        """returns the specified sub-Tensor, upper bounds are excluded"""
        pass

    def deepcopy(this) -> 'PyTensor':
        """returns a deepcopy of this Tensor without gradient data"""
        pass

    def __add__(self, other: 'PyTensor') -> 'PyTensor':
        """Operator overload for +"""
        pass

    def __sub__(self, other: 'PyTensor') -> 'PyTensor':
        """Operator overload for -"""
        pass

    def __mul__(self, other: 'PyTensor') -> 'PyTensor':
        """Operator overload for * (Hadamard product)"""
        pass
        
    def __matmul__(self, other: 'PyTensor') -> 'PyTensor':
        """Operator overload for @ (matrix multiplication)"""
        pass

    @staticmethod
    def initCuda():
        """initalizes the cuda context, called automatically when importing library"""
        pass



# overwrite interface with actual functions
from cuTensorCpy import *


PyTensor.initCuda()