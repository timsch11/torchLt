import os
import ctypes
from pathlib import Path
import numpy as np
from sys import platform


# Set up the path to the library directory
project_root = Path(__file__).parent.parent


#lib_path = project_root / "bin" / "ubuntu_amd_x64" / "pylib"
core_path = project_root / "core"

if platform == "linux" or platform == "linux2":
    lib_path = os.path.join(project_root, "bin", "ubuntu_amd_x64", "pylib")

    # Check for CUDA in standard Linux locations
    CUDA_HOME = Path("/usr/local/cuda")
    if not CUDA_HOME.exists():
        CUDA_HOME = Path("/opt/cuda")
    if not CUDA_HOME.exists():
        CUDA_HOME = Path(os.environ.get('CUDA_HOME', ''))


    ctypes.cdll.LoadLibrary(os.path.join(CUDA_HOME, "lib64", "libcudart.so"))
    ctypes.cdll.LoadLibrary(os.path.join(CUDA_HOME, "lib64", "libcublas.so"))
    ctypes.cdll.LoadLibrary(os.path.join(CUDA_HOME, "lib64", "libcublasLt.so"))

    # Add the library paths to the dynamic linker search path
    if Path(lib_path).exists():
        ctypes.cdll.LoadLibrary(os.path.join(lib_path, "libTensor.so"))
    elif Path(core_path).exists():
        ctypes.cdll.LoadLibrary(os.path.join(core_path, "libTensor.so"))
    else:
        raise ImportError("Could not find libTensor.so in expected locations")

elif platform == "win32":
    lib_path = os.path.join(project_root, "bin", "win_amd_x64", "pylib")

    # fetch cuda path from env variable
    cuda_path = Path(os.environ.get('CUDA_PATH'))

    # add dlls
    os.add_dll_directory(str(cuda_path / 'bin'))
    os.add_dll_directory(str(project_root))

else: 
    print("Error: So far only windows and most linux distros are supported")
    exit()


"""interface for PyTensor API"""
class PyTensor:
    def __init__(self, values: np.ndarray = [], shape: tuple = (), _track_gradient: bool = True, xavierInit: bool = False, kaimingHeInit: bool = False, seed: int = 0):
        """
        Initialize a PyTensor object.

        Args:
            values (np.ndarray, optional): Initial values for the PyTensor. Defaults to empty list.
            shape (tuple, optional): Shape of the PyTensor (rows, columns). Defaults to empty tuple.
            _track_gradient (bool, optional): Whether to track gradients for backpropagation. Defaults to True.
            xavierInit (bool, optional): Use Xavier initialization for weights. Defaults to False.
            kaimingHeInit (bool, optional): Use Kaiming He initialization for weights. Defaults to False.
            seed: Seed that is to be used for weight initalization (if chosen), default: random seed

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

    def dot(self, other: 'PyTensor') -> 'PyTensor':
        """
        Dot product with another PyTensor.

        Args:
            other (Tensor): PyTensor

        Returns:
            PyTensor: New PyTensor containing dot product

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

    def l2backprop(self, other: 'PyTensor') -> 'PyTensor':
        """
        Loss wrapper for backpropagation: Speeds up backpropagation if the actual loss values is of no interest

        Args:
            other (Tensor): PyTensor to represent truth-value

        Returns:
            PyTensor

        Raises:
            TypeError: If other is not a PyTensor
        """
        pass

    def categoricalCrossEntropy(self, other: 'PyTensor') -> 'PyTensor':
        """
        categorical cross entropy: <self> represents prediction, <other> represents actual values

        Args:
            other (Tensor): PyTensor to represent truth-value

        Returns:
            PyTensor: New PyTensor containing cross entropy loss

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

    def softmax(self) -> 'PyTensor':
        """
        Apply softmax function to the vector.

        Returns:
            PyTensor: New PyTensor with softmax applied.
        """
        pass

    def transpose(self) -> 'PyTensor':
        """
        Returns new Tensor holding the transpose of this Tensor
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

    def sgd(lr: float) -> 'PyTensor':
        """Applies a stochastic gradient descent step to this tensor. Requires the gradient to be set."""
        pass

    def synchronize():
        """Waits until weight updates completed."""
        pass

    def deepcopy(self) -> 'PyTensor':
        """Returns a deepcopy without gradient data"""
        pass

    @staticmethod
    def initCuda():
        """Initalizes the cuda context, called automatically when importing library."""
        pass


# overwrite interface with actual functions
from cuTensorCpy import PyTensor


PyTensor.initCuda()


class AsyncOP():
    """
    Context manager to perform asynchronous operations: Synchronizes all operations on exit.
    """


from cuTensorCpy import AsyncOP