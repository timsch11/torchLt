from torchLt import PyTensor


class Linear:

    def __init__(self, neurons_in: int, neurons_out: int, xavierInit: bool = True, kaimingHeInit: bool = False):
        """
        A linear (fully connected) layer for a neural network.
        Args:
            neurons_in (int): The number of input neurons.
            neurons_out (int): The number of output neurons.
            xavierInit (bool, optional): Whether to initialize the weights using Xavier initialization. Default is True.
            kaimingHeInit (bool, optional): Whether to initialize the weights using Kaiming He initialization. Default is False.
        Raises:
            ValueError: If both xavierInit and kaimingHeInit are set to True.
            ValueError: If neither xavierInit nor kaimingHeInit are set to True.
        """
        pass

    def getParams(self) -> list:
        """
        Returns the parameters (weights and biases) of the layer.
        """
        pass
    
    def forward(self, X: PyTensor) -> PyTensor:
        """
        Performs the forward pass of the layer.
        """
        pass
    
    def __call__(self, X: PyTensor) -> PyTensor:
        """
        Calls the forward method. Ensures that the input is of type PyTensor.
        """
        pass
    

class Relu():

    """
    A class representing the Relu (Rectified Linear Unit) activation function.
    Methods:
        __init__():
        getParams() -> list:
        forward(X: PyTensor) -> PyTensor:
        __call__(X: PyTensor) -> PyTensor:
    """

    def __init__(self):
        """
        Initialize the Relu activation function.
        """
        pass

    def getParams(self) -> list:
        """
        Get the parameters of the Relu layer.
        Returns:
            list: An empty list as Relu has no parameters.
        """
        pass
    
    def forward(self, X: PyTensor) -> PyTensor:
        """
        Perform the forward pass using the Relu activation function.
        Args:
            X (PyTensor): The input tensor.
        Returns:
            PyTensor: The output tensor after applying the Relu function.
        """
        pass
    
    def __call__(self, X: PyTensor) -> PyTensor:
        """
        Make the Relu layer callable.
        Args:
            X (PyTensor): The input tensor.
        Returns:
            PyTensor: The output tensor after applying the Relu function.
        """
        pass
    

class Sigmoid():

    """
    A class representing the Sigmoid activation function.
    Methods:
        __init__():
        getParams() -> list:
        forward(X: PyTensor) -> PyTensor:
        __call__(X: PyTensor) -> PyTensor:
    """

    def __init__(self):
        """
        Initialize the Sigmoid activation function.
        """
        pass

    def getParams(self) -> list:
        """
        Get the parameters of the Sigmoid layer.
        Returns:
            list: An empty list as Sigmoid has no parameters.
        """
        pass
    
    def forward(self, X: PyTensor) -> PyTensor:
        """
        Perform the forward pass using the Sigmoid activation function.
        Args:
            X (PyTensor): The input tensor.
        Returns:
            PyTensor: The output tensor after applying the Sigmoid function.
        """
        pass
    
    def __call__(self, X: PyTensor) -> PyTensor:
        """
        Make the Sigmoid layer callable.
        Args:
            X (PyTensor): The input tensor.
        Returns:
            PyTensor: The output tensor after applying the Sigmoid function.
        """
        pass
    

class Tanh():

    """
    A class representing the Tanh activation function.
    Methods:
        __init__():
        getParams() -> list:
        forward(X: PyTensor) -> PyTensor:
        __call__(X: PyTensor) -> PyTensor:
    """

    def __init__(self):
        """
        Initialize the Tanh activation function.
        """
        pass

    def getParams(self) -> list:
        """
        Get the parameters of the Tanh layer.
        Returns:
            list: An empty list as Tanh has no parameters.
        """
        pass
    
    def forward(self, X: PyTensor) -> PyTensor:
        """
        Perform the forward pass using the Tanh activation function.
        Args:
            X (PyTensor): The input tensor.
        Returns:
            PyTensor: The output tensor after applying the Tanh function.
        """
        pass
    
    def __call__(self, X: PyTensor) -> PyTensor:
        """
        Make the Tanh layer callable.
        Args:
            X (PyTensor): The input tensor.
        Returns:
            PyTensor: The output tensor after applying the Tanh function.
        """
        pass


class Softmax():

    """
    A class representing the Softmax activation function.
    Methods:
        __init__():
        getParams() -> list:
        forward(X: PyTensor) -> PyTensor:
        __call__(X: PyTensor) -> PyTensor:
    """

    def __init__(self):
        """
        Initialize the Softmax activation function.
        """
        pass

    def getParams(self) -> list:
        """
        Get the parameters of the Softmax layer.
        Returns:
            list: An empty list as Softmax has no parameters.
        """
        pass
    
    def forward(self, X: PyTensor) -> PyTensor:
        """
        Perform the forward pass using the Softmax activation function.
        Args:
            X (PyTensor): The input tensor.
        Returns:
            PyTensor: The output tensor after applying the Softmax function.
        """
        pass
    
    def __call__(self, X: PyTensor) -> PyTensor:
        """
        Make the Softmax layer callable.
        Args:
            X (PyTensor): The input tensor.
        Returns:
            PyTensor: The output tensor after applying the Softmax function.
        """
        pass
    

# overwrite interface with actual functions
from cuTensorCpy import Linear, Relu, Sigmoid, Tanh, Softmax