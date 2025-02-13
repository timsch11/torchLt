from torchLt import PyTensor


class Sequential:

    """
    A class used to represent a Sequential model in a neural network.
    Attributes
    ----------
    layers : list[PyTensor]
        A list of layers that make up the sequential model.
    Methods
    -------
    getParams() -> list:
        Returns a list of parameters of the model.
    forward(X: PyTensor) -> PyTensor:
        Performs a forward pass through the model with input tensor X.
    __call__(X: PyTensor) -> PyTensor:
        Allows the object to be called as a function, performing a forward pass.
    """

    def __init__(self, *layers: list[PyTensor]):
        pass

    def getParams(self) -> list:
        """
        Returns a list of parameters of the model.
        """
        pass

    def paramCount(self) -> int:
        """
        Returns the number of parameters of this Network
        """
    
    def forward(self, X: PyTensor) -> PyTensor:
        """
        Performs a forward pass through the model with input tensor X.
        """
        pass
    
    def __call__(self, X: PyTensor) -> PyTensor:
        """
        Allows the object to be called as a function, performing a forward pass.
        """
        pass


from cuTensorCpy import Sequential