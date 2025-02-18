from torchLt import PyTensor


class SGD():
    """
    Implements the Stochstic gradient descent optimization algorithm.
    Attributes:
        params (list[PyTensor]): List of parameters to optimize.
        lr (float): Learning rate for the optimizer. Default is 0.01.
    Methods:
        syncstep():
            Performs an optimization step where parameter updates are carried out sequentially.
        asyncstep():
            Performs an optimization step where parameter updates are carried out in parallel. 
            The function terminates after all updates have occurred.
    """
    def __init__(self, params: list[PyTensor], lr=0.01):
        pass

    def syncstep(self):
        """
        Performs an optimization step: Parameter updates are carried out sequentially
        """
        pass

    def asyncstep(self):
        """
        Performs an optimization step: Parameter updates are carried out in parallel. Function terminated after all updates have occured.
        """
        pass


class Momentum():
    """
    Implements the Momentum optimization algorithm.
    Attributes:
        params (list[PyTensor]): List of parameters to optimize.
        lr (float): Learning rate for the optimizer. Default is 0.01.
        beta (float): Momentum factor. Default is 0.9.
    Methods:
        syncstep():
            Performs an optimization step where parameter updates are carried out sequentially.
        asyncstep():
            Performs an optimization step where parameter updates are carried out in parallel. 
            The function terminates after all updates have occurred.
    """
    def __init__(self, params: list[PyTensor], lr=0.01, beta=0.9):
        pass

    def syncstep(self):
        """
        Performs an optimization step: Parameter updates are carried out sequentially
        """
        pass

    def asyncstep(self):
        """
        Performs an optimization step: Parameter updates are carried out in parallel. Function terminated after all updates have occured.
        """
        pass

class RMSProp():
    """
    Implements RMSProp optimization algorithm.
    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate. Default: 0.01.
        alpha (float, optional): Coefficient used for computing running averages of squared gradients. Default: 0.9.
        eps (float, optional): Term added to the denominator to improve numerical stability. Default: 1e-8.
    Methods:
        syncstep():
            Performs an optimization step where parameter updates are carried out sequentially.
        asyncstep():
            Performs an optimization step where parameter updates are carried out in parallel. The function terminates after all updates have occurred.
    """
    def __init__(self, params, lr: float = 0.01, alpha: float = 0.9, eps: float = 0.00000001):
        pass

    def syncstep(self):
        """
        Performs an optimization step: Parameter updates are carried out sequentially
        """
        pass

    def asyncstep(self):
        """
        Performs an optimization step: Parameter updates are carried out in parallel. Function terminated after all updates have occured.
        """
        pass


class Adam():
    """
    Implements the Adam optimization algorithm.
    Attributes:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate. Default: 0.001.
        alpha (float): Coefficient used for computing running averages of squared gradients. Default: 0.99.
        momentum (float): Coefficient used for computing running averages of gradients. Default: 0.9.
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-8.
        Initializes the Adam optimizer with the given parameters.
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate. Default: 0.001.
            alpha (float): Coefficient used for computing running averages of squared gradients. Default: 0.99.
            momentum (float): Coefficient used for computing running averages of gradients. Default: 0.9.
            eps (float): Term added to the denominator to improve numerical stability. Default: 1e-8.
        Performs an optimization step: Parameter updates are carried out sequentially.
        Performs an optimization step: Parameter updates are carried out in parallel. Function terminates after all updates have occurred.
    """
    def __init__(self, params, lr: float = 0.001, alpha: float = 0.99, momentum: float = 0.9, eps: float = 0.00000001):
        pass

    def syncstep(self):
        """
        Performs an optimization step: Parameter updates are carried out sequentially
        """
        pass

    def asyncstep(self):
        """
        Performs an optimization step: Parameter updates are carried out in parallel. Function terminated after all updates have occured.
        """
        pass


from cuTensorCpy import SGD, Momentum, RMSProp, Adam