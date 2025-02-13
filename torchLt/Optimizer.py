from torchLt import PyTensor


class SGD():
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


from cuTensorCpy import SGD