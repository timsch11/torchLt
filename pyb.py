from cuTensor import *
import numpy as np

tensor = PyTensor(np.array([1, 2]), (2, 1))
print(type(tensor))
print(tensor.getShapeX())
