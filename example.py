from cuTensor import *
import numpy as np

import time


SIZE = 1000000

arr_a = np.random.rand(SIZE)
arr_b = np.random.rand(SIZE)
print("python done...")
tensor = PyTensor(arr_a, (SIZE, 1))
tensor2 = PyTensor(arr_b, (SIZE, 1))


print("starting...")

#tensor.hadamard(tensor2)
#tensor.printValue()

# t = time.time()
# c = np.multiply(arr_a, arr_b)
# print("numpy needed: ", time.time() - t)

start_time = time.time()
tensor3 = tensor * tensor2
end_time = time.time()
print(end_time - start_time)
print(f"Tensor size: {SIZE}")
print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")
#tensor3.printValue()

#tensor4 = tensor.tanh()
#tensor4.printValue()

#tensor3.printValue()