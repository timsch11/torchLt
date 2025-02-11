
import unittest
import numpy as np

import sys
import os


# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from Tensor import PyTensor


class TestPyTensor(unittest.TestCase):
    def setUp(self):
        # Common setup for tests
        self.size = 10
        self.arr_a = np.random.rand(self.size, self.size).astype(np.float32)
        self.arr_b = np.random.rand(self.size, self.size).astype(np.float32)
        
   
    def test_edge_cases(self):
        # Test empty tensor creation
        with self.assertRaises(ValueError):
            tensor = PyTensor([], ())
            
        # Test mismatched shape
        wrong_shape = (5, 5)
        with self.assertRaises(ValueError):
            tensor = PyTensor(self.arr_a, wrong_shape)
            
        # Test invalid dimensions
        with self.assertRaises(ValueError):
            tensor = PyTensor(self.arr_a, (0, 0))

        with self.assertRaises(ValueError):
            tensor = PyTensor(self.arr_a, (1, -1))

        with self.assertRaises(ValueError):
            tensor = PyTensor(self.arr_a, tuple([2]))

        with self.assertRaises(ValueError):
            tensor = PyTensor(self.arr_a, (3, 2, 1))
            
    def test_broadcasting(self):
        # Test operations between different sized tensors
        small_arr = np.random.rand(1, self.size).astype(np.float32)
        big_tensor = PyTensor(self.arr_a, self.arr_a.shape)
        small_tensor = PyTensor(small_arr, small_arr.shape)

        print("here")
        
        with self.assertRaises(ValueError):
            result = big_tensor + small_tensor
            
    def test_zero_division(self):
        # Test division by zero handling
        zero_arr = np.zeros((self.size, self.size), dtype=np.float32)
        tensor1 = PyTensor(self.arr_a, self.arr_a.shape)
        tensor2 = PyTensor(zero_arr, zero_arr.shape)
        
        with self.assertRaises(RuntimeError):
            result = tensor1 / tensor2
            
    def test_gradient_edge_cases(self):
        # Test backward on non-scalar
        tensor = PyTensor(self.arr_a, self.arr_a.shape)
        
        with self.assertRaises(RuntimeError):
            tensor.backward()
            
        # Test gradient calculation with disabled tracking
        tensor_no_grad = PyTensor(self.arr_a, self.arr_a.shape, _track_gradient=False)
        with self.assertRaises(RuntimeError):
            tensor_no_grad.backward()
            
    def test_shape_operations(self):
        # Test transpose of non-square matrix
        rect_arr = np.random.rand(2, 3).astype(np.float32)
        tensor = PyTensor(rect_arr, (2, 3))
        transposed = tensor.transpose()
        
        self.assertEqual(transposed.getShapeX(), 3)
        self.assertEqual(transposed.getShapeY(), 2)
        np.testing.assert_array_almost_equal(
            transposed.getValue(),
            rect_arr.T
        )
        
    def test_memory_management(self):
        # Test large tensor creation and deletion
        large_size = 10000
        large_arr = np.random.rand(large_size, large_size).astype(np.float32)
        tensor = PyTensor(large_arr, (large_size, large_size))

        # Force garbage collection
        import gc
        del tensor
        gc.collect()


if __name__ == '__main__':
    unittest.main()