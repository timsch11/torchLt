import unittest
import numpy as np

import sys
import os


# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from torchLt import PyTensor


class TestPyTensor(unittest.TestCase):
    def setUp(self):
        # Common setup for tests
        self.size = 10
        self.arr_a = np.random.rand(self.size, self.size).astype(np.float32)
        self.arr_b = np.random.rand(self.size, self.size).astype(np.float32)
        
    def test_tensor_creation(self):
        # Test basic tensor creation
        tensor = PyTensor(self.arr_a, self.arr_a.shape)
        np.testing.assert_array_almost_equal(tensor.getValue(), self.arr_a)
        
        # Test shape properties
        self.assertEqual(tensor.getShapeX(), self.size)
        self.assertEqual(tensor.getShapeY(), self.size)
        
        # Test gradient tracking flag
        self.assertTrue(tensor.getTrackGradient())
        tensor_no_grad = PyTensor(self.arr_a, self.arr_a.shape, _track_gradient=False)
        self.assertFalse(tensor_no_grad.getTrackGradient())

    def test_basic_operations(self):
        tensor1 = PyTensor(self.arr_a, self.arr_a.shape)
        tensor2 = PyTensor(self.arr_b, self.arr_b.shape)
        
        # Test addition
        add_result = tensor1 + tensor2
        np.testing.assert_array_almost_equal(
            add_result.getValue(), 
            self.arr_a + self.arr_b
        )
        
        # Test subtraction
        sub_result = tensor1 - tensor2
        np.testing.assert_array_almost_equal(
            sub_result.getValue(), 
            self.arr_a - self.arr_b
        )
        
        # Test hadamard product
        hadamard_result = tensor1 * tensor2
        np.testing.assert_array_almost_equal(
            hadamard_result.getValue(), 
            self.arr_a * self.arr_b
        )
        
        # Test matrix multiplication
        matmul_result = tensor1 @ tensor2
        np.testing.assert_array_almost_equal(
            matmul_result.getValue(), 
            self.arr_a @ self.arr_b
        )

    def test_activation_functions(self):
        tensor = PyTensor(self.arr_a, self.arr_a.shape)
        
        # Test ReLU
        relu_result = tensor.relu()
        np.testing.assert_array_almost_equal(
            relu_result.getValue(),
            np.maximum(0, self.arr_a)
        )
        
        # Test Sigmoid
        sigmoid_result = tensor.sigmoid()
        np.testing.assert_array_almost_equal(
            sigmoid_result.getValue(),
            1 / (1 + np.exp(-self.arr_a))
        )
        
        # Test Tanh
        tanh_result = tensor.tanh()
        np.testing.assert_array_almost_equal(
            tanh_result.getValue(),
            np.tanh(self.arr_a)
        )

    def test_gradient_calculation(self):
        # Test simple gradient calculation
        x = PyTensor([[1.0]], (1,1))
        y = PyTensor([[2.0]], (1,1))
        
        # Calculate gradients for L2 loss
        loss = x.l2(y)
        loss.backward()
        
        # Check gradient value (should be 2*(1-2) = -2)
        np.testing.assert_array_almost_equal(
            x.getGradient(),
            np.array([[-2.0]])
        )

    def test_matrix_operations(self):
        tensor = PyTensor(self.arr_a, self.arr_a.shape)

        # Test transpose
        tensor2 = tensor.transpose()

        np.testing.assert_array_almost_equal(
            tensor2.getValue(),
            self.arr_a.T
        )
        
        # Test slicing
        slice_result = tensor.get(0, 5, 0, 5)
        np.testing.assert_array_almost_equal(
            slice_result.getValue(),
            self.arr_a[0:5, 0:5]
        )

    def test_deep_copy(self):
        original = PyTensor(self.arr_a, self.arr_a.shape)
        copied = original.deepcopy()
        
        # Modify original
        self.arr_a[0,0] = 999.0
        original = PyTensor(self.arr_a, self.arr_a.shape)
        
        # Check that copy is unchanged
        self.assertNotEqual(
            original.getValue()[0,0],
            copied.getValue()[0,0]
        )

    def test_optimization(self):
        # Test SGD optimization
        x = PyTensor([[1.0]], (1,1))
        y = PyTensor([[2.0]], (1,1))
        
        # Single optimization step
        loss = x.l2(y)
        loss.backward()
        x.sgd(lr=0.1)
        
        # Value should have moved towards target
        self.assertTrue(abs(x.getValue()[0,0] - 2.0) < 1.0)

    def test_complex_gradient(self):
        # Test more complex gradient computation
        x = PyTensor([[1.0, 2.0]], (2,1))
        y = PyTensor([[2.0, 3.0]], (2,1))
        
        # Create computation graph
        intermediate = x.sigmoid()
        loss = intermediate.l2(y)
        
        loss.backward()
        
        # Check that x has gradients
        self.assertTrue(x.isGradientSet())
        self.assertIsNotNone(x.getGradient())


if __name__ == '__main__':
    unittest.main()