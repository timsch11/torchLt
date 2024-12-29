cimport cython
import numpy as np
cimport numpy as np

from libcpp.pair cimport pair
from libcpp.utility cimport pair
from libcpp cimport bool  # Add bool type


np.import_array()  # Initialize NumPy C-API


# Import from Tensor library
cdef extern from "Tensor.h":
    cdef cppclass Tensor:
        # Constructors
        Tensor(float* _d_value, pair[unsigned int, unsigned int] _shape, bool _track_gradient) except +
        
        # Basic operations
        float* getValueCPU() const
        float* getGradientCPU() const
        unsigned int getShapeX() const
        unsigned int getShapeY() const

        void backward()

        bool getTrackGradient() const
        bool isLeaf() const
        bool isGradientSet() const
        unsigned int getLowerGraphSize() const
        
        # Math operations
        Tensor* add(Tensor& other)
        Tensor* sub(Tensor& other)
        Tensor* matmul(Tensor& other)
        Tensor* hadamardProduct(Tensor& other)
        
        # Activation functions
        Tensor* relu()
        Tensor* sigmoid()
        Tensor* tanh()
        
        # Printing
        void printValue() const
        void printGradient() const


# Python wrapper class
cdef class PyTensor:
    cdef Tensor* _tensor

    def __cinit__(self, object values, tuple shape):

        cdef np.ndarray[np.float32_t, ndim=1] np_array

        if isinstance(values, np.ndarray):
            np_array = np.ascontiguousarray(values, dtype=np.float32)
        else:
            np_array = np.ascontiguousarray(np.array(values, dtype=np.float32))
        
        # Get pointer to data
        cdef float* c_array = <float*>np_array.data
        
        # Create shape pair
        cdef pair[unsigned int, unsigned int] cpp_shape
        cpp_shape.first = shape[0]
        cpp_shape.second = shape[1]
        
        # Create tensor
        self._tensor = new Tensor(c_array, cpp_shape, True)
        
    def __dealloc__(self):
        if self._tensor != NULL:
            del self._tensor
        
    def backward(self):
        self._tensor.backward()

    def getShapeX(self):
        return self._tensor.getShapeX()

    def getShapeY(self):
        return self._tensor.getShapeY()

    def printValue(self):
        self._tensor.printValue()

    def printGradient(self):
        self._tensor.printGradient()

    # def getValue(self):
      #   return self._tensor.getValueCPU()