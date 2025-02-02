import numpy as np
cimport numpy as np

from libcpp.pair cimport pair
from libcpp.utility cimport pair
from libcpp cimport bool  # Add bool type
from libc.string cimport memcpy
from libc.stdio cimport printf

np.import_array()  # Initialize NumPy C-API

# Import from Factory
cdef extern from "Factory.h":
    Tensor* createTensorFromHost(float* _h_value, pair[unsigned int, unsigned int] _shape, bool _track_gradient)
    void init()


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

        # Loss Functions
        Tensor* l2(Tensor &other)

        # Matrix operations
        void transpose()
        
        # Printing
        void printValue() const
        void printGradient() const


# Python wrapper class
cdef class PyTensor:
    cdef Tensor* _tensor

    def __cinit__(self, object values=[], tuple shape=(), bool _track_gradient=True, bool xavierInit=False, bool kaimingHeInit=False):
        """
        Initialize a Tensor object.

        raise ValueError("cannot create Tensor with value both from param and from initialization function")
        values (object): The initial values for the tensor, can be a numpy array, list, or tuple.
        shape (tuple): The shape of the tensor.
        _track_gradient (bool): Whether to track gradients for this tensor.
        xavierInit (bool): Whether to initialize the tensor using Xavier initialization.
        kaimingHeInit (bool): Whether to initialize the tensor using Kaiming He initialization.
        """

        if len(values) == 0:
            if xavierInit or kaimingHeInit:
                if len(shape) == 0:
                    raise ValueError("Shape must be provided when generating Tensor values using an initialization function.")

                # xavier init  TODO

                # kaimingHe init TODO

            else:
                # init empty Tensor 
                # this is necessary to give the py wrapper the actual (cuda c++) Tensor reference easier
                pass

            return

        if len(shape) == 0:
            raise ValueError("missing shape")
        elif len(shape) > 2:
            raise ValueError("unsupported shape, so far only up to two dimensional tensors are allowed")

        # create Tensor
        cdef np.ndarray[np.float32_t, ndim=1] np_array

        if isinstance(values, np.ndarray):
            np_array = np.ascontiguousarray(values.flatten(), dtype=np.float32)
        elif type(values) in {list, tuple}:
            np_array = np.ascontiguousarray(np.array(values, dtype=np.float32).flatten())
        else:
            raise ValueError("invalid type for value: must be either np.ndarray or python list or tuple")
        
        # Get pointer to data
        cdef float* c_array = <float*>np_array.data
        
        # Create shape pair
        cdef pair[unsigned int, unsigned int] cpp_shape
        cpp_shape.first = shape[0]
        cpp_shape.second = shape[1]
        
        # Create tensor
        self._tensor = createTensorFromHost(c_array, cpp_shape, _track_gradient)
        
    cdef initOneDim(self, object values=[], tuple shape=(), bool _track_gradient=True):
        cdef np.ndarray[np.float32_t, ndim=1] np_array

        if isinstance(values, np.ndarray):
            np_array = np.ascontiguousarray(values, dtype=np.float32)
        elif type(values) in {list, tuple}:
            np_array = np.ascontiguousarray(np.array(values, dtype=np.float32))
        else:
            raise ValueError("invalid type for value: must be either np.ndarray or python list or tuple")
        
        # Get pointer to data
        cdef float* c_array = <float*>np_array.data
        
        # Create shape pair
        cdef pair[unsigned int, unsigned int] cpp_shape
        cpp_shape.first = shape[0]
        cpp_shape.second = shape[1]
        
        # Create tensor
        self._tensor = createTensorFromHost(c_array, cpp_shape, _track_gradient)

    cdef initTwoDim(self, object values=[], tuple shape=(), bool _track_gradient=True):
        cdef np.ndarray[np.float32_t, ndim=2] np_array

        if isinstance(values, np.ndarray):
            # Use contiguous array to supply row-major data
            np_array = np.ascontiguousarray(values, dtype=np.float32)
        elif type(values) in {list, tuple}:
            np_array = np.ascontiguousarray(np.array(values, dtype=np.float32))
        else:
            raise ValueError("invalid type for value: must be either np.ndarray, list, or tuple")

        # Get pointer to data (data is now in row-major order)
        cdef float* c_array = <float*>np_array.data

        # Create shape pair (remember: first = rows, second = columns)
        cdef pair[unsigned int, unsigned int] cpp_shape
        cpp_shape.first = shape[0]
        cpp_shape.second = shape[1]

        # Create tensor using the host data
        self._tensor = createTensorFromHost(c_array, cpp_shape, _track_gradient)

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

    def getTrackGradient(self):
        return self._tensor.getTrackGradient()

    def isLeaf(self):
        return self._tensor.isLeaf()

    def isGradientSet(self):
        return self._tensor.isGradientSet()

    def getLowerGraphSize(self):
        return self._tensor.getLowerGraphSize()

    def getValue(self):
        cdef float* data = self._tensor.getValueCPU()
        cdef unsigned int shape_x = self._tensor.getShapeX()
        cdef unsigned int shape_y = self._tensor.getShapeY()

        # Create numpy array from the data
        cdef np.npy_intp dims[1]
        dims[0] = shape_x * shape_y
        
        # Create numpy array and copy data
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT32)
            
        # Copy data into numpy array
        memcpy(np.PyArray_DATA(arr), data, shape_x * shape_y * sizeof(float))

        # free data
        #free(data)

        return arr.reshape(-1, shape_y)
        
        # Create numpy array from the data
        cdef np.npy_intp dims[2]
        dims[0] = shape_x  # First dimension: rows (y)
        dims[1] = shape_y  # Second dimension: columns (x)
        
        # Create numpy array and copy data
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNew(2, dims, np.NPY_FLOAT32)
            
        # Copy data into numpy array
        memcpy(np.PyArray_DATA(arr), data, shape_x * shape_y * sizeof(float))

        # free data
        #free(data)

        return arr

    def getGradient(self):
        if not self._tensor.getTrackGradient():
            return None

        cdef float* data = self._tensor.getGradientCPU()
        cdef unsigned int shape_x = self._tensor.getShapeX()
        cdef unsigned int shape_y = self._tensor.getShapeY()
        
        # Create numpy array from the data
        cdef np.npy_intp dims[2]
        dims[0] = shape_x  # First dimension: rows (y)
        dims[1] = shape_y  # Second dimension: columns (x)
        
        # Create numpy array and copy data
        cdef np.ndarray arr
        if shape_x == 1:
            arr = np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT32)
        if shape_y == 1:
            arr = np.PyArray_SimpleNew(2, dims, np.NPY_FLOAT32)
        else:
            arr = np.PyArray_SimpleNew(2, dims, np.NPY_FLOAT32)
            
        # Copy data into numpy array
        memcpy(np.PyArray_DATA(arr), data, shape_x * shape_y * sizeof(float))
        
        return arr

    """math operations"""

    def add(self, PyTensor other):
        # check for correct type
        if not isinstance(other, PyTensor):
            raise TypeError("operation is only defined for other Tensors")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.add(other._tensor[0])

        return result

    def sub(self, PyTensor other):
        # check for correct type
        if not isinstance(other, PyTensor):
            raise TypeError("operation is only defined for other Tensors")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.sub(other._tensor[0])

        return result

    def hadamard(self, PyTensor other):
        # check for correct type
        if not isinstance(other, PyTensor):
            raise TypeError("operation is only defined for other Tensors")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.hadamardProduct(other._tensor[0])

        return result

    def matmul(self, PyTensor other):
        # check for correct type
        if not isinstance(other, PyTensor):
            raise TypeError("operation is only defined for other Tensors")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.matmul(other._tensor[0])

        return result

    """activation functions"""

    def relu(self):

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.relu()

        return result

    def sigmoid(self):

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.sigmoid()

        return result

    def tanh(self):

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.tanh()

        return result

    """loss functions"""

    def l2(self, PyTensor other):
        # check for correct type
        if not isinstance(other, PyTensor):
            raise TypeError("operation is only defined for other Tensors")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.l2(other._tensor[0])

        return result

    """matrix operations"""

    def transpose(self):
        self._tensor.transpose()

    """operator overloading"""
    
    def __add__(self, PyTensor other):
        return self.add(other)

    def __sub__(self, PyTensor other):
        return self.sub(other)

    def __mul__(self, PyTensor other):
        return self.hadamard(other)
        
    def __matmul__(self, PyTensor other):
        return self.matmul(other)

    """init cuda"""

    @staticmethod
    def initCuda():
        init()
