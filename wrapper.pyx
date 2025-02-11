import numpy as np
cimport numpy as np

from libcpp.pair cimport pair
from libcpp.utility cimport pair
from libcpp cimport bool  # Add bool type
from libc.string cimport memcpy

from libc.stdlib cimport rand


np.import_array()  # Initialize NumPy C-API

# Import from Factory
cdef extern from "Factory.h":
    Tensor* createTensorFromHost(float* _h_value, pair[unsigned int, unsigned int] _shape, bool _track_gradient)
    Tensor* createTensorWithXavierInit(pair[unsigned int, unsigned int] _shape, bool _track_gradient, int seed)
    Tensor* createTensorWithKaimingHeInit(pair[unsigned int, unsigned int] _shape, bool _track_gradient, int seed)
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
        Tensor* dot(Tensor &other)
        Tensor* hadamardProduct(Tensor& other)

        # Value acceses and slicing
        Tensor* getRows(unsigned int fromRow, unsigned int toRow)
        Tensor* getCols(unsigned int fromCol, unsigned int toCol);
        Tensor* getVal(unsigned int row, unsigned int col);
        Tensor* get(unsigned int fromRow, unsigned int toRow, unsigned int fromCol, unsigned int toCol)

        # Copy
        Tensor* deepcopy()

        # Deletion
        int getReferenceCount()
        void removeReference()

        void sgd(float lr)
        
        # Activation functions
        Tensor* relu()
        Tensor* sigmoid()
        Tensor* tanh()

        # Loss Functions
        Tensor* l2(Tensor &other)

        # Matrix operations
        Tensor* transpose()
        
        # Printing
        void printValue() const
        void printGradient() const


# Python wrapper class
cdef class PyTensor:
    cdef Tensor* _tensor

    def __cinit__(self, object values=[], tuple shape=(0, 0), bool _track_gradient=True, bool xavierInit=False, bool kaimingHeInit=False, int seed=0):
        """
        Initialize a Tensor object.

        raise ValueError("cannot create Tensor with value both from param and from initialization function")
        values (object): The initial values for the tensor, can be a numpy array, list, or tuple.
        shape (tuple): The shape of the tensor.
        _track_gradient (bool): Whether to track gradients for this tensor.
        xavierInit (bool): Whether to initialize the tensor using Xavier initialization.
        kaimingHeInit (bool): Whether to initialize the tensor using Kaiming He initialization.
        """

        # Create shape pair
        cdef pair[unsigned int, unsigned int] cpp_shape

        if shape == (0, 0):
            if not values:
                # empty wrapper
                pass

            elif isinstance(values, np.ndarray):
                cpp_shape.first = values.shape[0]
                cpp_shape.second = values.shape[1]

            else:
                raise ValueError("Shapes can only be implicitly given through numpy arrays, provide a numpy array as value or a shape")

        elif len(shape) != 2:
            raise ValueError("Shapes must be two dimensional")

        elif shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("Shapes must be positive non-zero integers")

        else:
            cpp_shape.first = shape[0]
            cpp_shape.second = shape[1]

        # Determine seed
        cdef int randnum = rand() if seed == 0 else seed

        if len(values) == 0:
            if xavierInit or kaimingHeInit:
                if shape == (0, 0):
                    raise ValueError("Shape must be provided when generating Tensor values using an initialization function.")
                elif len(values) > 0:
                    raise ValueError("Tensor can only be initalized with either <values> or <xavierInit> or <kaimingHeInit>. Pick one.")
                elif xavierInit and kaimingHeInit:
                    raise ValueError("Tensor cannot be inialized with xavier and kaimingHe init. Pick up to one.")

                if kaimingHeInit:
                    self._tensor = createTensorWithKaimingHeInit(cpp_shape, _track_gradient, randnum)

                else:
                    self._tensor = createTensorWithXavierInit(cpp_shape, _track_gradient, randnum)

            else:
                # init empty Tensor 
                # this is necessary to later give the py wrapper the actual (cuda c++) Tensor reference easier
                pass

            return

        if len(shape) == 0:
            raise ValueError("missing shape")
        elif len(shape) > 2:
            raise ValueError("unsupported shape, so far only up to two dimensional tensors are allowed")

        # create Tensor
        cdef np.ndarray[np.float32_t, ndim=1] np_array

        if isinstance(values, np.ndarray):
            if shape and len(shape) == 2:
                if ((values.shape[0]) != shape[0]) or ((values.shape[1]) != shape[1]):
                    raise ValueError("Shape of numpy array passed as value does not match provided shape. Provide the correct shape or no shape to implicitly pass the shape.")

            np_array = np.ascontiguousarray(values.flatten(), dtype=np.float32)
    
        elif type(values) in {list, tuple}:
            np_array = np.ascontiguousarray(np.array(values, dtype=np.float32).flatten())
        else:
            raise ValueError("invalid type for value: must be either np.ndarray or python list or tuple")
        
        # Get pointer to data
        cdef float* c_array = <float*>np_array.data
        
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
            self._tensor.removeReference()

    def sameShape(self, other: PyTensor):
        return (self._tensor.getShapeX() == other._tensor.getShapeX()) and (self._tensor.getShapeY() == other._tensor.getShapeY())
        
    def backward(self):
        if self._tensor.getShapeX() != 1 or self._tensor.getShapeY() != 1:
            raise RuntimeError("Backpropagation must start from a scalar value")

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

    def getReferenceCount(self):
        return self._tensor.getReferenceCount()

    def getValue(self):
        cdef float* data = self._tensor.getValueCPU()
        cdef unsigned int shape_x = self._tensor.getShapeX()
        cdef unsigned int shape_y = self._tensor.getShapeY()

        # Create numpy array from the data
        cdef np.npy_intp[1] dims
        dims[0] = shape_x * shape_y
        
        # Create numpy array and copy data
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNew(1, dims, np.NPY_FLOAT32)
            
        # Copy data into numpy array
        memcpy(np.PyArray_DATA(arr), data, shape_x * shape_y * sizeof(float))

        # free data
        #free(data)

        return arr.reshape(-1, shape_y) # -1, shape_y


    def getGradient(self):
        if not self._tensor.getTrackGradient():
            return None

        # load gradient into host memory
        cdef float* data = self._tensor.getGradientCPU()

        # cache shapes
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

        if not self.sameShape(other):
            raise ValueError("Incompatible shapes: Shapes must match for addition to work")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.add(other._tensor[0])

        return result

    def sub(self, PyTensor other):
        # check for correct type
        if not isinstance(other, PyTensor):
            raise TypeError("operation is only defined for other Tensors")

        if not self.sameShape(other):
            raise ValueError("Incompatible shapes: Shapes must match for subtraction to work")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.sub(other._tensor[0])

        return result

    def hadamard(self, PyTensor other):
        # check for correct type
        if not isinstance(other, PyTensor):
            raise TypeError("operation is only defined for other Tensors")

        if not self.sameShape(other):
            raise ValueError("Incompatible shapes: Shapes must match for hadamard product to work")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.hadamardProduct(other._tensor[0])

        return result

    def matmul(self, PyTensor other):
        # check for correct type
        if not isinstance(other, PyTensor):
            raise TypeError("operation is only defined for other Tensors")

        #if self._tensor.getShapeY() != other._tensor.getShapeX():
        #    raise ValueError("Incompatible shapes: #col of A must match #row of B")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.matmul(other._tensor[0])

        return result

    def dot(self, PyTensor other):
        # check for correct type
        if not isinstance(other, PyTensor):
            raise TypeError("operation is only defined for other Tensors")

        if not self.sameShape(other):
            raise ValueError("Incompatible shapes: Shapes must match for dot product to work")
        
        if self._tensor.getShapeY() != 1:
            raise ValueError("Incompatible shapes: Tensors need to be column vectors to perform dot product")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.dot(other._tensor[0])

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

        if not self.sameShape(other):
            raise ValueError("Incompatible shapes: Shapes must match for l2 loss")
        
        if self._tensor.getShapeY() != 1:
            raise ValueError("Incompatible shapes: Tensors need to be column vectors to calculate l2 loss")

        # create empty Tensor wrapper
        result = PyTensor()

        # carry out calculation with actual tensor and wrap result with our empty Tensor wrapper
        result._tensor = self._tensor.l2(other._tensor[0])

        return result

    """matrix operations"""

    def transpose(self):
        # create empty Tensor wrapper
        result = PyTensor()
        
        result._tensor = self._tensor.transpose()

        return result

    """operator overloading"""
    
    def __add__(self, PyTensor other):
        return self.add(other)

    def __sub__(self, PyTensor other):
        return self.sub(other)

    def __mul__(self, PyTensor other):
        return self.hadamard(other)
        
    def __matmul__(self, PyTensor other):
        return self.matmul(other)

    def __truediv__(self, other):
        raise RuntimeError("Division not defined for Tensors")

    # values accesses and slicing

    def __getitem__(self, index):
        # create empty Tensor wrapper
        result = PyTensor()

        if isinstance(index, slice):
            # row slice
            if index.step is not None and index.step != 1:
                raise ValueError("Operation does not support step sizes different than 1 (yet)")

            result._tensor = self._tensor.getRows(index.start, index.stop)

        elif isinstance(index, tuple):
            if len(index) > 2:
                raise ValueError("Operation only supports up to two slices")

            if isinstance(index[0], slice):
                if index[0].step is not None and index[0].step != 1:
                    raise ValueError("Operation does not support step sizes different than 1 (yet)")

                fromRow = index[0].start if index[0].start >= 0 else self.getShapeX() + index[0].start
                toRow = index[0].stop if index[0].stop >= 0 else self.getShapeX() + index[0].stop
            elif isinstance(index[0], int):
                fromRow = index[0] if index[0] >= 0 else self.getShapeX() + index[0]
                toRow = fromRow + 1

            if isinstance(index[1], slice):
                if index[1].step is not None and index[1].step != 1:
                    raise ValueError("Operation does not support step sizes different than 1 (yet)")

                fromCol = index[1].start if index[1].start >= 0 else self.getShapeY() + index[1].start
                toCol = index[1].stop if index[1].stop >= 0 else self.getShapeY() + index[1].stop
            elif isinstance(index[1], int):
                fromCol = index[1] if index[1] >= 0 else self.getShapeY() + index[1]
                toCol = fromCol + 1

            
            result._tensor = self._tensor.get(fromRow, toRow, fromCol, toCol)

        elif isinstance(index, int):
            # Single element indexing (a scalar) -> return row
            result._tensor = self._tensor.getRows(index, index + 1) if index >= 0 else self._tensor.getRows(self.getShapeX() + index, self.getShapeX() + index + 1)

        else: 
            raise TypeError("unsupported type for slicing")

        return result

    """Copy"""

    def deepcopy(self):
        # create empty Tensor wrapper
        result = PyTensor()

        result._tensor = self._tensor.deepcopy()
        return result

    def get(self, fromRow: int, toRow: int, fromCol: int, toCol: int):
        return self[fromRow:toRow, fromCol:toCol]

    """Optimization"""
    
    def sgd(self, lr: float):
        self._tensor.sgd(lr)

    """init cuda"""

    @staticmethod
    def initCuda():
        init()
