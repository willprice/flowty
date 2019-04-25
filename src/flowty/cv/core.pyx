# cython: language_level=3

from cpython cimport Py_buffer
from .c_core cimport Mat as c_Mat
from . cimport c_core
import numpy as np
cimport numpy as np
np.import_array()

CV_8U = c_core.CV_8U
CV_8S = c_core.CV_8S
CV_16U = c_core.CV_16U
CV_16S = c_core.CV_16S
CV_32S = c_core.CV_32S
CV_32F = c_core.CV_32F
CV_64F = c_core.CV_64F

CV_8UC1 = c_core.CV_8UC1
CV_8UC2 = c_core.CV_8UC2
CV_8UC3 = c_core.CV_8UC3
CV_8UC4 = c_core.CV_8UC4

CV_8SC1 = c_core.CV_8SC1
CV_8SC2 = c_core.CV_8SC2
CV_8SC3 = c_core.CV_8SC3
CV_8SC4 = c_core.CV_8SC4

CV_16UC1 = c_core.CV_16UC1
CV_16UC2 = c_core.CV_16UC2
CV_16UC3 = c_core.CV_16UC3
CV_16UC4 = c_core.CV_16UC4

CV_16SC1 = c_core.CV_16SC1
CV_16SC2 = c_core.CV_16SC2
CV_16SC3 = c_core.CV_16SC3
CV_16SC4 = c_core.CV_16SC4

CV_32SC1 = c_core.CV_32SC1
CV_32SC2 = c_core.CV_32SC2
CV_32SC3 = c_core.CV_32SC3
CV_32SC4 = c_core.CV_32SC4

CV_32FC1 = c_core.CV_32FC1
CV_32FC2 = c_core.CV_32FC2
CV_32FC3 = c_core.CV_32FC3
CV_32FC4 = c_core.CV_32FC4

CV_64FC1 = c_core.CV_64FC1
CV_64FC2 = c_core.CV_64FC2
CV_64FC3 = c_core.CV_64FC3
CV_64FC4 = c_core.CV_64FC4

# CV type code -> (byte_count, type_str, channel_count)
# See type_str details at https://docs.python.org/3/library/struct.html#format-characters
_mat_type_lookup = {
    c_core.CV_8UC1: (1, b'B', 1),
    c_core.CV_8UC2: (1, b'B', 2),
    c_core.CV_8UC3: (1, b'B', 3),
    c_core.CV_8UC4: (1, b'B', 4),

    c_core.CV_8SC1: (1, b'b', 1),
    c_core.CV_8SC2: (1, b'b', 2),
    c_core.CV_8SC3: (1, b'b', 3),
    c_core.CV_8SC4: (1, b'b', 4),

    c_core.CV_16UC1: (2, b'H', 1),
    c_core.CV_16UC2: (2, b'H', 2),
    c_core.CV_16UC3: (2, b'H', 3),
    c_core.CV_16UC4: (2, b'H', 4),

    c_core.CV_16SC1: (2, b'h', 1),
    c_core.CV_16SC2: (2, b'h', 2),
    c_core.CV_16SC3: (2, b'h', 3),
    c_core.CV_16SC4: (2, b'h', 4),

    c_core.CV_32SC1: (4, b'i', 1),
    c_core.CV_32SC2: (4, b'i', 2),
    c_core.CV_32SC3: (4, b'i', 3),
    c_core.CV_32SC4: (4, b'i', 4),

    c_core.CV_32FC1: (4, b'f', 1),
    c_core.CV_32FC2: (4, b'f', 2),
    c_core.CV_32FC3: (4, b'f', 3),
    c_core.CV_32FC4: (4, b'f', 4),

    c_core.CV_64FC1: (8, b'd', 1),
    c_core.CV_64FC2: (8, b'd', 2),
    c_core.CV_64FC3: (8, b'd', 3),
    c_core.CV_64FC4: (8, b'd', 4),
}

_np_dtype_to_cv_dtype_lookup = {
     (np.dtype(np.uint8), 1): CV_8UC1,
     (np.dtype(np.uint8), 2): CV_8UC2,
     (np.dtype(np.uint8), 3): CV_8UC3,
     (np.dtype(np.uint8), 4): CV_8UC4,

     (np.dtype(np.uint16), 1): CV_16UC1,
     (np.dtype(np.uint16), 2): CV_16UC2,
     (np.dtype(np.uint16), 3): CV_16UC3,
     (np.dtype(np.uint16), 4): CV_16UC4,

     (np.dtype(np.int16), 1): CV_16SC1,
     (np.dtype(np.int16), 2): CV_16SC2,
     (np.dtype(np.int16), 3): CV_16SC3,
     (np.dtype(np.int16), 4): CV_16SC4,

     (np.dtype(np.int32), 1): CV_32SC1,
     (np.dtype(np.int32), 2): CV_32SC2,
     (np.dtype(np.int32), 3): CV_32SC3,
     (np.dtype(np.int32), 4): CV_32SC4,

     (np.dtype(np.float32), 1): CV_32FC1,
     (np.dtype(np.float32), 2): CV_32FC2,
     (np.dtype(np.float32), 3): CV_32FC3,
     (np.dtype(np.float32), 4): CV_32FC4,

     (np.dtype(np.float64), 1): CV_64FC1,
     (np.dtype(np.float64), 2): CV_64FC2,
     (np.dtype(np.float64), 3): CV_64FC3,
     (np.dtype(np.float64), 4): CV_64FC4,
}

cdef class Mat:
    def __cinit__(self, int rows=0, int cols=0, int dtype=CV_8UC3):
        self.c_mat = c_Mat(rows, cols, dtype)
        self.view_count = 0

    @staticmethod
    cdef Mat from_mat(c_Mat mat):
        cdef Mat py_mat = Mat.__new__(Mat)
        py_mat.c_mat = mat
        return py_mat

    @staticmethod
    def fromarray(np.ndarray array not None) -> Mat:
        if array.ndim not in (2, 3):
            raise ValueError("Can only create a 2D or 3D Mat, but the array passed was {}D".format(array.ndim))
        channels = array.shape[2] if array.ndim > 2 else 1
        dtype = _np_dtype_to_cv_dtype_lookup[array.dtype, channels]
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
        cdef void* data = <void*> array.data
        cdef c_Mat c_mat = c_Mat(array.shape[0], array.shape[1], dtype, data)
        return Mat.from_mat(c_mat)

    @property
    def rows(self):
        return self.c_mat.rows

    @property
    def cols(self):
        return self.c_mat.cols

    @property
    def channels(self):
        return self.c_mat.channels()

    @property
    def dtype(self):
        return self.c_mat.type()

    @property
    def empty(self):
        return self.c_mat.empty()

    @property
    def ndim(self):
        # OpenCV doesn't count the channel dimension, but to mimic numpy
        # we will.
        return self.c_mat.dims + 1

    @property
    def depth(self):
        return self.c_mat.depth()

    @property
    def shape(self):
        return (self.rows, self.cols, self.channels)

    def asarray(self):
        import numpy as np
        return np.asarray(self)

    def __bool__(self):
        return not self.empty

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        (byte_count, type_str, channel_count) = _mat_type_lookup[self.c_mat.type()]

        self.shape[0] = self.c_mat.rows
        self.shape[1] = self.c_mat.cols
        self.shape[2] = self.c_mat.channels()

        self.strides[0] = self.c_mat.step[0]
        self.strides[1] = self.c_mat.step[1]
        self.strides[2] = byte_count

        buffer.buf = <char *>(self.c_mat.data)
        buffer.format = type_str
        buffer.internal = NULL
        buffer.itemsize = byte_count
        buffer.len = self.shape[0] * self.shape[1] * self.shape[2] * byte_count
        buffer.ndim = self.ndim
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

        self.view_count += 1

    def __releasebuffer__(self, Py_buffer *buffer):
        self.view_count -= 1

    def __repr__(self):
        return "Mat(rows={rows}, cols={cols}, dtype={dtype})".format(
            rows=self.rows,
            cols=self.cols,
            dtype=self.dtype
        )
