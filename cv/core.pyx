# cython: language_level=3

from c_core cimport Mat as c_Mat
from cpython cimport Py_buffer
cimport c_core

CV_8UC1 = c_core.CV_8UC1
CV_8UC2 = c_core.CV_8UC2
CV_8UC3 = c_core.CV_8UC3
CV_32FC1 = c_core.CV_32FC1
CV_32FC2 = c_core.CV_32FC2
CV_32FC3 = c_core.CV_32FC3

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

cdef class Mat:
    def __cinit__(self, int rows=0, int cols=0, int dtype=CV_8UC3):
        self.c_mat = c_Mat(rows, cols, dtype)
        self.view_count = 0

    @staticmethod
    cdef Mat from_mat(c_Mat mat):
        cdef Mat py_mat = Mat.__new__(Mat)
        py_mat.c_mat = mat
        return py_mat

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
        return (self.rows, self.cols)

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
        print(self.c_mat.channels())

        self.strides[0] = self.c_mat.step[0]
        self.strides[1] = self.c_mat.step[1]
        self.strides[2] = byte_count
        print("strides: ", self.c_mat.step[0], self.c_mat.step[1],
              self.c_mat.step[2])


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
        print("end of __getbuffer__")

    def __releasebuffer__(self, Py_buffer *buffer):
        self.view_count -= 1
