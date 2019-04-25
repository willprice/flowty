# cython: language_level=3

from .c_core cimport Mat as c_Mat

cdef class Mat:
    cdef:
        c_Mat c_mat
        int view_count
        Py_ssize_t shape[3] 
        Py_ssize_t strides[3]

    @staticmethod
    cdef Mat from_mat(c_Mat mat)

