# cython: language_level=3

from .c_core cimport Mat as c_Mat
from libcpp cimport bool

cdef class Mat:
    cdef:
        c_Mat c_mat
        int view_count
        Py_ssize_t _shape[3]
        Py_ssize_t _strides[3]

    @staticmethod
    cdef Mat from_mat(c_Mat mat, bool copy = *)
