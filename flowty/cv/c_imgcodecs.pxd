# cython: language_level=3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from .c_core cimport InputArray


cdef extern from "opencv2/imgcodecs.hpp" namespace "cv":
    bool imwrite(string&, InputArray)
    bool imwrite(string&, InputArray, vector[int]& params)
