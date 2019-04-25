# cython: language_level=3
import numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from .c_core cimport InputArray
from .core cimport Mat
from .core import Mat
from .c_imgcodecs cimport imwrite as c_imwrite

def imwrite(file_path: str, img: np.ndarray):
    cdef string c_file_path = file_path.encode('UTF-8')
    cdef Mat mat = Mat.fromarray(img)
    cdef bool success = c_imwrite(c_file_path, <InputArray> mat.c_mat)
    if not success:
        raise RuntimeError("Could not write image to {}".format(file_path))

