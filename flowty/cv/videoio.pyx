# cython: language_level=3
from libcpp.string cimport string
from libcpp cimport bool

from .c_core cimport Mat as c_Mat, OutputArray
from .core cimport Mat
from .core import Mat
from .c_videoio cimport CAP_ANY, VideoCapture


cdef class VideoSource:
    cdef VideoCapture c_cap

    def __cinit__(self, str filename):
        cdef string cpp_filename = filename.encode('UTF-8')
        self.c_cap = VideoCapture(cpp_filename, CAP_ANY)

    def __iter__(self):
        return self

    def __next__(self):
        cdef c_Mat frame
        cdef bool read = self.c_cap.read(<OutputArray>frame)
        if not read:
            raise StopIteration()
        return Mat.from_mat(frame)
