from .c_core cimport InputArray, OutputArray


cdef extern from "opencv2/imgproc.hpp" namespace "cv" nogil:
    enum ColorConversionCodes:
        COLOR_BGR2GRAY

    void cvtColor(InputArray, OutputArray, int, int)
    void cvtColor(InputArray, OutputArray, int)

