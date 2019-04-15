cdef extern from "opencv2/core.hpp":
    cdef int CV_WINDOW_AUTOSIZE
    cdef int CV_8UC3

cdef extern from "opencv2/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int)
        void* data

    cdef exter
