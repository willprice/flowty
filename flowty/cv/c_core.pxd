# cython: language_level = 3
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr


cdef extern from "opencv2/imgproc.hpp" namespace "cv":
    enum ColorConversionCodes:
        COLOR_BGR2GRAY

    void cvtColor(InputArray, OutputArray, int, int)
    void cvtColor(InputArray, OutputArray, int)


cdef extern from "opencv2/core.hpp":
    cdef enum:
        CV_8U
        CV_8S
        CV_16U
        CV_16S
        CV_32S
        CV_32F
        CV_64F

        CV_8UC1
        CV_8UC2
        CV_8UC3
        CV_8UC4

        CV_8SC1
        CV_8SC2
        CV_8SC3
        CV_8SC4

        CV_16UC1
        CV_16UC2
        CV_16UC3
        CV_16UC4

        CV_16SC1
        CV_16SC2
        CV_16SC3
        CV_16SC4

        CV_32SC1
        CV_32SC2
        CV_32SC3
        CV_32SC4

        CV_32FC1
        CV_32FC2
        CV_32FC3
        CV_32FC4

        CV_64FC1
        CV_64FC2
        CV_64FC3
        CV_64FC4

    unsigned int CV_MAT_DEPTH_MASK
    ctypedef shared_ptr Ptr

cdef extern from "opencv2/core.hpp" namespace "cv":
    cdef cppclass Size:
        Size()
        Size(int, int)
        Size(Size&)
        int area()
        double aspectRatio()
        bool empty()
        Size& operator=(Size&)

    cdef cppclass MatSize:
        MatSize(int *)
        int dims()
        bool operator!=(MatSize &)
        bool operator==(MatSize &)
        int operator[](int i)

    cdef cppclass MatStep:
        MatStep()
        MatStep(size_t s)
        MatStep & operator=(size_t s)
        size_t & operator[](int i)
        size_t buf[2]
        size_t* p

    cdef cppclass Mat:
        Mat() except +
        Mat(int, int, int) except +

        void create(int, int, int) except +
        bool empty()
        int type()
        Mat clone()
        int depth()
        int channels()
        size_t elemSize()
        size_t elemSize1()
        T at[T](int, int)

        void* data
        MatSize size
        MatStep step
        unsigned char* datastart
        unsigned char* dataend
        int cols, rows, flags, dims


cdef extern from "opencv2/core/mat.hpp" namespace "cv":
    cdef cppclass InputArray:
        InputArray()
        InputArray(int, void*)
        InputArray(const Mat&)
        InputArray(vector[Mat] &)

    cdef cppclass OutputArray:
        OutputArray()
        OutputArray(int, void*)
        OutputArray(const Mat&)
        OutputArray(vector[Mat] &)

    cdef cppclass InputOutputArray:
        InputOutputArray()
        InputOutputArray(int, void*)
        InputOutputArray(const Mat&)
        InputOutputArray(vector[Mat] &)
