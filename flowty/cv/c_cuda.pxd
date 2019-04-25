# cython: language_level = 3
from libcpp cimport bool
from libcpp.vector cimport vector
from .c_core cimport InputArray, OutputArray

cdef extern from "opencv2/core/cuda.hpp" namespace "cv::cuda":
    int getDevice() nogil except +
    int getCudaEnabledDeviceCount() nogil except +
    void printCudaDeviceInfo(int) nogil except +
    void printShortCudaDeviceInfo(int) nogil except +
    void resetDevice() nogil except +
    void setDevice(int) nogil except +

    cdef cppclass GpuMat:
        GpuMat() nogil except +
        GpuMat(int, int, int) nogil except +
        GpuMat(InputArray) nogil except +

        bool empty()
        int type()
        GpuMat clone() except +
        int depth()
        int channels()
        size_t elemSize()
        size_t elemSize1()
        T at[T](int, int)

        void download(OutputArray) nogil except +
        void upload(InputArray) nogil except +

        unsigned char* data
        size_t step
        unsigned char* datastart
        unsigned char* dataend
        int* refcount
        int cols, rows, flags


    cdef cppclass Stream:
        Stream() nogil except +
        bool queryIfComplete() nogil
        void waitForCompletion() nogil
