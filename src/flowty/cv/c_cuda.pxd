# cython: language_level = 3
from libcpp cimport bool
from libcpp.vector cimport vector
from .c_core cimport InputArray, OutputArray

cdef extern from "opencv2/core/cuda.hpp" namespace "cv::cuda" nogil:
    int getDevice() except +
    int getCudaEnabledDeviceCount() except +
    void printCudaDeviceInfo(int) except +
    void printShortCudaDeviceInfo(int) except +
    void resetDevice() except +
    void setDevice(int) except +

    cdef cppclass GpuMat:
        GpuMat() except +
        GpuMat(int, int, int) except +
        GpuMat(InputArray) except +

        bool empty()
        int type()
        GpuMat clone() except +
        int depth()
        int channels()
        size_t elemSize()
        size_t elemSize1()
        T at[T](int, int)

        void download(OutputArray) except +
        void upload(InputArray) except +

        unsigned char* data
        size_t step
        unsigned char* datastart
        unsigned char* dataend
        int* refcount
        int cols, rows, flags

