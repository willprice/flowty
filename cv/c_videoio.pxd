# cython: language_level = 3
from c_core cimport Mat, OutputArray
from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "opencv2/videoio.hpp" namespace "cv":
    cdef enum:
        CAP_ANY
        CAP_V4L
        CAP_V4L2
        CAP_FIREWIRE
        CAP_OPENNI
        CAP_OPENNI_ASUS
        CAP_ANDROID
        CAP_XIAPI
        CAP_AVFOUNDATION
        CAP_GIGANETIX
        CAP_MSMF
        CAP_WINRT
        CAP_INTELPERC
        CAP_REALSENSE
        CAP_OPENNI2
        CAP_OPENNI2_ASUS
        CAP_GPHOTO2
        CAP_GSTREAMER
        CAP_FFMPEG
        CAP_IMAGES
        CAP_ARAVIS
        CAP_OPENCV_MJPEG
        CAP_INTEL_MFX
        CAP_XINE

cdef extern from "opencv2/videoio.hpp" namespace "cv":
    cdef cppclass VideoCapture:
        VideoCapture() except +
        VideoCapture(const string&, int) except +
        VideoCapture(int, int) except +
        double get(int)
        string getBackendName()
        bool grab()
        bool isOpened()
        bool open(const string&, int) except +
        bool open(int, int) except +
        bool read(OutputArray)
        void release()
        void retrieve(OutputArray, int)
        set(int, double)
        VideoCapture & operator>>(Mat)

    cdef cppclass VideoWriter:
        VideoWriter(const string&, int, double, Size, bool) except +
        VideoWriter(const string&, int, int, double, Size, bool) except +
        double get(int)
        string getBackendName()
        bool isOpened()
        open(const string &, int, double, Size, bool)
        open(const string &, int, int, double, Size, bool)
        VideoWriter & operator<<(const Mat&)
        void release()
