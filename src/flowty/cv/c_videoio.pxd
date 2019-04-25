# cython: language_level = 3
from libcpp cimport bool
from libcpp.string cimport string
from .c_core cimport Mat, OutputArray, Size

cdef extern from "opencv2/videoio.hpp" namespace "cv":
    cdef enum VideoCaptureAPIs:
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

    cdef enum VideoCaptureProperties:
          CAP_PROP_POS_MSEC
          CAP_PROP_POS_FRAMES
          CAP_PROP_POS_AVI_RATIO
          CAP_PROP_FRAME_WIDTH
          CAP_PROP_FRAME_HEIGHT
          CAP_PROP_FPS
          CAP_PROP_FOURCC
          CAP_PROP_FRAME_COUNT
          CAP_PROP_FORMAT
          CAP_PROP_MODE
          CAP_PROP_BRIGHTNESS
          CAP_PROP_CONTRAST
          CAP_PROP_SATURATION
          CAP_PROP_HUE
          CAP_PROP_GAIN
          CAP_PROP_EXPOSURE
          CAP_PROP_CONVERT_RGB
          CAP_PROP_WHITE_BALANCE_BLUE_U
          CAP_PROP_RECTIFICATION
          CAP_PROP_MONOCHROME
          CAP_PROP_SHARPNESS
          CAP_PROP_AUTO_EXPOSURE
          CAP_PROP_GAMMA
          CAP_PROP_TEMPERATURE
          CAP_PROP_TRIGGER
          CAP_PROP_TRIGGER_DELAY
          CAP_PROP_WHITE_BALANCE_RED_V
          CAP_PROP_ZOOM
          CAP_PROP_FOCUS
          CAP_PROP_GUID
          CAP_PROP_ISO_SPEED
          CAP_PROP_BACKLIGHT
          CAP_PROP_PAN
          CAP_PROP_TILT
          CAP_PROP_ROLL
          CAP_PROP_IRIS
          CAP_PROP_SETTINGS
          CAP_PROP_BUFFERSIZE
          CAP_PROP_AUTOFOCUS
          CAP_PROP_SAR_NUM
          CAP_PROP_SAR_DEN
          CAP_PROP_BACKEND
          CAP_PROP_CHANNEL
          CAP_PROP_AUTO_WB
          CAP_PROP_WB_TEMPERATURE

    cdef enum VideoWriterProperties:
        VIDEOWRITER_PROP_QUALITY
        VIDEOWRITER_PROP_FRAMEBYTES
        VIDEOWRITER_PROP_PROP_NSTRIPES

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
        VideoWriter() except +
        VideoWriter(const string&, int, int, double, Size) except +
        double get(int)
        string getBackendName()
        bool isOpened()
        open(const string &, int, int, double, Size)
        VideoWriter & operator<<(const Mat&)
        void release()
