# cython: language_level=3
from libcpp.string cimport string
from libcpp cimport bool

from .c_core cimport Mat as c_Mat, OutputArray, Size
from .core cimport Mat
from .core import Mat
from .c_videoio cimport VideoCapture
from . cimport  c_videoio


_backend_lookup = {
    'any': c_videoio.CAP_ANY,
    'ffmpeg': c_videoio.CAP_FFMPEG,
    'v4l': c_videoio.CAP_V4L,
    'v4l2': c_videoio.CAP_V4L2,
    'openni': c_videoio.CAP_OPENNI,
    'openni-asus': c_videoio.CAP_OPENNI_ASUS,
    'openni2': c_videoio.CAP_OPENNI2,
    'openni2-asus': c_videoio.CAP_OPENNI2,
    'firewire': c_videoio.CAP_FIREWIRE,
    'android': c_videoio.CAP_ANDROID,
    'xiapi': c_videoio.CAP_XIAPI,
    'avfoundation': c_videoio.CAP_AVFOUNDATION,
    'giganetix': c_videoio.CAP_GIGANETIX,
    'msmf': c_videoio.CAP_MSMF,
    'winrt': c_videoio.CAP_WINRT,
    'interlperc': c_videoio.CAP_INTELPERC,
    'realsense': c_videoio.CAP_REALSENSE,
    'gphoto2': c_videoio.CAP_GPHOTO2,
    'gstreamer': c_videoio.CAP_GSTREAMER,
    'image': c_videoio.CAP_IMAGES,
    'images': c_videoio.CAP_IMAGES,
    'aravis': c_videoio.CAP_ARAVIS,
    'mjpeg': c_videoio.CAP_OPENCV_MJPEG,
    'opencv-mjpeg': c_videoio.CAP_OPENCV_MJPEG,
    'mfx': c_videoio.CAP_INTEL_MFX,
    'intel-mfx': c_videoio.CAP_INTEL_MFX,
    'xine': c_videoio.CAP_XINE
}
_reverse_backend_lookup = {enum_value: video_key for video_key, enum_value in _backend_lookup.items()}

cdef class VideoSource:
    """
    Args:
        file_path (str): Path to video.
        backend (str): Backend to use to decode video.
    """
    cdef VideoCapture c_cap

    def __cinit__(self, str file_path, str backend = "ffmpeg"):
        cdef string cpp_file_path = file_path.encode('UTF-8')
        cdef int backend_enum = _backend_lookup[backend.lower()]
        self.c_cap = VideoCapture(cpp_file_path, backend_enum)

    def __iter__(self):
        return self

    def __next__(self):
        cdef c_Mat frame
        cdef bool read = self.c_cap.read(<OutputArray>frame)
        if not read:
            raise StopIteration()
        return Mat.from_mat(frame)

    cpdef open(self, file_path, backend=None):
        cdef string cpp_file_path = file_path.encode('UTF-8')
        if backend == None:
            backend = self.backend
        cdef int backend_enum = backend
        self.c_cap.open(cpp_file_path, backend_enum)

    @property
    def pos_ms(self):
        return self.c_cap.get(c_videoio.CAP_PROP_POS_MSEC)

    @property
    def pos_frames(self):
        return self.c_cap.get(c_videoio.CAP_PROP_POS_FRAMES)

    @property
    def frame_width(self):
        return self.c_cap.get(c_videoio.CAP_PROP_FRAME_WIDTH)

    @property
    def frame_height(self):
        return self.c_cap.get(c_videoio.CAP_PROP_FRAME_HEIGHT)

    @property
    def fps(self):
        return self.c_cap.get(c_videoio.CAP_PROP_FPS)

    @property
    def frame_count(self):
        return self.c_cap.get(c_videoio.CAP_PROP_FRAME_COUNT)

    @property
    def mat_format(self):
        return self.c_cap.get(c_videoio.CAP_PROP_FORMAT)

    @property
    def backend(self):
        return _reverse_backend_lookup[self.c_cap.get(c_videoio.CAP_PROP_BACKEND)]

    @property
    def fourcc(self):
        return self.c_cap.get(c_videoio.CAP_PROP_FOURCC)

    @property
    def brightness(self):
        return self.c_cap.get(c_videoio.CAP_PROP_BRIGHTNESS)

    @property
    def contrast(self):
        return self.c_cap.get(c_videoio.CAP_PROP_CONTRAST)

    @property
    def saturation(self):
        return self.c_cap.get(c_videoio.CAP_PROP_SATURATION)

    @property
    def hue(self):
        return self.c_cap.get(c_videoio.CAP_PROP_HUE)

    @property
    def gain(self):
        return self.c_cap.get(c_videoio.CAP_PROP_GAIN)

    @property
    def exposure(self):
        return self.c_cap.get(c_videoio.CAP_PROP_EXPOSURE)


cdef class VideoWriter:
    cdef c_videoio.VideoWriter c_writer

    def __cinit__(self, str file_path, str fourcc, float fps, int height, int width, str backend = 'ffmpeg'):
        if len(fourcc) != 4:
            raise ValueError("Expected fourcc to be a 4 characeter string but was '{}'".format(fourcc))

        cdef string cpp_file_path = file_path.encode('UTF-8')
        cdef int backend_enum = _backend_lookup[backend.lower()]
        cdef int fourcc_int = VideoWriter.fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3])
        cdef Size size = Size(height, width)
        self.c_writer = c_videoio.VideoWriter(cpp_file_path, backend_enum, fourcc_int, fps, size)

