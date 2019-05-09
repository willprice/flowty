from flowty.cv.videoio import VideoSource
from tests.resources import VIDEO_PATHS


class TestVideoSource:
    expected_shape = (480, 480, 3)
    expected_frame_count = 486

    def test_reading_mp4(self):
        n_frames, shape = self.read_video(VIDEO_PATHS['mp4'])
        assert self.expected_shape == shape
        assert self.expected_frame_count == n_frames

    def test_reading_avi(self):
        n_frames, shape = self.read_video(VIDEO_PATHS['avi'])
        assert self.expected_shape == shape
        assert self.expected_frame_count == n_frames

    def test_reading_webm(self):
        n_frames, shape = self.read_video(VIDEO_PATHS['webm'])
        assert self.expected_shape == shape
        assert self.expected_frame_count == self.expected_frame_count

    def test_reading_jpegs(self):
        n_frames, shape = self.read_video(VIDEO_PATHS['jpeg'])
        assert self.expected_shape == shape
        assert self.expected_frame_count == n_frames

    def test_ffmpeg_backend(self):
        src = VideoSource(VIDEO_PATHS['mp4'], backend='ffmpeg')
        assert self.expected_shape == next(src).shape

    def test_image_backend(self):
        src = VideoSource(VIDEO_PATHS['jpeg'], backend='images')
        assert self.expected_shape == next(src).shape

    def test_gstreamer_backend(self):
        src = VideoSource(VIDEO_PATHS['mp4'], backend='gstreamer')
        assert self.expected_shape == next(src).shape

    def test_pos_ms_property(self):
        assert 0 == self.get_ffmpeg_src().pos_ms

    def test_pos_frames_property(self):
        assert 0 == self.get_ffmpeg_src().pos_frames

    def test_frame_width_property(self):
        assert self.expected_shape[1] == self.get_ffmpeg_src().frame_width

    def test_frame_height_property(self):
        assert self.expected_shape[0] == self.get_ffmpeg_src().frame_height

    def test_fps_property(self):
        assert 30 == self.get_ffmpeg_src().fps

    def test_frame_count_property(self):
        assert self.expected_frame_count == self.get_ffmpeg_src().frame_count

    def test_backend_property(self):
        assert 'ffmpeg' == self.get_ffmpeg_src().backend

    def get_ffmpeg_src(self):
        return VideoSource(VIDEO_PATHS['mp4'], backend='ffmpeg')

    def read_video(self, video_path, backend='ffmpeg'):
        print(video_path)
        src = VideoSource(video_path, backend=backend)
        n_frames = 0
        shape = (0, 0)
        for frame in iter(src):
            shape = frame.shape
            n_frames += 1
        return n_frames, shape