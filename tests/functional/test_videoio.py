from pathlib import Path

import pytest

from flowty.cv.videoio import VideoSource
import os
import numpy as np

from flowty.videoio import FlowImageWriter

MEDIA_ROOT = os.path.join(
    os.path.dirname(__file__),
    '..',  '..', 'media'
)
VIDEO_PATHS = {
    'mp4': os.path.join(MEDIA_ROOT, 'mr-bubz.mp4'),
    'webm': os.path.join(MEDIA_ROOT, 'mr-bubz.webm'),
    'avi': os.path.join(MEDIA_ROOT, 'mr-bubz.avi'),
    'jpeg': os.path.join(MEDIA_ROOT, 'mr-bubz', 'frame_%05d.jpg'),
}


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


class TestFlowImageWriter:
    def test_saving_single_flow_image(self, tmp_path):
        image_writer = self.get_flow_writer(tmp_path)
        flow = np.random.randint(low=0, high=255, size=(5, 5, 2), dtype=np.uint8)

        image_writer.write(flow)

        for axis in ['u', 'v']:
            assert (Path(tmp_path) / axis / '00001.jpg').exists()

    def test_throws_error_if_flow_not_uint8(self, tmp_path):
        image_writer = self.get_flow_writer(tmp_path)
        flow = np.random.randint(low=0, high=255, size=(5, 5, 2), dtype=np.int32)

        with pytest.raises(ValueError):
            image_writer.write(flow)

    def test_throws_error_if_flow_not_2_channels(self, tmp_path):
        image_writer = self.get_flow_writer(tmp_path)
        flow = np.random.randint(low=0, high=255, size=(5, 5, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            image_writer.write(flow)

    def test_throws_error_if_flow_not_3_dimensional(self, tmp_path):
        image_writer = self.get_flow_writer(tmp_path)
        flow = np.random.randint(low=0, high=255, size=(5, 5), dtype=np.uint8)

        with pytest.raises(ValueError):
            image_writer.write(flow)

    def get_flow_writer(self, tmp_path):
        filename_template = tmp_path / '{axis}/{index:05d}.jpg'
        return FlowImageWriter(str(filename_template))
