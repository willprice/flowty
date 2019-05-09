from pathlib import Path

import numpy as np
import pytest

from flowty.videoio import FlowImageWriter


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