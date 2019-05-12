import argparse
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from flowty.videoio import FlowUVImageWriter, FlowNumpyWriter, get_flow_writer


class TestGetFlowWriter:
    @pytest.mark.parametrize("extension", ["jpg", "jpeg", "JPEG", "png", "PNG"])
    def test_image_extensions_returns_flow_uv_writer(self, extension):
        writer = get_flow_writer(self.create_args(extension))
        assert isinstance(writer, FlowUVImageWriter)

    @pytest.mark.parametrize("extension", ["npy", "np", "NPY"])
    def test_npy_returns_flow_numpy_writer(self, extension):
        writer = get_flow_writer(self.create_args(extension))
        assert isinstance(writer, FlowNumpyWriter)

    def test_raises_error_on_unknown_extension(self):
        with pytest.raises(ValueError):
            get_flow_writer(self.create_args("asdf"))

    def create_args(self, extension: str) -> argparse.Namespace:
        args = argparse.Namespace()
        args.dest = "path/{axis}/frame_{index:05d}." + extension
        return args


class TestFlowUVImageWriter:
    def test_saving_single_flow_image(self, tmp_path):
        image_writer = self.get_flow_writer(tmp_path)
        flow = np.random.uniform(low=-20, high=20, size=(5, 5, 2))
        image_writer.write(flow)

        for axis in ['u', 'v']:
            assert (Path(tmp_path) / axis / '00001.jpg').exists()

    def test_throws_error_if_flow_not_2_channels(self, tmp_path):
        image_writer = self.get_flow_writer(tmp_path)
        flow = np.random.uniform(low=-20, high=20, size=(5, 5, 3))

        with pytest.raises(ValueError):
            image_writer.write(flow)

    def test_throws_error_if_flow_not_3_dimensional(self, tmp_path):
        image_writer = self.get_flow_writer(tmp_path)
        flow = np.random.uniform(low=-20, high=20, size=(5, 5))

        with pytest.raises(ValueError):
            image_writer.write(flow)

    def get_flow_writer(self, tmp_path):
        filename_template = tmp_path / '{axis}/{index:05d}.jpg'
        return FlowUVImageWriter(str(filename_template))


class TestFlowNumpyWriter:
    def test_saving_single_flow_field(self, tmp_path):
        writer = self.get_flow_writer(tmp_path)

        flow = np.random.uniform(low=-20, high=20, size=(5, 5, 2))

        writer.write(flow)

        expect_flow_file = Path(tmp_path / 'frame_00001.npy')
        assert expect_flow_file.exists()
        loaded_flow = np.load(expect_flow_file)
        assert_array_equal(loaded_flow, flow)

    def test_raises_error_if_flow_not_3_dimensional(self, tmp_path):
        writer = self.get_flow_writer(tmp_path)
        flow = np.random.uniform(low=-20, high=20, size=(5, 5))

        with pytest.raises(ValueError):
            writer.write(flow)

    def test_raises_error_if_flow_not_2_channel(self, tmp_path):
        writer = self.get_flow_writer(tmp_path)
        flow = np.random.uniform(low=-20, high=20, size=(5, 5, 3))

        with pytest.raises(ValueError):
            writer.write(flow)

    def get_flow_writer(self, tmp_path):
        filename_template = tmp_path / 'frame_{index:05d}.npy'
        return FlowNumpyWriter(str(filename_template))
