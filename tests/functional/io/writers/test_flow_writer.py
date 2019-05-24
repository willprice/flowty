import argparse
from abc import ABC
from pathlib import Path

import numpy as np
import pytest
from imageio import imread
from numpy.testing import assert_array_almost_equal, assert_allclose

from flowty.cv.optflow import read_flo
from flowty.videoio import (
    FlowUVImageWriter,
    FlowNumpyWriter,
    MiddleburyFlowWriter,
    get_flow_writer,
)


class TestGetFlowWriter:
    @pytest.mark.parametrize("extension", ["jpg", "jpeg", "JPEG", "png", "PNG"])
    def test_image_extensions_returns_flow_uv_writer(self, extension):
        writer = get_flow_writer(self.create_args(extension, bound=25))
        assert isinstance(writer, FlowUVImageWriter)

    @pytest.mark.parametrize("extension", ["npy", "np", "NPY"])
    def test_npy_returns_flow_numpy_writer(self, extension):
        writer = get_flow_writer(self.create_args(extension))
        assert isinstance(writer, FlowNumpyWriter)

    def test_flo_returns_middlebury_flow_writer(self):
        writer = get_flow_writer(self.create_args("flo"))
        assert isinstance(writer, MiddleburyFlowWriter)

    def test_raises_error_on_unknown_extension(self):
        with pytest.raises(ValueError):
            get_flow_writer(self.create_args("asdf"))

    def test_bound_set_in_uv_flow_writer(self):
        bound = 25
        writer = get_flow_writer(self.create_args('jpg', bound=bound))
        assert isinstance(writer, FlowUVImageWriter)
        assert writer.bound == bound

    def create_args(self, extension: str, **namespace_kwargs) -> argparse.Namespace:
        args = argparse.Namespace(**namespace_kwargs)
        args.dest = "path/{axis}/frame_{index:05d}." + extension
        return args


class AbstractTestFlowWriter(ABC):
    bound = 20

    def test_throws_error_if_flow_not_2_channels(self, tmp_path):
        image_writer = self.get_flow_writer(tmp_path)
        flow = np.random.uniform(low=-self.bound, high=self.bound, size=(5, 5, 3))

        with pytest.raises(ValueError, match="Expected flow to have 2 channels.*"):
            image_writer.write(flow)

    def test_throws_error_if_flow_not_3_dimensional(self, tmp_path):
        image_writer = self.get_flow_writer(tmp_path)
        flow = np.random.uniform(low=-self.bound, high=self.bound, size=(5, 5))

        with pytest.raises(ValueError, match="Expected flow to be 3D.*"):
            image_writer.write(flow)

    def test_saving_single_flow_image(self, tmp_path):
        image_writer = self.get_flow_writer(tmp_path)
        flow = np.random.uniform(
            low=-self.bound, high=self.bound, size=(5, 5, 2)
        ).astype(np.float32)

        image_writer.write(flow)

        self.assert_flow_exists(tmp_path, index=0)
        self.assert_flow_equal(tmp_path, index=0, flow=flow)

    def get_flow_writer(self, tmp_path):
        raise NotImplementedError()

    def assert_flow_exists(self, tmp_dir: Path, index: int):
        raise NotImplementedError()

    def assert_flow_equal(self, tmp_dir, index: int, flow: np.ndarray):
        raise NotImplementedError()


class TestFlowUVImageWriter(AbstractTestFlowWriter):
    def test_throws_error_if_template_missing_axis_field(self):
        with pytest.raises(ValueError):
            FlowUVImageWriter("flow/{index:05d}.jpg")

    def test_throws_error_if_template_missing_index_field(self):
        with pytest.raises(ValueError):
            FlowUVImageWriter("flow/{axis}/frame.jpg")

    def get_flow_writer(self, tmp_path):
        filename_template = tmp_path / "{axis}/{index:05d}.jpg"
        return FlowUVImageWriter(str(filename_template))

    def assert_flow_exists(self, tmp_dir: Path, index: int):
        for axis in ["u", "v"]:
            assert (tmp_dir / axis / "0000{}.jpg".format(index + 1)).exists()

    def assert_flow_equal(self, tmp_dir, index: int, flow: np.ndarray):
        u_flow = imread(tmp_dir / "u" / "{:05}.jpg".format(index + 1))
        v_flow = imread(tmp_dir / "v" / "{:05}.jpg".format(index + 1))
        jpg_flow_quantised = np.concatenate(
            (u_flow[..., np.newaxis], v_flow[..., np.newaxis]), axis=-1
        ).astype(np.float32)
        flow_range = 2 * self.bound
        jpg_flow = ((jpg_flow_quantised / 255) * flow_range) - self.bound
        tolerance = flow_range / 30  # We allow up to 3% error due to JPEG compression.
        assert_allclose(jpg_flow, flow, atol=tolerance)


class TestFlowNumpyWriter(AbstractTestFlowWriter):
    def test_throws_error_if_template_missing_index_field(self):
        with pytest.raises(ValueError):
            FlowNumpyWriter("flow/frame.jpg")

    def assert_flow_exists(self, tmp_dir: Path, index: int):
        assert (tmp_dir / "frame_{:05d}.npy".format(index + 1)).exists()

    def assert_flow_equal(self, tmp_dir, index: int, flow: np.ndarray):
        loaded_flow = np.load(tmp_dir / "frame_{:05d}.npy".format(index + 1))
        assert_array_almost_equal(loaded_flow, flow)

    def get_flow_writer(self, tmp_path):
        filename_template = tmp_path / "frame_{index:05d}.npy"
        return FlowNumpyWriter(str(filename_template))


class TestFlowMiddleburyWriter(AbstractTestFlowWriter):
    def test_throws_error_if_template_missing_index_field(self):
        with pytest.raises(ValueError):
            MiddleburyFlowWriter("flow/frame.jpg")

    def assert_flow_exists(self, tmp_dir: Path, index: int):
        assert (tmp_dir / "frame_{:05d}.flo".format(index + 1)).exists()

    def assert_flow_equal(self, tmp_dir, index: int, flow: np.ndarray):
        loaded_flow = np.array(
            read_flo(str(tmp_dir / "frame_{:05d}.flo".format(index + 1)))
        )
        assert_array_almost_equal(loaded_flow, flow)

    def get_flow_writer(self, tmp_path):
        filename_template = tmp_path / "frame_{index:05d}.flo"
        return MiddleburyFlowWriter(str(filename_template))
