import pytest
from pytest import approx

import flowty
import numpy as np

from flowty.cv.cuda_optflow import CudaTvL1OpticalFlow, CudaBroxOpticalFlow, \
    CudaPyramidalLucasKanade, CudaFarnebackOpticalFlow
from flowty.cv.core import Mat

from tests.unit.cv.test_optflow import OpticalFlowAlgorithmTestBase

if not flowty.cuda_available:
    pytest.skip("skipping CUDA-only module: flowty.cv.cuda_optflow", allow_module_level=True)


def to_rgb(grayscale_img: np.ndarray):
    assert grayscale_img.ndim == 2
    return Mat.fromarray(np.stack([grayscale_img] * 3, axis=-1).astype(np.uint8))


class TestCudaTvL1OpticalFlow(OpticalFlowAlgorithmTestBase):
    def get_flow_algorithm(self):
        return CudaTvL1OpticalFlow()

    @pytest.mark.parametrize("property,expected_value", [
        ("tau", 0.25),
        ("lambda_", 0.15),
        ("theta", 0.3),
        ("epsilon", 0.01),
        ("gamma", 0.0),
        ("scale_step", 0.8),
        ("scale_count", 5),
        ("warp_count", 5),
        ("iterations", 300),
        ("use_initial_flow", False)
    ])
    def test_property(self, property, expected_value):
        assert getattr(self.get_flow_algorithm(), property) == expected_value

    def test_repr(self):
        assert repr(CudaTvL1OpticalFlow()) == ("CudaTvL1OpticalFlow(" +
        "tau=0.25, lambda_=0.15, theta=0.3, epsilon=0.01, gamma=0.0, scale_step=0.8, "
        "scale_count=5, warp_count=5, iterations=300, use_initial_flow=False"
        ")")


class TestCudaBroxOpticalFlow(OpticalFlowAlgorithmTestBase):
    def get_flow_algorithm(self):
        return CudaBroxOpticalFlow()

    @pytest.mark.parametrize("property,expected_value", [
        ("alpha", approx(0.197)),
        ("gamma", 50.0),
        ("scale_factor", approx(0.8)),
        ("inner_iterations", 5),
        ("outer_iterations", 150),
        ("solver_iterations", 10),
    ])
    def test_property(self, property, expected_value):
        assert getattr(self.get_flow_algorithm(), property) == expected_value

    def test_repr(self):
        assert repr(CudaBroxOpticalFlow(alpha=1.0, scale_factor=0.8)) == \
               "CudaBroxOpticalFlow(alpha=1.0, gamma=50.0, scale_factor=0.8, " \
               "inner_iterations=5, outer_iterations=150, solver_iterations=10)"


class TestCudaPyramidalLucasKanade(OpticalFlowAlgorithmTestBase):
    def get_flow_algorithm(self):
        return CudaPyramidalLucasKanade()

    @pytest.mark.parametrize("property,expected_value", [
        ("window_size", 13),
        ("max_scales", 3),
        ("iterations", 30),
    ])
    def test_property(self, property, expected_value):
        assert getattr(self.get_flow_algorithm(), property) == expected_value


class TestCudaFarnebackOpticalFlow(OpticalFlowAlgorithmTestBase):
    def get_flow_algorithm(self):
        return CudaFarnebackOpticalFlow()

    @pytest.mark.parametrize("property,expected_value", [
        ("scale_count", 5),
        ("scale_factor", 0.5),
        ("use_fast_pyramids", False),
        ("window_size", 13),
        ("iterations", 10),
        ("poly_count", 5),
        ("poly_sigma", 1.1)

    ])
    def test_property(self, property, expected_value):
        assert getattr(self.get_flow_algorithm(), property) == expected_value

