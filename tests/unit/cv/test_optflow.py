from abc import ABC

import numpy as np
from numpy.testing import assert_equal
from flowty.cv.optflow import TvL1OpticalFlow, FarnebackOpticalFlow, \
    DenseInverseSearchOpticalFlow, VariationalRefinementOpticalFlow
from flowty.cv.core import Mat, CV_32FC2
import pytest


def make_random_uint8_mat(rows, cols, channels):
    return Mat.fromarray((np.random.rand(rows, cols, channels) * 255).astype(
            np.uint8), copy=True)


class OpticalFlowAlgorithmTestBase(ABC):
    img_size = (10, 20)

    def get_flow_algorithm(self):
        raise NotImplementedError()

    def test_creation(self):
        self.get_flow_algorithm()

    def test_computing_flow(self):
        alg = self.get_flow_algorithm()
        print(self.img_size)
        reference = make_random_uint8_mat(self.img_size[0], self.img_size[1], 3)
        target = make_random_uint8_mat(self.img_size[0], self.img_size[1], 3)

        flow = alg(reference, target)

        assert flow.shape[:2] == target.shape[:2]
        assert flow.shape[2] == 2
        assert flow.dtype == CV_32FC2

    def test_input_frames_arent_modified(self):
        alg = self.get_flow_algorithm()
        reference = make_random_uint8_mat(self.img_size[0], self.img_size[1], 3)
        target = make_random_uint8_mat(self.img_size[0], self.img_size[1], 3)
        reference_original = reference.asarray().copy()
        target_original = target.asarray().copy()

        alg(reference, target)

        assert_equal(reference.asarray(), reference_original)
        assert_equal(target.asarray(), target_original)

    def test_flow_mat_isnt_changed_when_computing_multiple_flows(self):
        alg = self.get_flow_algorithm()
        reference = make_random_uint8_mat(self.img_size[0], self.img_size[1], 3)
        target = make_random_uint8_mat(self.img_size[0], self.img_size[1], 3)

        flow1 = alg(reference, target)
        flow1_original = flow1.asarray().copy()
        alg(target, reference)

        assert_equal(flow1.asarray(), flow1_original)


class TestTvL1OpticalFlow(OpticalFlowAlgorithmTestBase):
    def get_flow_algorithm(self):
        return TvL1OpticalFlow()

    @pytest.mark.parametrize("property,expected_value", [
        ("tau", 0.25),
        ("lambda_", 0.15),
        ("theta", 0.3),
        ("epsilon", 0.01),
        ("gamma", 0.0),
        ("scale_step", 0.8),
        ("scale_count", 5),
        ("warp_count", 5),
        ("outer_iterations", 10),
        ("inner_iterations", 30),
        ("median_filtering", 5),
        ("use_initial_flow", False)
    ])
    def test_property(self, property, expected_value):
        assert getattr(self.get_flow_algorithm(), property) == expected_value

    def test_repr(self):
        assert repr(TvL1OpticalFlow()) == \
               ("TvL1OpticalFlow(" 
                "tau=0.25, lambda_=0.15, theta=0.3, epsilon=0.01, gamma=0.0, scale_step=0.8, "
                "scale_count=5, warp_count=5, outer_iterations=10, inner_iterations=30, "
                "median_filtering=5, use_initial_flow=False"
                ")")


class TestFarnebackOpticalFlow(OpticalFlowAlgorithmTestBase):
    def get_flow_algorithm(self):
        return FarnebackOpticalFlow()


class TestVariationalRefinementOpticalFlow(OpticalFlowAlgorithmTestBase):
    def get_flow_algorithm(self):
        return VariationalRefinementOpticalFlow()


class TestDenseInverseSearchOpticalFlow(OpticalFlowAlgorithmTestBase):
    img_size = (128, 128)

    def get_flow_algorithm(self):
        return DenseInverseSearchOpticalFlow()

