import numpy as np
from flowty.cv.optflow import TvL1OpticalFlow
from flowty.cv.core import Mat, CV_32FC2


def make_random_uint8_mat(rows, cols, channels):
    return Mat.fromarray((np.random.rand(rows, cols, channels) * 255).astype(np.uint8))


class TestTvL1OpticalFlow:
    def test_creation(self):
        TvL1OpticalFlow()

    def test_tau_property(self):
        assert TvL1OpticalFlow().tau == 0.25

    def test_lambda_property(self):
        assert TvL1OpticalFlow().lambda_ == 0.15

    def test_theta_propertys(self):
        assert TvL1OpticalFlow().theta == 0.3

    def test_epsilon_property(self):
        assert TvL1OpticalFlow().epsilon == 0.01

    def test_gamma_property(self):
        assert TvL1OpticalFlow().gamma == 0.0

    def test_scale_step_property(self):
        assert TvL1OpticalFlow().scale_step == 0.8

    def test_scale_count_property(self):
        assert TvL1OpticalFlow().scale_count == 5

    def test_warp_count_property(self):
        assert TvL1OpticalFlow().warp_count == 5

    def test_outer_iterations_property(self):
        assert TvL1OpticalFlow().outer_iterations == 10

    def test_inner_iterations_property(self):
        assert TvL1OpticalFlow().inner_iterations == 30

    def test_median_filtering_property(self):
        assert TvL1OpticalFlow().median_filtering == 5

    def test_use_initial_flow_property(self):
        assert not TvL1OpticalFlow().use_initial_flow

    def test_computing_flow(self):
        alg = TvL1OpticalFlow()
        reference = make_random_uint8_mat(10, 20, 3)
        target = make_random_uint8_mat(10, 20, 3)
        flow = alg(reference, target)
        assert flow.shape[:2] == target.shape[:2]
        assert flow.shape[2] == 2
        assert flow.dtype == CV_32FC2
