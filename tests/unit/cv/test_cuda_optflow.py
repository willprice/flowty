import pytest
import flowty
import numpy as np

from flowty.cv.cuda_optflow import CudaTvL1OpticalFlow, CudaBroxOpticalFlow
from flowty.cv.core import Mat, CV_8UC3, CV_32FC2
from numpy.testing import assert_equal


if not flowty.cuda_available:
    pytest.skip("skipping CUDA-only module: flowty.cv.cuda_optflow", allow_module_level=True)


def make_random_uint8_mat(rows, cols, channels):
    return Mat.fromarray((np.random.rand(rows, cols, channels) * 255).astype(np.uint8))


class TestCudaTvL1OpticalFlow:
    def test_creation(self):
        CudaTvL1OpticalFlow()

    def test_tau_property(self):
        assert CudaTvL1OpticalFlow().tau == 0.25

    def test_lambda_property(self):
        assert CudaTvL1OpticalFlow().lambda_ == 0.15

    def test_theta_propertys(self):
        assert CudaTvL1OpticalFlow().theta == 0.3

    def test_epsilon_property(self):
        assert CudaTvL1OpticalFlow().epsilon == 0.01

    def test_gamma_property(self):
        assert CudaTvL1OpticalFlow().gamma == 0.0

    def test_scale_step_property(self):
        assert CudaTvL1OpticalFlow().scale_step == 0.8

    def test_scale_count_property(self):
        assert CudaTvL1OpticalFlow().scale_count == 5

    def test_warp_count_property(self):
        assert CudaTvL1OpticalFlow().warp_count == 5

    def test_iterations_property(self):
        assert CudaTvL1OpticalFlow().iterations == 300

    def test_use_initial_flow_property(self):
        assert not CudaTvL1OpticalFlow().use_initial_flow

    def test_repr(self):
        assert repr(CudaTvL1OpticalFlow()) == ("CudaTvL1OpticalFlow(" +
        "tau=0.25, lambda_=0.15, theta=0.3, epsilon=0.01, gamma=0.0, scale_step=0.8, "
        "scale_count=5, warp_count=5, iterations=300, use_initial_flow=False"
        ")")

    def test_computing_flow(self):
        alg = CudaTvL1OpticalFlow()
        reference = make_random_uint8_mat(10, 20, 3)
        target = make_random_uint8_mat(10, 20, 3)

        flow = alg(reference, target)

        assert flow.shape[:2] == target.shape[:2]
        assert flow.shape[2] == 2
        assert flow.dtype == CV_32FC2
 
    def test_input_frames_arent_modified(self):
        alg = CudaTvL1OpticalFlow()
        reference = make_random_uint8_mat(10, 20, 3)
        target = make_random_uint8_mat(10, 20, 3)
        reference_original = reference.asarray().copy()
        target_original = target.asarray().copy()

        alg(reference, target)

        assert_equal(reference.asarray(), reference_original)
        assert_equal(target.asarray(), target_original)

    def test_flow_mat_isnt_changed_when_computing_multiple_flows(self):
        alg = CudaTvL1OpticalFlow()
        reference = make_random_uint8_mat(10, 20, 3)
        target = make_random_uint8_mat(10, 20, 3)

        flow1 = alg(reference, target)
        flow1_original = flow1.asarray().copy()
        alg(target, reference)

        assert_equal(flow1.asarray(), flow1_original)


class TestCudaBroxOpticalFlow:
    def test_creation(self):
        CudaBroxOpticalFlow()

    def test_computing_flow(self):
        alg = CudaBroxOpticalFlow()
        reference = make_random_uint8_mat(10, 20, 3)
        target = make_random_uint8_mat(10, 20, 3)

        flow = alg(reference, target)

        assert flow.shape[:2] == target.shape[:2]
        assert flow.shape[2] == 2
        assert flow.dtype == CV_32FC2

    def test_input_frames_arent_modified(self):
        alg = CudaBroxOpticalFlow()
        reference = make_random_uint8_mat(10, 20, 3)
        target = make_random_uint8_mat(10, 20, 3)
        reference_original = reference.asarray().copy()
        target_original = target.asarray().copy()

        alg(reference, target)

        assert_equal(reference.asarray(), reference_original)
        assert_equal(target.asarray(), target_original)

    def test_flow_mat_isnt_changed_when_computing_multiple_flows(self):
        alg = CudaBroxOpticalFlow()
        reference = make_random_uint8_mat(10, 20, 3)
        target = make_random_uint8_mat(10, 20, 3)

        flow1 = alg(reference, target)
        flow1_original = flow1.asarray().copy()
        alg(target, reference)

        assert_equal(flow1.asarray(), flow1_original)

    def test_alpha_property(self):
        alpha = 0.5
        alg = CudaBroxOpticalFlow(alpha=alpha)
        assert alg.alpha == alpha

    def test_gamma_property(self):
        gamma = 40.0
        alg = CudaBroxOpticalFlow(gamma=gamma)
        assert alg.gamma == gamma

    def test_inner_iterations_property(self):
        iterations = 100.0
        alg = CudaBroxOpticalFlow(inner_iterations=iterations)
        assert alg.inner_iterations == iterations

    def test_outer_iterations_property(self):
        iterations = 100.0
        alg = CudaBroxOpticalFlow(outer_iterations=iterations)
        assert alg.outer_iterations == iterations

    def test_solver_iterations_property(self):
        iterations = 100.0
        alg = CudaBroxOpticalFlow(solver_iterations=iterations)
        assert alg.solver_iterations == iterations

    def test_repr(self):
        assert repr(CudaBroxOpticalFlow(alpha=1.0, scale_factor=0.8)) == \
               "CudaBroxOpticalFlow(alpha=1.0, gamma=50.0, scale_factor=0.8, " \
               "inner_iterations=5, outer_iterations=150, solver_iterations=10)"
