import numpy as np
from numpy.testing import assert_array_equal

from flowty.imgproc import quantise_flow


class TestQuantiseFlow:
    def test_max_values_are_mapped_to_upper_bound(self):
        flow = np.array([20], dtype=np.float32)
        quantised_flow = quantise_flow(flow, bound=20)
        assert_array_equal(np.array([255], dtype=np.uint8), quantised_flow)

    def test_values_above_bound_are_mapped_to_upper_bound(self):
        flow = np.array([25], dtype=np.float32)
        quantised_flow = quantise_flow(flow, bound=20)
        assert_array_equal(np.array([255], dtype=np.uint8), quantised_flow)

    def test_min_values_are_mapped_to_lower_bound(self):
        flow = np.array([-20], dtype=np.float32)
        quantised_flow = quantise_flow(flow, bound=20)
        assert_array_equal(np.array([0], dtype=np.uint8), quantised_flow)

    def test_values_below_bound_are_mapped_to_lower_bound(self):
        flow = np.array([-25], dtype=np.float32)
        quantised_flow = quantise_flow(flow, bound=20)
        assert_array_equal(np.array([0], dtype=np.uint8), quantised_flow)

    def test_0_is_mapped_to_127(self):
        flow = np.array([0], dtype=np.float32)
        quantised_flow = quantise_flow(flow, bound=20)
        assert_array_equal(np.array([255//2], dtype=np.uint8), quantised_flow)

    def test_half_bound_is_mapped_to_3_quarters_of_255(self):
        bound = 20
        flow = np.array([bound / 2], dtype=np.float32)
        quantised_flow = quantise_flow(flow, bound=20)
        assert_array_equal(np.array([3 * (255 / 4)], dtype=np.uint8), quantised_flow)

    def test_negative_half_bound_is_mapped_to_quarter_of_255(self):
        bound = 20
        flow = np.array([-bound / 2], dtype=np.float32)
        quantised_flow = quantise_flow(flow, bound=20)
        assert_array_equal(np.array([(255 / 4)], dtype=np.uint8), quantised_flow)

