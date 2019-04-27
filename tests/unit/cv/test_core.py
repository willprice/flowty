import numpy as np
from flowty.cv.core import Mat, CV_64F, CV_8U, CV_8UC2, CV_8UC3, CV_32FC2, CV_32FC3, CV_64FC3
from numpy.testing import assert_equal


class TestMat:
    def test_creation(self):
        Mat()

    def test_non_emtpy_creation(self):
        Mat(rows=10, cols=20)

    def test_rows_property(self):
        n_rows = 10
        mat = Mat(rows=n_rows, cols=20)
        assert mat.rows == n_rows

    def test_cols_property(self):
        n_cols = 20
        mat = Mat(rows=10, cols=n_cols)
        assert mat.cols == n_cols

    def test_default_dtype_is_CV_8UC3(self):
        mat = Mat(rows=5, cols=5)
        assert mat.dtype == CV_8UC3

    def test_channels(self):
        mat = Mat(rows=5, cols=5, dtype=CV_8UC3)
        assert mat.channels == 3

    def test_empty(self):
        mat = Mat(rows=0, cols=0)
        assert mat.empty

    def test_ndim(self):
        mat = Mat(rows=0, cols=0, dtype=CV_8UC3)
        assert mat.ndim == 3

    def test_depth(self):
        mat = Mat(rows=1, cols=1, dtype=CV_64FC3)
        assert mat.depth == CV_64F

    def test_shape(self):
        mat = Mat(rows=10, cols=20, dtype=CV_8UC3)
        assert mat.shape == (10, 20, 3)

    def test_empty_array_is_false(self):
        mat = Mat()
        assert bool(mat) == False

    def test_non_empty_array_is_true(self):
        mat = Mat(rows=1, cols=1)
        assert bool(mat)

    def test_repr(self):
        mat = Mat(rows=1, cols=2)
        assert repr(mat) == "Mat(rows=1, cols=2, dtype=16)"

    def test_asarray_uint(self):
        mat = Mat(rows=1, cols=2, dtype=CV_8UC3)
        array = mat.asarray()
        assert array.dtype == np.uint8
        assert array.shape == (1, 2, 3)

    def test_asarray_float(self):
        mat = Mat(rows=1, cols=2, dtype=CV_32FC2)
        array = mat.asarray()
        assert array.dtype == np.float32
        assert array.shape == (1, 2, 2)
        del array

    def test_fromarray_maintains_shape(self):
        array = np.zeros((5, 10, 3), dtype=np.uint8)
        mat = Mat.fromarray(array)
        assert (5, 10, 3) == mat.shape

    def test_fromarray_preserves_data(self):
        array = np.random.randint(0, 255, size=(4, 5, 3), dtype=np.uint8)
        mat = Mat.fromarray(array)
        mat_as_array = mat.asarray()
        assert_equal(array, mat_as_array)

    def test_setting_value_in_buffer(self):
        mat = Mat(rows=1, cols=2, dtype=CV_32FC3)
        array1 = mat.asarray()
        array1[0, 0, :] = [0, 1, 2]
        array2 = mat.asarray()
        assert np.array_equal(array2[0, 0, :], np.array([0, 1, 2]))
        del array1
        del array2
