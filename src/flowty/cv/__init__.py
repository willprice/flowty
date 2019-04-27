from flowty.cv.core import Mat


def mat_to_array(mat):
    """Convert OpenCV Mat to np.ndarray"""
    return mat.asarray()


def array_to_mat(array):
    """Convert np.ndarray to OpenCV Mat"""

    # We copy the data as when the np.ndarray goes out of scope it's data will be
    # released and we can't take ownership any other way :(
    return Mat.fromarray(array, copy=True)