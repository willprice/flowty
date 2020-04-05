from flowty.cv.core import Mat


def mat_to_array(mat):
    """Convert OpenCV Mat to np.ndarray"""
    return mat.asarray()


def array_to_mat(array):
    """Convert np.ndarray to OpenCV Mat"""
    return Mat.fromarray(array)