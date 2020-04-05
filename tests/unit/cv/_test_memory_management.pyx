from flowty.cv.c_core cimport Mat as c_Mat, CV_8UC3
from flowty.cv.core cimport Mat
from flowty.cv.core import Mat
import numpy as np


cpdef test_ref_count_of_wrapped_cv_mat_is_2():
    cdef c_Mat cv_allocated_mat = c_Mat(1, 1, CV_8UC3)
    cdef int original_refcount = cv_allocated_mat.u.refcount

    cdef Mat python_mat = Mat.from_mat(cv_allocated_mat)

    cdef int expected_refcount = original_refcount + 1
    cdef int actual_refcount = cv_allocated_mat.u.refcount
    assert actual_refcount == expected_refcount, \
      "Expected refcount to be {} but was {}".format(expected_refcount, actual_refcount)


cpdef test_ref_count_of_cv_mat_after_conversion_to_numpy_is_greater_than_original_refcount():
    cdef c_Mat cv_allocated_mat = c_Mat(1, 1, CV_8UC3)
    cdef int original_refcount = cv_allocated_mat.u.refcount

    array = np.asarray(Mat.from_mat(cv_allocated_mat))

    cdef int actual_refcount = cv_allocated_mat.u.refcount
    assert actual_refcount == original_refcount + 1, \
        "Expected refcount to be {} but was {}".format(original_refcount, actual_refcount)

