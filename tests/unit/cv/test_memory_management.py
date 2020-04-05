import sys
from copy import copy

from . import _test_memory_management as tests

def collect_test_fns(module_name, tests_module):
    for obj_name in dir(tests_module):
        if obj_name.startswith("test_"):
            _test_fn = getattr(tests, obj_name)

            def wrapped_test_fn(_test_fn=_test_fn):
                _test_fn()

            wrapped_test_fn.__name__ = obj_name
            setattr(sys.modules[module_name], obj_name, wrapped_test_fn)

collect_test_fns(__name__, tests)

# def test_ref_count_of_wrapped_cv_mat_is_2():
#     tests.test_ref_count_of_wrapped_cv_mat_is_2()
#
# def test_ref_count_of_cv_mat_after_conversion_to_numpy_is_greater_than_original_refcount():
#     tests.test_ref_count_of_cv_mat_after_conversion_to_numpy_is_greater_than_original_refcount()
#
# def test_fail():
#     tests.test_fail()
