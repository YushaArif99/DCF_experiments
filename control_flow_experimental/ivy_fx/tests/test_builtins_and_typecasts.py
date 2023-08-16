import pytest
import ivy
import control_flow_experimental.ivy_fx.fx as fx
import numpy as np

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
def test_bool_cast(fw):
    def bool_cast(x):
        return bool(x)

    orig_fn = bool_cast
    ivy.set_backend(fw)
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(ivy.array(1),), generate_source=True)
    assert np.allclose(orig_fn(ivy.array(1)), traced_fn(ivy.array(1)))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
def test_int_float_cast(fw):
    def int_float_cast(x):
        return int(x), float(x)

    orig_fn = int_float_cast
    ivy.set_backend(fw)
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(ivy.array(1.5),), generate_source=True)
    assert np.allclose(orig_fn(ivy.array(1.5)), traced_fn(ivy.array(1.5)))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
def test_min_max(fw):
    def min_max(x):
        return min(x), max(x)

    orig_fn = min_max
    ivy.set_backend(fw)
    _, traced_fn = fx.symbolic_trace(orig_fn, args=([1, 2, 3],), generate_source=True)
    assert np.allclose(orig_fn([1, 2, 3]), traced_fn([1, 2, 3]))

"""TODO: add sourcegen support to import builtin modules eg: math, itertools etc."""
# @pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
# def test_math_functions(fw):
#     import math

#     def math_functions(x):
#         return math.sin(x), math.cos(x)

#     orig_fn = math_functions
#     ivy.set_backend(fw)
#     _, traced_fn = fx.symbolic_trace(orig_fn, args=(ivy.array(1.0),), generate_source=True)
#     assert np.allclose(orig_fn(ivy.array(1.0)), traced_fn(ivy.array(1.0)))
