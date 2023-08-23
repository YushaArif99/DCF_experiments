import pytest
import ivy
import control_flow_experimental.ivy_fx.fx as fx
import numpy as np

"""TODO: add support via transpilation for JAX and TF"""
@pytest.mark.parametrize("fw", ['torch', 'numpy'])
def test_setitem(fw):
    def setitem(x):
        y = x[0]
        y[0] = 3.0
        return y
    orig_fn = setitem
    ivy.set_backend(fw)
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(ivy.array([[10.,20.,30.,]]),), generate_source=True)
    assert np.allclose(orig_fn(ivy.array([[-2.,16.,5.,100]])), traced_fn(ivy.array([[-2.,16.,5.,100]])))
