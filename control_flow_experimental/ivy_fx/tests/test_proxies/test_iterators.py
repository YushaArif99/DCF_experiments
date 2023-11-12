import pytest
import numpy as np
import graph_compiler.globals as glob 
import control_flow_experimental.ivy_fx.fx as fx 
import ivy 
from copy import copy

def zip_operations(x, y):
    res = list(zip(x, y))
    return res 

def enumerate_operations(x):
    res = [(i, v) for i, v in enumerate(x)]
    return res 

def range_operations(x):
    res = [i for i in range(x)]
    return res 


@pytest.mark.parametrize("x, y", [(list(range(5)), list(range(5, 10))), (list(range(10, 15)), list(range(15, 20)))])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_zip_operations(x , y, mode):
    orig_fn = zip_operations
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))

@pytest.mark.parametrize("x", [list(range(5)), list(range(10))])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_enumerate_operations(x,mode):
    orig_fn = enumerate_operations
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x)), traced_fn(copy(x)))

@pytest.mark.parametrize("x", [5, 10])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_range_operations(x, mode):
    orig_fn = range_operations
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,), control_flow=True, generate_source=True,mode=mode)
    assert np.allclose(orig_fn(copy(x)), traced_fn(copy(x)))
