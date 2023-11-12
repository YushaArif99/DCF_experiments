import pytest
import numpy as np
import graph_compiler.globals as glob 
import control_flow_experimental.ivy_fx.fx as fx 
import ivy 
from copy import copy

def tuple_operations_1(x, y):
    x = x + (10,)
    y = y[:-1]
    res = x + y
    res = (15.0,) + res[1:]
    return res 

def tuple_operations_2(x, y):
    x = x + (20,)
    y = y[1:]
    res = x[:len(x)//2] + y[len(y)//2:]
    res = (25.0,) + res[1:]
    return res 

def tuple_operations_3(x, y):
    a, *b, c, = x
    b.append(a + c)
    b.sort()
    y = list(y)
    y.reverse()
    res = tuple(b) + tuple(y)
    res = (30.0,) + res[1:]
    return res 

@pytest.mark.parametrize("x, y", [(tuple([1, 2, 3]), tuple([4, 5, 6])), (tuple([7, 8, 9]), tuple([10, 11, 12]))])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_tuple_operations_1(x , y, mode):
    orig_fn = tuple_operations_1
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))

@pytest.mark.parametrize("x, y", [(tuple([13, 14, 15]), tuple([16, 17, 18])), (tuple([19, 20, 21]), tuple([22, 23, 24]))])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_tuple_operations_2(x , y, mode):
    orig_fn = tuple_operations_2
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))

@pytest.mark.parametrize("x, y", [(tuple([25, 26, 27]), tuple([28, 29, 30])), (tuple([31, 32, 33]), tuple([34, 35, 36]))])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_tuple_operations_3(x , y, mode):
    orig_fn = tuple_operations_3
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))
