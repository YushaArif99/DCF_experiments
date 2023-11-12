import pytest
import numpy as np
import graph_compiler.globals as glob 
import control_flow_experimental.ivy_fx.fx as fx 
import ivy 
from copy import copy

def list_operations_1(x, y):
    x.append(10)
    y.pop()
    x.append(100)
    res = x + y
    val = y.pop(0)
    res[0] = 15.0 
    res[1] = val ** 2
    return res 

def list_operations_2(x, y):
    x.append(20)
    y.remove(y[0])
    x.insert(1, 200)
    res = x[:len(x)//2] + y[len(y)//2:]
    val = y.pop(-1)
    res[0] = 25.0 
    res[1] = val ** 3
    return res 

def list_operations_3(x, y):
    a, *b, c, = x
    b.append(a + c)
    b.sort()
    y.reverse()
    res = b + y
    val = res.pop(0)
    res[0] = 30.0 
    res[1] = val ** 4
    return res 

@pytest.mark.parametrize("x, y", [([1, 2, 3], [4, 5, 6]), ([7, 8, 9], [10, 11, 12])])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_list_operations_1(x , y, mode):
    orig_fn = list_operations_1
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))

@pytest.mark.parametrize("x, y", [([13, 14, 15], [16, 17, 18]), ([19, 20, 21], [22, 23, 24])])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_list_operations_2(x , y, mode):
    orig_fn = list_operations_2
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))


@pytest.mark.parametrize("x, y", [([25, 26, 27], [28, 29, 30]), ([31, 32, 33], [34, 35, 36])])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_list_operations_3(x , y, mode):
    orig_fn = list_operations_3
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))
