import pytest
import numpy as np
import graph_compiler.globals as glob 
import control_flow_experimental.ivy_fx.fx as fx 
import ivy 
from copy import copy

def dict_operations_1(x, y):
    x['a'] = 10
    val = y.get('b', 0)
    keys = list(x.keys())
    res = {**x, **y}
    res[keys[0]] = val ** 2
    return res 

def dict_operations_2(x, y):
    items = list(x.items())
    y.pop('c', None)
    res = {k: v for k, v in items if v > 2}
    val = res.pop('a', 0)
    res['d'] = val ** 3
    return res 

def dict_operations_3(x, y):
    x.update(y)
    values = list(y.values())
    res = {k: v for k, v in x.items() if v in values}
    val = res.pop('b', 0)
    res['e'] = val ** 4
    return res 


@pytest.mark.parametrize("x, y", [({'a': 1, 'b': 2}, {'c': 3, 'd': 4}), ({'e': 5, 'f': 6}, {'g': 7, 'h': 8})])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_dict_operations_1(x , y, mode):
    orig_fn = dict_operations_1
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))

@pytest.mark.parametrize("x, y", [({'i': 9, 'j': 10}, {'k': 11, 'l': 12}), ({'m': 13, 'n': 14}, {'o': 15, 'p': 16})])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_dict_operations_2(x , y,mode):
    orig_fn = dict_operations_2
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))

@pytest.mark.parametrize("x, y", [({'q': 17, 'r': 18}, {'s': 19, 't': 20}), ({'u': 21, 'v': 22}, {'w': 23, 'x': 24})])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_dict_operations_3(x , y, mode):
    orig_fn = dict_operations_3
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))
