import pytest
import numpy as np
import graph_compiler.globals as glob 
import control_flow_experimental.ivy_fx.fx as fx 
import ivy 
from copy import copy

def set_operations_1(x, y):
    x.add(10)
    y.remove(5)
    res = x.union(y)
    res.add(15.0)
    return res 

def set_operations_2(x, y):
    inter = x.intersection(y)
    diff = x.difference(y)
    res = inter.union(diff)
    res.add(20.0)
    return res 

def set_operations_3(x, y):
    is_subset = x.issubset(y)
    is_superset = x.issuperset(y)
    res = x.union(y) if is_subset else x.intersection(y)
    res.add(is_superset * 25.0)
    return res 


@pytest.mark.parametrize("x, y", [(set([1, 2, 3]), set([4, 5, 6])), (set([7, 8, 9]), set([10, 11, 12]))])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_set_operations_1(x , y, mode):
    orig_fn = set_operations_1
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))

@pytest.mark.parametrize("x, y", [(set([13, 14, 15]), set([16, 17, 18])), (set([19, 20, 21]), set([22, 23, 24]))])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_set_operations_2(x , y, mode):
    orig_fn = set_operations_2
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))

@pytest.mark.parametrize("x, y", [(set([25, 26, 27]), set([28, 29, 30])), (set([31, 32, 33]), set([34, 35, 36]))])
@pytest.mark.parametrize("mode", ('SYMBOLIC', 'EAGER'))
def test_set_operations_3(x , y, mode):
    orig_fn = set_operations_3
    ivy.set_backend("numpy") # setting random backend
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True, mode=mode)
    assert np.allclose(orig_fn(copy(x),copy(y)), traced_fn(copy(x),copy(y)))
