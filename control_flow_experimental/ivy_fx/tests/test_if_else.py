import graph_compiler.globals as glob 
import control_flow_experimental.ivy_fx.fx as fx 

import ivy 
import numpy as np 
import pytest 


def if_else_basic(x, y):
    if x > y:
        return x + y
    else:
        return x - y

def if_elif_else(x, y):
    if x > y:
        return x / y
    elif x == y:
        return x + y
    else:
        return y - x

def if_else_nested(x, y, z):
    if x > y:
        if y < z:
            return x * z
        else:
            return x * y
    else:
        return y - z

def if_else_logical_pred(x, y, z):
    if x > 0 and y > z:
        return x + y + z
    else:
        return x - y - z

def if_else_None_pred(x, y, z=None):
    if z is None:
        return x + y 
    else:
        return x - y - z

def if_else_bool_pred(x, y, flag=False):
    if flag:
        return x + y
    else:
        return x - y

def if_else_with_locals(x, y):
    w = list(range(x))
    if x > y:
        w[0] = x + y 
    else:
        w[0] = x - y 
    return w    

def if_else_static(x, y):
    if False:
        return x + y
    else:
        return x - y
    
def if_expr(x, y):
    return x + y if x > 0 and y > 0  else x - y

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y", [(1,2), (-5, -3)])
def test_if_else_basic(fw, x , y):
    orig_fn = if_else_basic
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, y), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, y), traced_fn(x, y))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y", [(1,2), (-5, -3)])
def test_if_elif_else(fw, x , y):
    orig_fn = if_elif_else
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, y), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, y), traced_fn(x, y))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x,y,z", [(2,1,3), (3,2,1), (-5, -3, 10)])
def test_if_else_nested(fw, x , y, z):
    orig_fn = if_else_nested
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, y, z), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, y, z), traced_fn(x, y, z))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y, z", [(2,3,1), (3,1,2), (-5, -3, 10)])
def test_if_else_logical_pred(fw, x , y, z):
    orig_fn = if_else_logical_pred
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, y, z), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, y, z), traced_fn(x, y, z))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y, z", [(1,2,3), (5,6, None)])
def test_if_else_None_pred(fw, x , y, z):
    orig_fn = if_else_None_pred
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, y, z), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, y, z), traced_fn(x, y, z))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y, flag", [(1,2, True,), (6,7, False)])
def test_if_else_bool_pred(fw, x , y, flag):
    orig_fn = if_else_bool_pred
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, y, flag), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, y, flag), traced_fn(x, y, flag))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y", [(5,10), (10,20)])
def test_if_else_with_locals(fw, x , y):
    orig_fn = if_else_with_locals
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, y), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, y), traced_fn(x, y))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y", [(1,2), (-5, -3)])
def test_if_else_static(fw, x , y):
    orig_fn = if_else_static
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, y), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, y), traced_fn(x, y))
    # should only trace one of the branches so there should'nt be an if_else/cond op
    graph_fns = [fn.__name__ for fn in traced_fn._functions] 
    assert not any([op in f for op in ("if_else", "cond") for f in graph_fns])

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y", [(1,2), (-1,-2)])
def test_if_expr(fw, x , y):
    orig_fn = if_expr
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, y), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, y), traced_fn(x, y))