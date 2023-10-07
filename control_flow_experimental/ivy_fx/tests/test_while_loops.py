import graph_compiler.globals as glob 
import control_flow_experimental.ivy_fx.fx as fx 

import ivy 
import numpy as np 
import pytest 

def basic_while(n):
    i = 0
    sum = 0
    while i < n:
        sum += i
        i += 1
    return sum

def while_with_break(n):
    i = 0
    sum = 0
    while i < n:
        if i == n // 2:
            break
        sum += i
        i += 1
    return sum

def while_with_continue(n):
    i = 0
    sum = 0
    while i < n:
        i += 1
        if i == n // 2:
            continue
        sum += i
    return sum

def nested_while(n, m):
    i = 0
    j = 0
    count = 0
    while i < n:
        while j < m:
            count += j
            j += 1
        i += 1
    return count

def infinite_loop_with_break(n):
    result = 0
    while True:
        result += n 
        if result > n * 10:
            break 
    return result 

def static_while(x):
    i = 0
    while i < 10:
        x+= i
        i += 1
    return x  

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("n", [(5),(10)])
def test_basic_while(fw, n):
    orig_fn = basic_while
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(n,), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(n,), traced_fn(n,))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("n", [(5),(10)])
def test_while_with_break(fw, n):
    orig_fn = while_with_break
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(n,), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(n,), traced_fn(n,))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("n", [(5),(10)])
def test_while_with_continue(fw, n):
    orig_fn = while_with_continue
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(n,), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(n,), traced_fn(n,))


@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("n", [(5),(10)])
def test_infinite_loop_with_break(fw, n):
    orig_fn = infinite_loop_with_break
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(n,), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(n,), traced_fn(n,))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("n, m", [(5,10),(13,12)])
def test_nested_while(fw, n, m):
    orig_fn = nested_while
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(n,m), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(n,m), traced_fn(n,m))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x", [(3),(-12)])
def test_static_while(fw, x):
    orig_fn = static_while
    ivy.set_backend(fw)
    glob.dummy_trace = True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x,), traced_fn(x,))
    # graph should be unrolled so there shouldn't be a while_loop op
    graph_fns = [fn.__name__ for fn in traced_fn._functions] 
    assert all(["while_loop" not in f for f in graph_fns])
