import pytest
import control_flow_experimental.ivy_fx.fx as fx
import graph_compiler.globals as glob
import ivy 
import numpy as np

def for_enum(x,y,itr): 
    for i,val in enumerate(itr):
        x = x + y + val
    return x

def for_range(x, sum,start,stop,step): 
    for i in range(start,stop,step):
        sum += x[i] * i
    return sum 

def for_enum_with_locals(x,y,itr):
    res = 1 
    for i,val in enumerate(itr):
        res += x + y + val
    return x

def for_itr_with_locals(data):
    sum_i = 0
    sum_j = 0
    sum_k = 0
    for i, (j, k) in data:
        sum_i += i
        sum_j += j
        sum_k += k

    return sum_i - sum_j - sum_k 

def for_enum_static(x,y): 
    for i,val in enumerate([1,2,3]):
        x = x + y + val
    return x

def for_range_static(x, sum): 
    for i in range(4):
        sum += x[i] * i
    return sum 

def for_zip(x, y): 
    res = []
    for i, j in zip(x, y):
        res = res + i + j
    return res

def for_in_list_comp(itr):
    #TODO: add support for tracing list ops (eg: list.pop(), list.append() etc.)
    res = [i+j for i,j in itr]
    return res


@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y, itr", [(0, 0, [1,2,3,4]), (-4, 11, [-10,22,1, 16,110, 77,-32])])
def test_for_enum(fw, x, y, itr):
    orig_fn = for_enum
    ivy.set_backend(fw)
    glob.dummy_trace=True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y,itr), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x,y,itr), traced_fn(x,y,itr))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, sum, start, stop, step", [([1,2,3,4], 0, 0, 4, 1), ([16,22, -100, 11, 19, 108, -3], 0, 6, 0, -1)])
def test_for_range(fw, x,sum,start,stop,step):
    orig_fn = for_range
    ivy.set_backend(fw)
    glob.dummy_trace=True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, sum,start,stop,step,), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, sum,start,stop,step), traced_fn(x, sum,start,stop,step))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y, itr", [(1, 1, [11,12,-9]), (-2, -10, [1,2,3,4,19,55,6,7,3,21])])
def test_for_enum_with_locals(fw, x,y,itr):
    orig_fn = for_enum_with_locals
    ivy.set_backend(fw)
    glob.dummy_trace=True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y,itr), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x,y,itr), traced_fn(x,y,itr))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("data", [[(1, [2, 3]), (4, [5, 6]), (7, [8, 9])], [(6, [12, -3]), (40, [50, -16]), (7, [-28, 9])]])
def test_for_itr_with_locals(fw, data):
    orig_fn = for_itr_with_locals
    ivy.set_backend(fw)
    glob.dummy_trace=True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(data,), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(data), traced_fn(data))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x,sum", [([1,2,3,4,5], 0), ([11,12,100,-19,33,74,-319], 12)])
def test_for_range_static(fw, x,sum):
    orig_fn = for_range_static
    ivy.set_backend(fw)
    glob.dummy_trace=True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,sum), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x,sum), traced_fn(x,sum))
    # graph should be unrolled so there shouldn't be a while_loop op
    graph_fns = [fn.__name__ for fn in traced_fn._functions] 
    assert all(["while_loop" not in f for f in graph_fns])

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x,y", [(1,2), (16, -5)])
def test_for_enum_static(fw, x,y):
    orig_fn = for_enum_static
    ivy.set_backend(fw)
    glob.dummy_trace=True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x,y), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x,y), traced_fn(x,y))
    # graph should be unrolled so there shouldn't be a while_loop op
    graph_fns = [fn.__name__ for fn in traced_fn._functions] 
    assert all(["while_loop" not in f for f in graph_fns])

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x, y", [([[1,2,3,4]], [[5,6,7,8]]), ([[1,2,3], [10,11,12]], [[4,5,6,7,8], [-4,-5,-6,-7,-8]])])
def test_for_zip(fw, x, y):
    orig_fn = for_zip
    ivy.set_backend(fw)
    glob.dummy_trace=True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x, y,), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(x, y), traced_fn(x, y))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("itr", [[(1,2),(3,4)], [(-1,-2),(-19,100),(-3,-7),(15,31)]])
def test_for_in_list_comp(fw, itr):
    #TODO: remove this once list comprehension is handled correctly
    pytest.skip()
    orig_fn = for_in_list_comp
    ivy.set_backend(fw)
    glob.dummy_trace=True
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(itr), control_flow=True, generate_source=True)
    assert np.allclose(orig_fn(itr), traced_fn(itr))