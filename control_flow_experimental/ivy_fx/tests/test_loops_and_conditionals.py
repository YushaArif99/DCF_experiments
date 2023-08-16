import pytest
import control_flow_experimental.ivy_fx.fx as fx
import graph_compiler.globals as glob 

import ivy 
import numpy as np
import torch 

def simple_while_loop(res,i):
    while i < 10:
        res += i
        i += 1
    return res

def if_else_with_unused_arg(x,y, z=None,):
    res = torch.sin(x) + torch.cos(y)
    if z is not None: 
        res += torch.tan(z) 
    return res 


@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("res, i", [([1], 1,), ([2], 2)])
def test_simple_while_loop(fw, res, i):
    orig_fn = simple_while_loop
    ivy.set_backend(fw)
    _, traced_fn = fx.symbolic_trace(orig_fn, args=(res,), constant_args={'i': i}, generate_source=True)
    assert np.allclose(orig_fn(ivy.array(res), i), traced_fn(ivy.array(res), i))

@pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
@pytest.mark.parametrize("x,y,z", [([1.0], [2.0], [3.0]), ([1.0], [2.0], None)])
def test_transpile_if_else_unused_arg(fw,x,y,z):
    if fw in ["tensorflow"]:
        """TODO: with_unsupported_dtype in the frontends is an important decorator that shouldnt get unwrapped. 
        Fix this issue."""
        pytest.skip()

    orig_fn = if_else_with_unused_arg
    ivy.set_backend('torch')
    x_torch = ivy.native_array(x)
    y_torch = ivy.native_array(y)
    z_torch = ivy.native_array(z) if z is not None else z

    _, traced_fn = fx.symbolic_trace(orig_fn, args=(x_torch,y_torch,z_torch),generate_source=False)
    glob.do_dummy_trace=True
    traced_fn.reload_sourcecode(frontend='torch')
    ivy.set_backend(fw)
    x = ivy.native_array(x)
    y = ivy.native_array(y)
    z = ivy.native_array(z) if z is not None else z
    _, traced_fn = fx.symbolic_trace(traced_fn, frontend='torch', args=(x,y,z), generate_source=True)
    
    assert np.allclose(orig_fn(x_torch,y_torch,z_torch), traced_fn(x,y,z))

"""TODO: add these while loop tests once while_loop is handled correctly."""
# def find_largest_prime_number(numbers, largest_prime):
#     def is_prime(x):
#         if x < 2:
#             return False
#         i = 2
#         while i <= x // 2:
#             if x % i == 0:
#                 return False
#             i += 1
#         return True
    
#     i = 0
#     while i < len(numbers):
#         if is_prime(numbers[i]):
#             if numbers[i] > largest_prime:
#                 largest_prime = numbers[i]
#         i += 1
#     return largest_prime


# def find_largest_prime_factor(numbers, largest_prime_factor):
#     def find_prime_factors(x):
#         prime_factors = torch.tensor([])
#         while x > 1:
#             if x % i == 0:
#                 prime_factors = torch.concat((prime_factors, torch.tensor([i]) ) )
#                 x /= i
#             else:
#                 i += 1
#         return prime_factors

#     i = 0
#     while i < len(numbers):
#         prime_factors = find_prime_factors(numbers[i])
#         if len(prime_factors) > 0:
#             if prime_factors[-1] > largest_prime_factor:
#                 largest_prime_factor = prime_factors[-1]
#         i += 1
#     return largest_prime_factor



# def find_even_odd_numbers(nums, res):
#     numbers = torch.tensor(nums)
#     even_numbers = torch.tensor([])
#     odd_numbers = torch.tensor([])
#     i = 0
#     while i < len(numbers):
#         if numbers[i] % 2 == 0:
#             even_numbers = torch.concat((even_numbers, torch.tensor([numbers[i]])) )
#         else:
#             odd_numbers = torch.concat((odd_numbers, torch.tensor([numbers[i]])))
#         i += 1
#     if len(numbers) %2 ==0:
#         return even_numbers
#     else:
#         return odd_numbers

# @pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
# @pytest.mark.parametrize("numbers,largest_prime", [([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0]), ([-3, 12, 31, 9, 11],[0])])
# def test_find_largest_prime_number(fw, numbers,largest_prime):
#     orig_fn = find_largest_prime_number
#     ivy.set_backend(fw)
#     _, traced_fn = fx.symbolic_trace(orig_fn, args=(numbers,largest_prime), generate_source=True)
#     assert np.allclose(orig_fn(ivy.array(numbers), largest_prime), traced_fn(ivy.array(numbers),largest_prime))

# @pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
# @pytest.mark.parametrize("numbers, largest_prime_factor", [([13, 25, 19, 55, 81, 105, 37],[0]), ([-71,15, 2, 88, 1024, 6],[0])])
# def test_find_largest_prime_factor(fw, numbers, largest_prime_factor):
#     orig_fn = find_largest_prime_factor
#     ivy.set_backend(fw)
#     _, traced_fn = fx.symbolic_trace(orig_fn, args=(numbers,largest_prime_factor), generate_source=True)
#     assert np.allclose(orig_fn(numbers,largest_prime_factor), traced_fn(numbers,largest_prime_factor))

# @pytest.mark.parametrize("fw", ['jax', 'tensorflow', 'torch', 'numpy'])
# @pytest.mark.parametrize("nums", [[1, 2, 3, 4], [5, 6, 7]])
# def test_find_even_odd_numbers(fw, nums):
#     orig_fn = find_even_odd_numbers
#     ivy.set_backend(fw)
#     _, traced_fn = fx.symbolic_trace(orig_fn, args=(nums,), generate_source=True)
#     assert np.allclose(orig_fn(nums), traced_fn(nums))
