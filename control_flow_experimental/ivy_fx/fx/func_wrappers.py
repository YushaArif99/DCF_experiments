import ivy
#from ivy.func_wrapper import FN_DECORATORS
from .proxy import IvyProxy,Proxy, FRONTEND_PROXIES

import functools
import inspect
import sys

arg_ivyproxy_list = []
kwarg_ivyproxy_list = []

arg_frontendproxy_list = []
kwarg_frontendproxy_list = []

FN_DECORATORS = [
    "infer_device",
    "infer_dtype",
    "handle_array_function",
    "integer_arrays_to_float",
    #"outputs_to_ivy_arrays",
    "outputs_to_ivy_shapes",
    #"outputs_to_native_arrays",
    #"inputs_to_native_arrays",
    "inputs_to_native_shapes",
    #"inputs_to_ivy_arrays",
    "handle_out_argument",
    "handle_view_indexing",
    "handle_view",
    "handle_array_like_without_promotion",
    "handle_partial_mixed_function",
    "handle_nestable",
    "handle_exceptions",
    "handle_nans",
]

ASARRAY_DECORATORS = [
    "asarray_infer_device",
    "asarray_inputs_to_native_shapes",
    "asarray_to_native_arrays_and_back",
    "asarray_handle_nestable",
]


def add_custom_decorator(custom_decorators, func, positions):
    """
    Adds a decorator to a function at a specific position.
    :param func: The function to add the decorator to.
    :param decorators: The decorators to add.
    :param positions: The positions at which to add the decorator. 0 is the first decorator and -1 is the last decorator.
    """
    actual_decorators = []
    for decorator in FN_DECORATORS + ASARRAY_DECORATORS + [
        "proxies_to_native_arrays",
        "native_arrays_to_proxies",
        "proxies_to_ivy_arrays",
        "ivy_arrays_to_proxies",
        "handle_numpy_arrays_in_specific_backend"
    ]:
        if hasattr(func, decorator):
            decorator_func = (
                getattr(ivy, decorator)
                if hasattr(ivy, decorator)
                else getattr(sys.modules[__name__], decorator)
            )
            actual_decorators.append(decorator_func)
    # reverse the list so that the first decorator is at index 0
    actual_decorators.reverse()
    for custom_dec, pos in zip(custom_decorators, positions):
        if pos == 0:
            actual_decorators.insert(0, custom_dec)
        elif pos == -1:
            actual_decorators.append(custom_dec)
        else:
            actual_decorators.insert(pos, custom_dec)

    # apply the decorators back to the function in reverse order
    func = inspect.unwrap(func)
    for decorator in reversed(actual_decorators):
        func = decorator(func)
    return func


def proxies_to_native_arrays(func):
    @functools.wraps(func)
    def _proxies_to_native_arrays(*args, **kwargs):

        def convert_to_native(proxy):
            return proxy._ivy_data.data
        
        arg_pxy_indices = ivy.nested_argwhere(
            args, lambda a: isinstance(a, Proxy)
        )
        proxy_args = ivy.multi_index_nest(args, arg_pxy_indices)
        args = ivy.map_nest_at_indices(
            args, arg_pxy_indices, convert_to_native, shallow=True
        )

        kwarg_pxy_indices = ivy.nested_argwhere(
            kwargs, lambda a: isinstance(a, Proxy)
        )
        proxy_kwargs = ivy.multi_index_nest(kwargs, kwarg_pxy_indices)
        kwargs = ivy.map_nest_at_indices(
            kwargs, kwarg_pxy_indices, convert_to_native, shallow=True
        )

        if arg_pxy_indices == []:
            proxy_args = []
        if kwarg_pxy_indices == []:
            proxy_kwargs = []   
        arg_ivyproxy_list.append((proxy_args, arg_pxy_indices))
        kwarg_ivyproxy_list.append((proxy_kwargs, kwarg_pxy_indices))
    
        result = func(*args, **kwargs)
        return result

    setattr(_proxies_to_native_arrays, "__name__", func.__name__)
    _proxies_to_native_arrays.proxies_to_native_arrays = True
    return _proxies_to_native_arrays


def native_arrays_to_proxies(func):
    @functools.wraps(func)
    def _native_arrays_to_proxies(*args, **kwargs): 
        
        def convert_to_proxy(native_arr, proxy):
            proxy._ivy_data.data = native_arr
            return proxy

        proxy_args, arg_indices = arg_ivyproxy_list.pop()
        proxy_kwargs, kwarg_indices = kwarg_ivyproxy_list.pop()
        
        for p_arg,i in zip(proxy_args,arg_indices):
            args = ivy.map_nest_at_index(args,i, lambda a: convert_to_proxy(a, p_arg) )

        for p_kwarg,i in zip(proxy_kwargs,kwarg_indices):
            kwargs = ivy.map_nest_at_index(kwargs,i, lambda a: convert_to_proxy(a, p_kwarg) )
    
        result = func(*args, **kwargs)
        return result
    
    setattr(_native_arrays_to_proxies, "__name__", func.__name__)
    _native_arrays_to_proxies.native_arrays_to_proxies = True
    return _native_arrays_to_proxies

def proxies_to_native_arrays_and_back(func):
    return native_arrays_to_proxies(proxies_to_native_arrays(func))

# Frontends

def proxies_to_ivy_arrays(func):
    @functools.wraps(func)
    def _proxies_to_ivy_arrays(*args, **kwargs):
        
        def convert_to_ivy(proxy):
            return proxy._ivy_data
        
        arg_pxy_indices = ivy.nested_argwhere(
            args, lambda a: isinstance(a, Proxy)
        )
        proxy_args = ivy.multi_index_nest(args, arg_pxy_indices)
        args = ivy.map_nest_at_indices(
            args, arg_pxy_indices, convert_to_ivy, shallow=True
        )

        kwarg_pxy_indices = ivy.nested_argwhere(
            kwargs, lambda a: isinstance(a, Proxy)
        )
        proxy_kwargs = ivy.multi_index_nest(kwargs, kwarg_pxy_indices)
        kwargs = ivy.map_nest_at_indices(
            kwargs, kwarg_pxy_indices, convert_to_ivy, shallow=True
        )

        if arg_pxy_indices == []:
            proxy_args = []
        if kwarg_pxy_indices == []:
            proxy_kwargs = []    
        arg_frontendproxy_list.append((proxy_args, arg_pxy_indices))
        kwarg_frontendproxy_list.append((proxy_kwargs, kwarg_pxy_indices))
    
        result = func(*args, **kwargs)
        return result

    setattr(_proxies_to_ivy_arrays, "__name__", func.__name__)
    _proxies_to_ivy_arrays.proxies_to_ivy_arrays = True
    return _proxies_to_ivy_arrays


def ivy_arrays_to_proxies(func):
    @functools.wraps(func)
    def _ivy_arrays_to_proxies(*args, **kwargs):
        
        def convert_to_proxy(ivy_arr, proxy):
            proxy._ivy_data = ivy_arr
            return proxy

        proxy_args, arg_indices = arg_frontendproxy_list.pop()
        proxy_kwargs, kwarg_indices = kwarg_frontendproxy_list.pop()
        
        for p_arg,i in zip(proxy_args,arg_indices):
            args = ivy.map_nest_at_index(args,i, lambda a: convert_to_proxy(a, p_arg) )

        for p_kwarg,i in zip(proxy_kwargs,kwarg_indices):
            kwargs = ivy.map_nest_at_index(kwargs,i, lambda a: convert_to_proxy(a, p_kwarg) )
    
        result = func(*args, **kwargs)
        return result
        
    setattr(_ivy_arrays_to_proxies, "__name__", func.__name__)
    _ivy_arrays_to_proxies.ivy_arrays_to_proxies = True
    return _ivy_arrays_to_proxies

def proxies_to_ivy_arrays_and_back(func):
    return ivy_arrays_to_proxies(proxies_to_ivy_arrays(func))

def convert_proxies_to_ivy_arrays(func):
    @functools.wraps(func)
    def _convert_proxies_to_ivy_arrays(*args, **kwargs):
        
        args = ivy.nested_map(args, lambda a: a._ivy_data if isinstance(a, Proxy) else a)
        kwargs = ivy.nested_map(kwargs, lambda a: a._ivy_data if isinstance(a, Proxy) else a)

        return func(*args, **kwargs)

    setattr(_convert_proxies_to_ivy_arrays, "__name__", func.__name__)
    _convert_proxies_to_ivy_arrays.convert_proxies_to_ivy_arrays = True
    return _convert_proxies_to_ivy_arrays
