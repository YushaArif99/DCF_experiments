import ivy
import ivy.functional.frontends.jax as jax_frontend
from control_flow_experimental.ivy_fx.fx.proxy import (
    IvyProxy,
    NativeProxy,
    JAX_FrontendProxy,
)
from typing import Any, Callable

import functools


"""JAXFrontendProxies"""


def _from_jax_frontend_proxy_to_ivy_proxy(x):
    if isinstance(x, JAX_FrontendProxy) and x.weak_type and x.ivy_array.shape == ():
        setattr(x.ivy_array, "weak_type", True)
        return x.ivy_array 
    if hasattr(x, "ivy_array"):
        return x.ivy_array 

    if isinstance(x, NativeProxy):
        return IvyProxy(node=x.node, tracer=x.tracer, data=x._native_data, native_proxy=x)
    return x


def _from_ivy_proxy_to_jax_frontend_proxy(x, nested=False, include_derived=None):
    if nested:
        return ivy.nested_map(
            x, _from_ivy_proxy_to_jax_frontend_proxy, include_derived, shallow=False
        )
    elif isinstance(x, IvyProxy):
        return JAX_FrontendProxy(node=x.node, tracer=x.tracer, data=x._ivy_data, ivy_proxy=x)
    return x


def _from_ivy_proxy_to_jax_frontend_proxy_weak_type(
    x, nested=False, include_derived=None
):
    if nested:
        return ivy.nested_map(
            x,
            _from_ivy_proxy_to_jax_frontend_proxy_weak_type,
            include_derived,
            shallow=False,
        )
    elif isinstance(x, IvyProxy):
        return JAX_FrontendProxy(
            node=x.node, tracer=x.tracer, data=x._ivy_data, ivy_proxy=x, weak_type=True
        )
    return x


def inputs_to_ivy_proxies_jax(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_proxies_jax(*args, **kwargs):
        # check if kwargs contains an out argument, and if so, remove it
        has_out = False
        out = None
        if "out" in kwargs:
            out = kwargs["out"]
            del kwargs["out"]
            has_out = True
        # convert all proxies in the inputs to IvyProxy instances
        new_args = ivy.nested_map(
            args,
            _from_jax_frontend_proxy_to_ivy_proxy,
            include_derived={tuple: True},
            shallow=False,
        )
        new_kwargs = ivy.nested_map(
            kwargs,
            _from_jax_frontend_proxy_to_ivy_proxy,
            include_derived={tuple: True},
            shallow=False,
        )
        # add the original out argument back to the keyword arguments
        if has_out:
            new_kwargs["out"] = out
        return fn(*new_args, **new_kwargs)

    _inputs_to_ivy_proxies_jax.inputs_to_ivy_proxies_jax = True
    return _inputs_to_ivy_proxies_jax


def outputs_to_frontend_proxies_jax(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_frontend_proxies_jax(*args, **kwargs):
        weak_type = not any(
            (isinstance(arg, JAX_FrontendProxy) and arg.weak_type is False)
            or (isinstance(arg, IvyProxy) and arg.weak_type is False)
            or isinstance(arg, (tuple, list))
            for arg in args
        )
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            weak_type = False
        # call unmodified function
        # ToDo: Remove this default dtype setting
        #  once frontend specific backend setting is added
        if jax_frontend.config.jax_enable_x64:
            ivy.set_default_int_dtype("int64")
            ivy.set_default_float_dtype("float64")
            try:
                ret = fn(*args, **kwargs)
            finally:
                ivy.unset_default_int_dtype()
                ivy.unset_default_float_dtype()
        else:
            ret = fn(*args, **kwargs)
        # convert all arrays in the return to `jax_frontend.DeviceArray` instances
        if weak_type:
            return _from_ivy_proxy_to_jax_frontend_proxy_weak_type(
                ret,
                nested=True,
                include_derived={tuple: True},
            )
        return _from_ivy_proxy_to_jax_frontend_proxy(
            ret, nested=True, include_derived={tuple: True}
        )

    _outputs_to_frontend_proxies_jax.outputs_to_frontend_proxies_jax = True
    return _outputs_to_frontend_proxies_jax
