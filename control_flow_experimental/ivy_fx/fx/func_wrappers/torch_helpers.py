import ivy
import ivy.functional.frontends.torch as torch_frontend
from control_flow_experimental.ivy_fx.fx.proxy import (
    IvyProxy,
    NativeProxy,
    Torch_FrontendProxy,
)
from typing import Callable

import functools


def _from_ivy_proxies_to_torch_frontend_proxies(
    x, nested=False, include_derived=None, frontend=None
):
    if nested:
        return ivy.nested_map(
            x,
            _from_ivy_proxies_to_torch_frontend_proxies,
            include_derived,
            shallow=False,
        )
    elif isinstance(x, IvyProxy):
        a = Torch_FrontendProxy(node=x.node, tracer=x.tracer, data=x._ivy_data, ivy_proxy=x)
        return a
    elif isinstance(x, NativeProxy):
        ivy_proxy = IvyProxy(node=x.node, tracer=x.tracer, data=x._native_data, native_proxy=x)
        a = Torch_FrontendProxy(node=x.node, tracer=x.tracer, data=x._native_data, ivy_proxy=ivy_proxy)
        return a
    return x


def _to_ivy_proxy(x):
    # if x is a native proxy return it as an ivy proxy
    if isinstance(x, NativeProxy):
        return IvyProxy(node=x.node, tracer=x.tracer, data=x._native_data, native_proxy=x)

    # else if x is a frontend proxy, return the wrapped ivy proxy # noqa: E501
    elif hasattr(x, "ivy_proxy"):
        return x.ivy_proxy

    # else just return x
    return x


"""TorchFrontendProxies"""


def inputs_to_ivy_proxies_torch(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_proxies_torch(*args, **kwargs):
        """
        Convert `TorchFrontendProxy` into `IvyProxy` instances.

        Convert all `TorchFrontendProxy` instances in both the positional and
        keyword arguments into `IvyProxy` instances, and then calls the
        function with the updated arguments.
        """
        # Remove out argument if present in kwargs
        if "out" in kwargs and not ivy.nested_any(
            kwargs["out"], lambda x: isinstance(x, (Torch_FrontendProxy, type(None)))
        ):
            raise ivy.utils.exceptions.IvyException(
                "Out argument must be a TorchFrontendProxy object"
            )
        # convert all frontend proxies to ivy proxy instances
        new_args = ivy.nested_map(
            args, _to_ivy_proxy, include_derived={tuple: True}, shallow=False
        )
        new_kwargs = ivy.nested_map(
            kwargs, _to_ivy_proxy, include_derived={tuple: True}, shallow=False
        )
        return fn(*new_args, **new_kwargs)

    _inputs_to_ivy_proxies_torch.inputs_to_ivy_proxies_torch = True
    return _inputs_to_ivy_proxies_torch


def outputs_to_frontend_proxies_torch(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_frontend_proxies_torch(*args, **kwargs):
        """
        Convert `IvyProxy` into `TorchFrontendProxy` instances.

        Call the function, and then converts all `IvyProxy` instances
        returned by the function into `TorchFrontendProxy` instances.
        """
        # call unmodified function
        # ToDo: Remove this default dtype setting
        #  once frontend specific backend setting is added
        set_default_dtype = False
        if not ("dtype" in kwargs and ivy.exists(kwargs["dtype"])) and all(
            [not (ivy.is_array(i) or hasattr(i, "ivy_array")) for i in args]
        ):
            if ivy.current_backend_str() == "jax":
                import jax

                jax.config.update("jax_enable_x64", True)
            ivy.set_default_int_dtype("int64")
            ivy.set_default_float_dtype(torch_frontend.get_default_dtype())
            set_default_dtype = True
        try:
            ret = fn(*args, **kwargs)
        finally:
            if set_default_dtype:
                ivy.unset_default_int_dtype()
                ivy.unset_default_float_dtype()

        ret = _from_ivy_proxies_to_torch_frontend_proxies(
            ret, nested=True, include_derived={tuple: True}
        )
        array_fn = lambda x: ivy.is_array(x) or hasattr(x, "ivy_array")
        if "inplace" in kwargs and kwargs["inplace"]:
            first_array = ivy.func_wrapper._get_first_array(
                *args, array_fn=array_fn, **kwargs
            )
            # ivy.inplace_update with ensure_in_backend=True fails in jax and tf
            # so update ._data directly
            # TODO: make sure inplace update works with proxies
            if ivy.is_array(first_array._meta_tensor):
                first_array._data = ret.ivy_array._data
            else:
                first_array.ivy_array._data = ret.ivy_array._data
            return first_array
        else:
            return ret

    _outputs_to_frontend_proxies_torch.outputs_to_frontend_proxies_torch = True
    return _outputs_to_frontend_proxies_torch
