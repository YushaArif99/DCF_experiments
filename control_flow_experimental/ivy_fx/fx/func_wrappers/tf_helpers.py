import ivy
from control_flow_experimental.ivy_fx.fx.proxy import (
    IvyProxy,
    NativeProxy,
    TF_FrontendProxy,
)
from typing import Callable

import functools

"""TFFrontendProxy"""


def _tf_frontend_proxy_to_ivy(x):
    if hasattr(x, "ivy_proxy"):
        return x.ivy_proxy
    return x


def _ivy_proxy_to_tensorflow(x):
    if isinstance(x, IvyProxy):
        return TF_FrontendProxy(node=x.node, tracer=x.tracer, data=x._ivy_data, ivy_proxy=x)
    elif isinstance(x, NativeProxy):
        ivy_proxy = IvyProxy(node=x.node, tracer=x.tracer, data=x._native_data, native_proxy=x)
        return TF_FrontendProxy(node=x.node, tracer=x.tracer, data=x._native_data, ivy_proxy=ivy_proxy)
    return x


def inputs_to_ivy_proxies_tf(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_proxies_tf(*args, **kwargs):
        """
        Convert all `TFFrontendProxy` instances in both the positional and keyword
        arguments into `IvyProxy` instances, and then calls the function with the
        updated arguments.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with ivy arrays passed in the arguments.
        """
        has_out = False
        out = None
        if "out" in kwargs:
            out = kwargs["out"]
            del kwargs["out"]
            has_out = True

        # convert all arrays in the inputs to IvyProxy instances
        ivy_args = ivy.nested_map(
            args, _ivy_proxy_to_tensorflow, include_derived=True, shallow=False
        )
        ivy_kwargs = ivy.nested_map(
            kwargs, _ivy_proxy_to_tensorflow, include_derived=True, shallow=False
        )
        if has_out:
            ivy_kwargs["out"] = out
        return fn(*ivy_args, **ivy_kwargs)

    _inputs_to_ivy_proxies_tf.inputs_to_ivy_proxies_tf = True
    return _inputs_to_ivy_proxies_tf


def outputs_to_frontend_proxies_tf(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_frontend_proxies_tf(*args, **kwargs):
        """
        Call the function, and then converts all `IvyProxy` instances in the
        function return into `TFFrontendProxy` instances.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with ivy proxies as frontend proxies.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)

        # convert all arrays in the return to `frontend.Tensorflow.tensor` instances
        return ivy.nested_map(
            ret, _tf_frontend_proxy_to_ivy, include_derived={tuple: True}
        )

    _outputs_to_frontend_proxies_tf.outputs_to_frontend_proxies_tf = True
    return _outputs_to_frontend_proxies_tf
