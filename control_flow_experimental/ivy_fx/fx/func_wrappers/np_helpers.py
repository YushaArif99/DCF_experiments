import ivy
from control_flow_experimental.ivy_fx.fx.proxy import (
    IvyProxy,
    NativeProxy,
    Numpy_FrontendProxy,
)
from typing import Any, Callable

import functools
import inspect
import platform

""""NumpyFrontendProxy"""


def _numpy_frontend_to_ivy(x: Any) -> Any:
    if hasattr(x, "ivy_array"):
        return IvyProxy(node=x.node, tracer=x.tracer, data=x.ivy_array)
    else:
        return x


def _ivy_to_numpy(x: Any) -> Any:
    if isinstance(x, (IvyProxy, NativeProxy)):
        return Numpy_FrontendProxy(
            node=x.node, tracer=x.tracer, shape=x, _init_overload=True
        )
    else:
        return x


def _ivy_to_numpy_order_F(x: Any) -> Any:
    if isinstance(x, (IvyProxy, NativeProxy)):
        a = Numpy_FrontendProxy(
            node=x.node, tracer=x.tracer, shape=0, order="F"
        )  # TODO Find better initialisation workaround
        a.ivy_array = x
        return a
    else:
        return x


def _check_C_order(x):
    if isinstance(x, IvyProxy):
        return True
    elif isinstance(x, Numpy_FrontendProxy):
        if x._f_contiguous:
            return False
        else:
            return True
    else:
        return None


def _set_order(args, order):
    ivy.utils.assertions.check_elem_in_list(
        order,
        ["C", "F", "A", "K", None],
        message="order must be one of 'C', 'F', 'A', or 'K'",
    )
    if order in ["K", "A", None]:
        check_order = ivy.nested_map(
            args, _check_C_order, include_derived={tuple: True}, shallow=False
        )
        if all(v is None for v in check_order) or any(
            ivy.multi_index_nest(check_order, ivy.all_nested_indices(check_order))
        ):
            order = "C"
        else:
            order = "F"
    return order


def inputs_to_ivy_proxies_np(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_proxies_np(*args, **kwargs):
        """
        Convert `NumpyFrontendProxy` into `IvyProxy` instances.

        Convert all `NumpyFrontendProxy` instances in both the positional and keyword arguments
        into `IvyProxy` instances, and then calls the function with the updated
        arguments.

        Parameters
        ----------
        args
            The arguments to be passed to the function.
        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with ivy proxies passed in the arguments.
        """
        # convert all proxies in the inputs to IvyProxy instances
        ivy_args = ivy.nested_map(
            args, _numpy_frontend_to_ivy, include_derived={tuple: True}
        )
        ivy_kwargs = ivy.nested_map(
            kwargs, _numpy_frontend_to_ivy, include_derived={tuple: True}
        )
        return fn(*ivy_args, **ivy_kwargs)

    _inputs_to_ivy_proxies_np.inputs_to_ivy_proxies_np = True
    return _inputs_to_ivy_proxies_np


def outputs_to_frontend_proxies_np(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_frontend_proxies_np(*args, order="K", **kwargs):
        """
        Convert `IvyProxy` into `NumpyFrontendProxy` instances.

        Call the function, and then converts all `IvyProxy` instances
        returned by the function into `NumpyFrontendProxy` instances.

        The return of the function, with ivy proxies as frontend proxies.
        """
        # handle order and call unmodified function
        # ToDo: Remove this default dtype setting
        #  once frontend specific backend setting is added
        set_default_dtype = False
        if not ("dtype" in kwargs and ivy.exists(kwargs["dtype"])) and any(
            [not (ivy.is_array(i) or hasattr(i, "ivy_array")) for i in args]
        ):
            if ivy.current_backend_str() == "jax":
                import jax

                jax.config.update("jax_enable_x64", True)
            (
                ivy.set_default_int_dtype("int64")
                if platform.system() != "Windows"
                else ivy.set_default_int_dtype("int32")
            )
            ivy.set_default_float_dtype("float64")
            set_default_dtype = True
        if contains_order:
            if len(args) >= (order_pos + 1):
                order = args[order_pos]
                args = args[:-1]
            order = _set_order(args, order)
            try:
                ret = fn(*args, order=order, **kwargs)
            finally:
                if set_default_dtype:
                    ivy.unset_default_int_dtype()
                    ivy.unset_default_float_dtype()
        else:
            try:
                ret = fn(*args, **kwargs)
            finally:
                if set_default_dtype:
                    ivy.unset_default_int_dtype()
                    ivy.unset_default_float_dtype()
        if not ivy.array_mode:
            return ret
        # convert all returned arrays to `ndarray` instances
        if order == "F":
            return ivy.nested_map(
                ret, _ivy_to_numpy_order_F, include_derived={tuple: True}
            )
        else:
            return ivy.nested_map(ret, _ivy_to_numpy, include_derived={tuple: True})

    if "order" in list(inspect.signature(fn).parameters.keys()):
        contains_order = True
        order_pos = list(inspect.signature(fn).parameters).index("order")
    else:
        contains_order = False
    _outputs_to_frontend_proxies_np.outputs_to_frontend_proxies_np = True
    return outputs_to_frontend_proxies_np
