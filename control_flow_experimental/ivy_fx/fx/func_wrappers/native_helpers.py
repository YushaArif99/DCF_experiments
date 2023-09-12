import ivy
from .replacement_funcs import _args_to_ivy, _args_to_native, _to_ivy, _to_native
from control_flow_experimental.ivy_fx.fx.proxy import Proxy, IvyProxy, NativeProxy
from typing import Any, Iterable, Union, Optional, Dict, Tuple, Callable
import functools
import warnings


####################################
# wrapper for ivy utility functions  #
####################################
def convert_proxies_to_ivy_arrays(func):
    @functools.wraps(func)
    def _convert_proxies_to_ivy_arrays(*args, **kwargs):
        args = ivy.nested_map(
            args, lambda a: a._meta_tensor if isinstance(a, Proxy) else a, shallow=False,
        )
        kwargs = ivy.nested_map(
            kwargs, lambda a: a._meta_tensor if isinstance(a, Proxy) else a, shallow=False,
        )

        return func(*args, **kwargs)

    _convert_proxies_to_ivy_arrays.convert_proxies_to_ivy_arrays = True
    return _convert_proxies_to_ivy_arrays

####################################
# asarray NativeProxy Conversions  #
####################################
def _ivy_to_native(x):
    # checks the first element of the leaf list and
    # converts it to a native proxy if it is an ivy proxy
    # assumes that either all elements in a leaf list are ivy proxies
    # or none of them are
    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x = list(x) if isinstance(x, tuple) else x
            x[i] = _ivy_to_native(item)
    else:
        if (isinstance(x, (list, tuple)) and len(x) > 0) and isinstance(x[0], IvyProxy):
            x = to_native(x, nested=True)
        elif isinstance(x, IvyProxy):
            x = to_native(x)
    return x


def asarray_to_native_proxies_and_back(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _asarray_to_native_proxies_and_back(*args, dtype=None, **kwargs):
        """
        Wrap `fn` so that input proxies are all converted to `NativeProxy` instances
        and return proxies are all converted to `IvyProxy` instances.

        This wrapper is specifically for the backend implementations of
        asarray.

        It assumes either all the elements in a leaf list are ivy proxies
        or none of them are. It checks the first element of all the leaf
        list. If it is an IvyProxy, it converts all the elements in the
        leaf list to NativeProxy otherwise it skips that leaf list.
        """
        new_arg = _ivy_to_native(args[0])
        new_args = (new_arg,) + args[1:]
        if dtype is not None:
            dtype = ivy.default_dtype(dtype=dtype, as_native=True)
        return to_ivy(fn(*new_args, dtype=dtype, **kwargs))

    _asarray_to_native_proxies_and_back.asarray_to_native_proxies_and_back = True
    return _asarray_to_native_proxies_and_back




def to_ivy(
    x: Union[IvyProxy, NativeProxy, Iterable],
    nested: bool = False,
    include_derived: Optional[Dict[type, bool]] = None,
) -> Union[IvyProxy, NativeProxy, Iterable]:
    """
    Return the input proxies converted to an IvyProxy instance if it is a native proxy
    type, otherwise the input is returned unchanged. If nested is set, the check is
    applied to all nested leafs of tuples, lists and dicts contained within x.

    Parameters
    ----------
    x
        The input to be converted.
    nested
        Whether to apply the conversion on arguments in a nested manner. If so, all
        dicts, lists and tuples will be traversed to their lowest leaves in search of
        IvyProxy instances. Default is ``False``.
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict. Default
        is False.

    Returns
    -------
    ret
        the input in its native framework form in the case of IvyProxy or instances.
    """
    if nested:
        return ivy.nested_map(x, _to_ivy, include_derived, shallow=False)
    return _to_ivy(x)


def to_native(
    x: Union[IvyProxy, NativeProxy, Iterable],
    nested: bool = False,
    include_derived: Optional[Dict[type, bool]] = None,
    cont_inplace: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
) -> Union[IvyProxy, NativeProxy, Iterable]:
    """Return the input item in its native backend framework form if it is an
    IvyProxy instance, otherwise the input is returned unchanged. If nested is set,
    the check is applied to all nested leaves of tuples, lists and dicts contained
    within ``x``.

    Parameters
    ----------
    x
        The input to maybe convert.
    nested
        Whether to apply the conversion on arguments in a nested manner. If so, all
        dicts, lists and tuples will be traversed to their lowest leaves in search of
        IvyProxy instances. Default is ``False``.
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is ``False``.
    cont_inplace
        Whether to update containers in place. Default is ``False``
    to_ignore
        Types to ignore when deciding whether to go deeper into the nest or not

    Returns
    -------
     ret
        the input in its native framework form in the case of IvyProxy instances.
    """
    if nested:
        return ivy.nested_map(
            x,
            lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
            include_derived,
            shallow=False,
        )
    return _to_native(x, inplace=cont_inplace, to_ignore=to_ignore)


##################
#  IVY WRAPPERS  #
##################


def inputs_to_native_proxies(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_native_proxies(*args, **kwargs):
        """
        Convert all `IvyProxies` instances in both the positional and keyword arguments
        into `NativeProxies` instances, and then calls the function with the updated
        arguments.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with native proxies passed in the arguments.
        """
        if not ivy.array_mode:
            return fn(*args, **kwargs)
        # check if kwargs contains an out argument, and if so, remove it
        has_out = False
        out = None
        if "out" in kwargs:
            out = kwargs["out"]
            del kwargs["out"]
            has_out = True
        # convert all proxies in the inputs to NativeProxy instances
        new_args, new_kwargs = _args_to_native(*args, **kwargs)
        # add the original out argument back to the keyword arguments
        if has_out:
            new_kwargs["out"] = out
        return fn(*new_args, **new_kwargs)

    _inputs_to_native_proxies.inputs_to_native_proxies = True
    return _inputs_to_native_proxies


def inputs_to_ivy_proxies(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_ivy_proxies(*args, **kwargs):
        """
        Convert all `NativeProxies` instances in both the positional and keyword arguments
        into `IvyProxies` instances, and then calls the function with the updated
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
        if not ivy.array_mode:
            warnings.warn(
                "In the case of Compositional function, operators might cause"
                " inconsistent behavior when array_mode is set to False"
            )
            return fn(*args, **kwargs)

        has_out = False
        if "out" in kwargs:
            out = kwargs["out"]
            has_out = True
        # convert all proxies in the inputs to IvyProxy instances
        ivy_args, ivy_kwargs = _args_to_ivy(
            *args, **kwargs, include_derived={tuple: True}
        )
        if has_out:
            ivy_kwargs["out"] = out
        return fn(*ivy_args, **ivy_kwargs)

    _inputs_to_ivy_proxies.inputs_to_ivy_proxies = True
    return _inputs_to_ivy_proxies


def outputs_to_ivy_proxies(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_ivy_proxies(*args, **kwargs):
        """
        Call the function, and then converts all `NativeProxy` instances in the
        function return into `IvyProxy` instances.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with native proxies as ivy proxies.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)
        # convert all proxies in the return to `IvyProxy` instances
        return (
            to_ivy(ret, nested=True, include_derived={tuple: True})
            if ivy.array_mode
            else ret
        )

    _outputs_to_ivy_proxies.outputs_to_ivy_proxies = True
    return _outputs_to_ivy_proxies


def outputs_to_native_proxies(fn: Callable) -> Callable:
    """
    Call the function, and then converts all `IvyProxy` instances in the function
    return into `NativeProxy` instances.

    Parameters
    ----------
    args
        The arguments to be passed to the function.

    kwargs
        The keyword arguments to be passed to the function.

    Returns
    -------
        The return of the function, with ivy proxys as native proxys.
    """

    @functools.wraps(fn)
    def _outputs_to_native_proxies(*args, **kwargs):
        ret = fn(*args, **kwargs)
        return to_native(ret, nested=True, include_derived={tuple: True})

    _outputs_to_native_proxies.outputs_to_native_proxies = True
    return _outputs_to_native_proxies
