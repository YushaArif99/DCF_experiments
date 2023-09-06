import ivy
from control_flow_experimental.ivy_fx.fx.proxy import Proxy, IvyProxy, NativeProxy
from typing import Any, Iterable, Union, Optional, Dict, Tuple, Callable
import inspect
import sys

###########
# Helpers #
###########
def _to_native(x: Any, inplace: bool = False, to_ignore: tuple = ()) -> Any:
    to_ignore = ivy.default(to_ignore, ())
    if isinstance(x, to_ignore):
        return x
    if isinstance(x, IvyProxy):
        return NativeProxy(node=x.node, tracer=x.tracer, native_data=x.data)
    # TODO: add support for ShapeProxies
    elif type(x) is ivy.Shape:
        return x.shape
    elif isinstance(x, ivy.Container):
        return x.cont_map(
            lambda x_, _: _to_native(x_, inplace=inplace, to_ignore=to_ignore),
            inplace=inplace,
        )
    return x


def _to_ivy(x: Any) -> Any:
    if isinstance(x, IvyProxy):
        return x
    elif isinstance(x, ivy.NativeShape):  # TODO: add support for ShapeProxies
        return ivy.Shape(x)
    elif isinstance(x, ivy.Container):
        return x.to_ivy()
    if isinstance(x, NativeProxy):
        return IvyProxy(node=x.node, tracer=x.tracer, data=x._native_data)
    return x


def _args_to_native(
    *args: Iterable[Any],
    include_derived: Dict[type, bool] = None,
    cont_inplace: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    """
    Return args and keyword args in their native backend framework form for all nested
    IvyProxy instances, otherwise the arguments are returned unchanged.

    Parameters
    ----------
    args
        The positional arguments to check
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is ``False``.
    cont_inplace
        Whether to update containers in place.
        Default is ``False``
    to_ignore
        Types to ignore when deciding whether to go deeper into the nest or not
    kwargs
        The key-word arguments to check

    Returns
    -------
     ret
        the same arguments, with any nested IvyProxy instances converted to their
        native form.
    """
    native_args = ivy.nested_map(
        args,
        lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
        include_derived,
        shallow=False,
    )
    native_kwargs = ivy.nested_map(
        kwargs,
        lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
        include_derived,
        shallow=False,
    )
    return native_args, native_kwargs


def _args_to_ivy(
    *args: Iterable[Any],
    include_derived: Optional[Dict[type, bool]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    """
    Return args and keyword args in their IvyProxy form for all nested instances,
    otherwise the arguments are returned unchanged.

    Parameters
    ----------
    args
        The positional arguments to check
    include_derived
        Whether to also recursive for classes derived from tuple, list and dict.
        Default is ``False``.
    kwargs
        The key-word arguments to check

    Returns
    -------
     ret
        the same arguments, with any nested NativeProxy instances converted to IvyProxies
    """
    native_args = ivy.nested_map(args, _to_ivy, include_derived, shallow=False)
    native_kwargs = ivy.nested_map(kwargs, _to_ivy, include_derived, shallow=False)
    return native_args, native_kwargs
    

#####################################
# Proxy-equivalent implementations  #
#####################################
def to_ivy(
    x: Union[ivy.Array, ivy.NativeArray, Iterable],
    nested: bool = False,
    include_derived: Optional[Dict[type, bool]] = None,
) -> Union[ivy.Array, ivy.NativeArray, Iterable]:
    if nested:
        return ivy.nested_map(x, _to_ivy, include_derived, shallow=False)
    return _to_ivy(x)

def args_to_ivy(
    *args: Iterable[Any],
    include_derived: Optional[Dict[type, bool]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    native_args = ivy.nested_map(args, _to_ivy, include_derived, shallow=False)
    native_kwargs = ivy.nested_map(kwargs, _to_ivy, include_derived, shallow=False)
    return native_args, native_kwargs 

def to_native(
    x: Union[ivy.Array, ivy.NativeArray, Iterable],
    nested: bool = False,
    include_derived: Optional[Dict[type, bool]] = None,
    cont_inplace: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
) -> Union[ivy.Array, ivy.NativeArray, Iterable]:
    if nested:
        return ivy.nested_map(
            x,
            lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
            include_derived,
            shallow=False,
        )
    return _to_native(x, inplace=cont_inplace, to_ignore=to_ignore)

def args_to_native(
    *args: Iterable[Any],
    include_derived: Dict[type, bool] = None,
    cont_inplace: bool = False,
    to_ignore: Optional[Union[type, Tuple[type]]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Iterable[Any], Dict[str, Any]]:
    native_args = ivy.nested_map(
        args,
        lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
        include_derived,
        shallow=False,
    )
    native_kwargs = ivy.nested_map(
        kwargs,
        lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
        include_derived,
        shallow=False,
    )
    return native_args, native_kwargs


def is_ivy_array(
    x: Union[IvyProxy, NativeProxy], /, *, exclusive: Optional[bool] = False
) -> bool:
    return isinstance(x, IvyProxy) or is_ivy_array._orig_fn(x, exclusive=exclusive)

def is_native_array(
    x: Union[IvyProxy, NativeProxy], /, *, exclusive: bool = False
) -> bool:
    return isinstance(x, NativeProxy) or is_native_array._orig_fn(x, exclusive=exclusive)




ALL_FUNCS = {name: func for name, func in sys.modules[__name__].__dict__.items() if inspect.isfunction(func)}