import ivy

from typing import Callable
import functools


# TODO: maybe add a ShapeProxy class
def _shape_to_native(x):
    # checks the first element of the leaf list and
    # converts it to a native array if it is an ivy array
    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x = list(x) if isinstance(x, tuple) else x
            x[i] = _shape_to_native(item)
    else:
        if (isinstance(x, (list, tuple)) and len(x) > 0) and (
            isinstance(x[0], ivy.Shape) and ivy.array_mode
        ):
            x = ivy.nested_map(x, lambda x: x.shape if isinstance(x, ivy.Shape) else x)
        elif isinstance(x, ivy.Shape) and ivy.array_mode:
            x = x.shape
    return x


def asarray_inputs_to_native_shapes(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_native_shapes(*args, **kwargs):
        new_arg = _shape_to_native(args[0])
        new_args = (new_arg,) + args[1:]
        return fn(*new_args, **kwargs)

    _inputs_to_native_shapes.inputs_to_native_shapes = True
    return _inputs_to_native_shapes


def inputs_to_native_shapes(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_native_shapes(*args, **kwargs):
        args, kwargs = ivy.nested_map(
            [args, kwargs],
            lambda x: (x.shape if isinstance(x, ivy.Shape) and ivy.array_mode else x),
        )
        return fn(*args, **kwargs)

    _inputs_to_native_shapes.inputs_to_native_shapes = True
    return _inputs_to_native_shapes


def outputs_to_ivy_shapes(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_ivy_shapes(*args, **kwargs):
        args, kwargs = ivy.nested_map(
            [args, kwargs],
            lambda x: (x.shape if isinstance(x, ivy.Shape) and ivy.array_mode else x),
        )
        return fn(*args, **kwargs)

    _outputs_to_ivy_shapes.outputs_to_ivy_shapes = True
    return _outputs_to_ivy_shapes


def to_native_shapes_and_back(fn: Callable) -> Callable:
    """
    Make `fn` receive `ivy.NativeShape` and return `ivy.Shape`.

    Wrap `fn` so that input shapes are all converted to
    `ivy.NativeShape` instances and return shapes are all converted to
    `ivy.Shape` instances.
    """
    return outputs_to_ivy_shapes(inputs_to_native_shapes(fn))
