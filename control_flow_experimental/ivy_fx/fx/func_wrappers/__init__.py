from .native_helpers import (
    inputs_to_ivy_proxies,
    inputs_to_native_proxies,
    outputs_to_ivy_proxies,
    outputs_to_native_proxies,
    asarray_to_native_proxies_and_back,
    convert_proxies_to_ivy_arrays,
)
from .np_helpers import inputs_to_ivy_proxies_np, outputs_to_frontend_proxies_np
from .jax_helpers import inputs_to_ivy_proxies_jax, outputs_to_frontend_proxies_jax
from .tf_helpers import inputs_to_ivy_proxies_tf, outputs_to_frontend_proxies_tf
from .torch_helpers import (
    inputs_to_ivy_proxies_torch,
    outputs_to_frontend_proxies_torch,
)
from .replacement_funcs import ALL_FUNCS 

import ivy
from ivy.func_wrapper import FN_DECORATORS

import inspect
import sys

#####################
#  CUSTOM WRAPPING  #
#####################

replacement_dict = {
    "inputs_to_ivy_arrays": inputs_to_ivy_proxies,
    "inputs_to_native_arrays": inputs_to_native_proxies,
    "outputs_to_ivy_arrays": outputs_to_ivy_proxies,
    "outputs_to_native_arrays": outputs_to_native_proxies,
}

frontend_replacement_dict = {
    "torch": [inputs_to_ivy_proxies_torch, outputs_to_frontend_proxies_torch],
    "numpy": [inputs_to_ivy_proxies_np, outputs_to_frontend_proxies_np],
    "tensorflow": [inputs_to_ivy_proxies_tf, outputs_to_frontend_proxies_tf],
    "jax": [inputs_to_ivy_proxies_jax, outputs_to_frontend_proxies_jax],
}

dtype_device_dict = {
    "unsupported_dtypes": ivy.with_unsupported_dtypes,
    "supported_dtypes": ivy.with_supported_dtypes,
    "unsupported_devices": ivy.with_unsupported_devices,
    "supported_devices": ivy.with_supported_devices,
    "unsupported_device_and_dtypes": ivy.with_unsupported_device_and_dtypes,
    "supported_device_and_dtypes": ivy.with_supported_device_and_dtypes,
}

asarray_decorators = [
    asarray_to_native_proxies_and_back,
    ivy.asarray_infer_device,
    ivy.asarray_handle_nestable,
    ivy.asarray_inputs_to_native_shapes,
]


def replace_decorators(func, frontend=None):
    """
    Replaces some decorators of a function with others using the global replacement dict.
    :param func: The function to replace the decorators of.
    """
    actual_decorators = []
    if func.__name__ == "asarray":
        # asarray contains asarray-specific decorators
        # so we will handle it seperately
        actual_decorators = asarray_decorators
    else:
        for decorator in (
            FN_DECORATORS
            + list(dtype_device_dict.keys())
            + [
                "handle_numpy_arrays_in_specific_backend",
            ]
        ):
            if hasattr(func, decorator):
                # check if the decorator is in the replacement dict
                if decorator in replacement_dict:
                    # get the proxy version of the decorator
                    decorator_func = replacement_dict[decorator]

                # handle supported/unsupported dtype and device decorators
                elif decorator in dtype_device_dict:
                    wrapper_cls = dtype_device_dict[decorator]
                    attr = decorator
                    attr_val = getattr(func, decorator)
                    if not hasattr(func, "dictionary_info"):
                        # some functions dont use the `with_unsupported_dtype` decorator
                        # for example but instead they directly set the supported_dtypes
                        # as an attribute of the function
                        # eg: einops_reduce(https://github.com/unifyai/ivy/blob/main/ivy/functional/ivy/general.py#L1991)

                        # we create a placeholder function to re-attach the original attribute
                        def wrapper(func):
                            setattr(func, attr, attr_val)
                            return func

                        decorator_func = wrapper
                    else:
                        decorator_func = wrapper_cls(
                            func.dictionary_info, func.version_info
                        )
                else:
                    decorator_func = (
                        getattr(ivy, decorator)
                        if hasattr(ivy, decorator)
                        else getattr(sys.modules[__name__], decorator)
                    )

                actual_decorators.append(decorator_func)
    if frontend:
        actual_decorators.extend(frontend_replacement_dict[frontend])

    # re-apply the decorators back to the function in reverse order
    original_func_dict = func.__dict__
    func = inspect.unwrap(func)
    for decorator in reversed(actual_decorators):
        func = decorator(func)

    # update the __dict__ of the new function
    for key, value in original_func_dict.items():
        if key in replacement_dict:
            new_key = replacement_dict[key].__name__
        else:
            new_key = key
        setattr(func, new_key, value)

    return func

def get_replacement_func(func):
   try:
    new_fn = ALL_FUNCS[func.__name__]
   except KeyError:
    raise KeyError(f"No replacement function found for {func.__name__}")
   
   return new_fn
