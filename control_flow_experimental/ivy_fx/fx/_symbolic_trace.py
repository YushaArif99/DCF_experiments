import builtins
import copy
import functools
import inspect
import math
import warnings
import collections
import importlib
from itertools import chain
from types import CodeType, FunctionType, ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    Iterable,
)

"""TODO (yusha): remove all torch dependencies"""
import torch
from . import _pytree as pytree

# from torch._C import ScriptObject  # type: ignore[attr-defined]

from ._compatibility import compatibility
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph
from .node import Argument, base_types, map_aggregate
from .proxy import Proxy, ParameterProxy, IvyProxy, TracerBase, Scope, ScopeContextManager
from .graph_converter import tracer_to_ivy_graph
from .func_wrappers import (
    add_custom_decorator,
    proxies_to_native_arrays,
    native_arrays_to_proxies,
    proxies_to_ivy_arrays,
    ivy_arrays_to_proxies,
    convert_proxies_to_ivy_arrays
)

import graph_compiler.globals as glob
from graph_compiler.wrapping import FUNC_TO_PATH
from graph_compiler.graph import Graph as IvyGraph
from graph_compiler.numpy_proxy import custom_np_classes, custom_np_class_names
import numpy as np
import ivy
from ivy.func_wrapper import FN_DECORATORS
import control_flow_experimental as cfe


HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


class SymTraceError(Exception):
    pass


class GraphConvertError(Exception):
    pass


class ASTTransformationError(Exception):
    pass


# These need to run in global scope to handle nested calls correctly
_orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_getattr: Callable = torch.nn.Module.__getattr__

_proxyable_classes: Dict[Type, None] = {}

_is_fx_tracing_flag = False


"""TODO (yusha): optimize the wrapping logic if possible. Maybe we can use the Patcher class
to do the wrapping. Though we'll need to handle 2 cases: 
- handling numpy ufuncs.
- 2 seperate wrappers (tracing wrapper + removing/restoring decorators)"""

# ----------------
# Wrapping Logic
# ----------------

SPECIAL_OPS = ["if_else", "if_exp", "while_loop", "for_loop"]


def _wrap_numpy_ufuncs(wrapped, original):
    """NumPy ufuncs (eg np.add) aren't functions, but instances of a class.
    Hence functools.wraps won't properly handle copying over the attributes to
    the wrapped function. This function does that manually.
    Also some attributes (eg np.add.reduce) could also be in the graph, so we
    wrap these methods before copying them over.
    """
    if isinstance(original, np.ufunc):
        wrapped.nin = original.nin
        wrapped.nout = original.nout
        wrapped.nargs = original.nargs
        wrapped.ntypes = original.ntypes
        wrapped.types = original.types
        wrapped.ntypes = original.ntypes
        wrapped.signature = original.signature
        wrapped.identity = original.identity
        wrapped.reduce = _dummy_tracing_func(original.reduce)
        wrapped.accumulate = _dummy_tracing_func(original.accumulate)
        wrapped.reduceat = _dummy_tracing_func(original.reduceat)
        wrapped.outer = _dummy_tracing_func(original.outer)
        wrapped.at = _dummy_tracing_func(original.at)

        FUNC_TO_PATH[original.reduce] = "numpy." + original.__name__ + ".reduce"
        FUNC_TO_PATH[original.accumulate] = "numpy." + original.__name__ + ".accumulate"
        FUNC_TO_PATH[original.reduceat] = "numpy." + original.__name__ + ".reduceat"
        FUNC_TO_PATH[original.outer] = "numpy." + original.__name__ + ".outer"
        FUNC_TO_PATH[original.at] = "numpy." + original.__name__ + ".at"


def _unwrap_numpy_ufuncs(wrapped, original):
    """Since called attributes of NumPy ufuncs aren't exposed through the normal paths,
    we need to look inside the attributes of wrapped functions
    during unwrapping to find and unwrap these.
    """
    if isinstance(original, np.ufunc):
        wrapped.reduce = _unwrap_function_from_dummy_tracing(wrapped.reduce)
        wrapped.accumulate = _unwrap_function_from_dummy_tracing(wrapped.accumulate)
        wrapped.reduceat = _unwrap_function_from_dummy_tracing(wrapped.reduceat)
        wrapped.outer = _unwrap_function_from_dummy_tracing(wrapped.outer)
        wrapped.at = _unwrap_function_from_dummy_tracing(wrapped.at)


def _unwrap_function_from_dummy_tracing(function_wrapped):
    if hasattr(function_wrapped, "wrapped_for_tracing"):
        _unwrap_numpy_ufuncs(function_wrapped, function_wrapped.__wrapped__)
        return function_wrapped.__wrapped__
    return function_wrapped


def _should_be_wrapped(obj):
    return (
        callable(obj)
        and not inspect.isclass(obj)
        and not (hasattr(obj, "__module__") and obj.__module__ == "typing")
    )

def _not_to_trace(orig_fn, *args, **kwargs):
    # attributes to ignore
    att_name = None
    if orig_fn.__name__ in ["__getattr__", "__setattr__", "__getattribute__"]:
        att_name = args[0]
        # return if the attribute being retrieved is another built-in method
        if att_name[0:2] == "__":
            return True
        # if the attribute is not recognized as one which can form part of the graph, then return
        if (
            att_name
            not in glob.GRAPH_ATTRIBUTES[ivy.current_backend_str()] 
        ):
            True
    
    return False

def _wrap_or_unwrap_module(
    wrap_or_unwrap_fn,
    module,
    framework=None,
    to_ivy=False,
    wrap_or_unwrap_all=False,
):
    framework = ivy.current_backend_str() if framework is None else framework
    framework = "ivy" if to_ivy else framework
    module_name = module.__name__ if not to_ivy else "ivy"
    for k in dir(module):
        v = getattr(module, k)
        if (
            not wrap_or_unwrap_all
            and k in (glob.FUNCTIONS_ATTRS_NOT_TO_WRAP[framework] + SPECIAL_OPS)
        ) or not _should_be_wrapped(  # or k[0] == "_"
            v
        ):
            continue
        try:
            setattr(module, k, wrap_or_unwrap_fn(v))
            if not hasattr(v, "wrapped_for_tracing"):
                FUNC_TO_PATH[v] = module_name + "." + k
        except Exception:
            pass


def _wrap_or_unwrap_class(
    wrap_or_unwrap_fn, cls, cls_path=None, framework=None, to_ivy=False
):
    if cls is None:
        return
    framework = ivy.current_backend_str() if framework is None else framework
    framework = "ivy" if to_ivy else framework
    for k in dir(cls):
        attr = getattr(cls, k)
        if k in glob.FUNCTIONS_ATTRS_NOT_TO_WRAP[framework] or not _should_be_wrapped(
            attr
        ):
            continue
        if ivy.current_backend_str() == "jax":
            import jaxlib

            if hasattr(jaxlib.xla_extension, "ArrayImpl"):
                if attr == jaxlib.xla_extension.ArrayImpl.__init__:
                    continue
        try:
            if hasattr(getattr(cls, k), "__name__"):
                if getattr(cls, k).__name__ != "":
                    setattr(cls, k, wrap_or_unwrap_fn(attr))
        except Exception as e:
            pass
        if cls_path is not None:
            if cls_path == "NewNDArray":
                FUNC_TO_PATH[attr] = "numpy.ndarray." + k
            elif cls_path in custom_np_class_names:
                FUNC_TO_PATH[attr] = k
            else:
                FUNC_TO_PATH[attr] = ".".join(cls_path) + "." + k

def _load_classes_from(ctw: List):
    classes = []
    for _ctw in ctw:
        try:
            classes.append(getattr(importlib.import_module(_ctw[0]), _ctw[1]))
        except AttributeError:
            classes.append(None)
    return classes

def _load_modules_from(mtw: List, add_path=None):
    modules = []
    for _mtw in mtw:
        try:
            if add_path == "frontend":
                path = "ivy.functional.frontends."
            elif add_path == "backend":
                path = "ivy.functional.backends."
            elif add_path == "ivy":
                path = "ivy.functional."
            else:
                path = ""
            _mtw = path + _mtw
            modules.append(importlib.import_module(_mtw))
        except:
            pass
    return modules


def _wrap_functions_for_dummy_tracing(
    stateful_classes=[],
    to_ivy=False,
    with_numpy=False,
    frontend=None,
    _wrap_fn=None,
    path=None,
):
    glob.wrapped_fns = {}
    target = "ivy" if to_ivy else frontend if frontend else ivy.current_backend_str()
    wrap_all = True if _wrap_fn.__name__ == "remove_decorators" else False
    private_class_paths = glob.PRIVATE_CLASSES_TO_WRAP(target)
    private_classes = _load_classes_from(private_class_paths)
    for cls, path in zip(private_classes, private_class_paths):
        _wrap_or_unwrap_class(
            _wrap_fn,
            cls,
            path,
            to_ivy=to_ivy,
        )
    class_paths = glob.CLASSES_TO_WRAP[target]
    classes = _load_classes_from(class_paths)
    for cls, path in zip(classes, class_paths):
        _wrap_or_unwrap_class(
            _wrap_fn,
            cls,
            path,
            to_ivy=to_ivy,
        )
    if target == "tensorflow":
        import tensorflow as tf

        # these tf modules can't be imported from a string, so adding them manually
        modules_to_wrap = [
            tf.compat.v2.compat.v1.nn,
            tf.compat.v2.compat.v1.linalg,
            tf.compat.v2.compat.v1.math,
        ]
    elif target == "ivy":
        modules_to_wrap = [ivy.linalg]
    else:
        modules_to_wrap = []
    modules_to_wrap += _load_modules_from(glob.MODULES_TO_WRAP[target], add_path=path)
    for module in modules_to_wrap:
        _wrap_or_unwrap_module(
            _wrap_fn,
            module,
            to_ivy=to_ivy,
            wrap_or_unwrap_all=wrap_all,
        )

    # wrap numpy after wrapping modules of current backend. wrapping before causes
    # issues with modules like jax.scipy.optimise where they import like
    # `from numpy import asarray` which would then import the wrapped version of
    # numpy.asarray, and would not be unwrapped afterwards. this is only a problem
    # with modules in jax.scipy because they are not initialised upon `import jax`,
    # and so will be initialised when we import them to wrap.
    if with_numpy:
        for custom_class in custom_np_classes:
            _wrap_or_unwrap_class(
                _wrap_fn,
                custom_class,
                custom_class.__name__,
                framework="numpy",
                to_ivy=to_ivy,
            )
        for module in _load_modules_from(glob.MODULES_TO_WRAP["numpy"], add_path=path):
            _wrap_or_unwrap_module(
                _wrap_fn,
                module,
                framework="numpy",
                to_ivy=to_ivy,
                wrap_or_unwrap_all=wrap_all,
            )
    """TODO (yusha): maybe only wrap ivy.vmap??"""
    # wrap functorch.vmap
    if target == "torch":
        try:
            import functorch

            functorch.vmap = _dummy_tracing_func(
                functorch.vmap,
            )
        except:
            # do not wrap functorch.vmap if it is not installed,
            # which can occur when using torch versions < 1.13.0
            pass

    # wrap stateful classes
    stateful_classes = ivy.default(stateful_classes, [])
    for cls in stateful_classes:
        assert hasattr(cls, "__setattr__") and (
            hasattr(cls, "__getattr__") or hasattr(cls, "__getattribute__")
        )
        assert hasattr(cls, "__init__")
        cls.__init__ = _dummy_tracing_func(
            cls.__init__,
        )
        cls.__setattr__ = _dummy_tracing_func(
            cls.__setattr__,
        )
        if hasattr(cls, "__getattr__"):
            cls.__getattr__ = _dummy_tracing_func(
                cls.__getattr__,
            )
        if hasattr(cls, "__getattribute__"):
            cls.__getattribute__ = _dummy_tracing_func(
                cls.__getattribute__,
            )


def _unwrap_functions_from_dummy_tracing(
    stateful_classes=[],
    to_ivy=False,
    with_numpy=False,
    frontend=None,
    _unwrap_fn=None,
    path=None,
):
    wrapped_dict = glob.wrapped_fns
    glob.wrapped_fns = {}
    unwrap_all = True if _unwrap_fn.__name__ == "restore_decorators" else False
    for _, v in wrapped_dict.items():
        if hasattr(v[1], "wrapped_for_tracing"):
            glob.wrapped_fns[id(v[1])] = (v[1], v[0])
    wrapped_dict = {}

    target = "ivy" if to_ivy else frontend if frontend else ivy.current_backend_str()
    if with_numpy:
        for custom_class in custom_np_classes:
            _wrap_or_unwrap_class(
                _unwrap_fn,
                custom_class,
                framework="numpy",
            )
        for module in _load_modules_from(glob.MODULES_TO_WRAP["numpy"], add_path=path):
            _wrap_or_unwrap_module(
                _unwrap_fn,
                module,
                framework="numpy",
                wrap_or_unwrap_all=unwrap_all,
            )

    modules_to_unwrap = _load_modules_from(glob.MODULES_TO_WRAP[target], add_path=path)
    if target == "tensorflow":
        import tensorflow as tf

        modules_to_unwrap += [
            tf.compat.v2.compat.v1.nn,
            tf.compat.v2.compat.v1.linalg,
            tf.compat.v2.compat.v1.math,
        ]
    elif target == "ivy":
        modules_to_unwrap += [ivy.linalg]
    # for module in modules_to_unwrap:
    #     _wrap_or_unwrap_class(
    #         _unwrap_fn, module, to_ivy=to_ivy, wrap_or_unwrap_all=unwrap_all
    #     )
    for module in modules_to_unwrap:
        _wrap_or_unwrap_module(
            _unwrap_fn,
            module,
            to_ivy=to_ivy,
            wrap_or_unwrap_all=unwrap_all,
        )
    # unwrap backend classes
    ctu = glob.CLASSES_TO_WRAP[target]
    classes_to_unwrap = _load_classes_from(ctu) + stateful_classes
    for cls in classes_to_unwrap:
        _wrap_or_unwrap_class(
            _unwrap_fn,
            cls,
            to_ivy=to_ivy,
        )

    # unwrap private classes
    pctw = glob.PRIVATE_CLASSES_TO_WRAP(target)[::-1]
    priv_classes_to_wrap = _load_classes_from(pctw)
    for pctw in priv_classes_to_wrap:
        _wrap_or_unwrap_class(
            _unwrap_fn,
            pctw,
            to_ivy=to_ivy,
        )

    # unwrap functorch.vmap
    if target == "torch":
        try:
            import functorch

            functorch.vmap = _unwrap_function_from_dummy_tracing(
                functorch.vmap,
            )
        except:
            pass

    # unwrap stateful classes
    stateful_classes = ivy.default(stateful_classes, [])
    for cls in stateful_classes:
        assert hasattr(cls, "__init__")
        cls.__init__ = _unwrap_function_from_dummy_tracing(cls.__init__)
        assert hasattr(cls, "__setattr__") and (
            hasattr(cls, "__getattr__") or hasattr(cls, "__getattribute__")
        )
        cls.__setattr__ = _unwrap_function_from_dummy_tracing(cls.__setattr__)
        if hasattr(cls, "__getattr__"):
            cls.__getattr__ = _unwrap_function_from_dummy_tracing(cls.__getattr__)
        if hasattr(cls, "__getattribute__"):
            cls.__getattribute__ = _unwrap_function_from_dummy_tracing(
                cls.__getattribute__
            )


# -------------------------
# Symbolic Tracing Logic
# -------------------------
def is_fx_tracing():
    return _is_fx_tracing_flag


@compatibility(is_backward_compatible=True)
class ProxyableClassMeta(type):
    """
    ProxyableClassMeta allows you to make construction of a given Python class
    symbolically traceable. For example::

        import torch
        import torch.fx

        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):
            def __init__(self, left, right):
                self.left, self.right = left, right

            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        def use_tensor_pair_ctor(x : TensorPair, y : torch.Tensor):
            s = x.add(TensorPair(y, y))
            return s.mul(x)

        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = torch.randn(5, 3)
        ref_out = use_tensor_pair_ctor(x, y)

        traced = torch.fx.symbolic_trace(use_tensor_pair_ctor)
        print(traced.code)
        '''
        def forward(self, x : __main___TensorPair, y : torch.Tensor):
            tensor_pair = __main___TensorPair(y, y);  y = None
            add = x.add(tensor_pair);  tensor_pair = None
            mul = add.mul(x);  add = x = None
            return mul
        '''

    From this example, we can see that construction of a class (``TensorPair``)
    defined with ``ProxyableClassMeta`` as metaclass can be recorded in symbolic
    tracing.
    """

    def __init__(cls, name, bases, attrs):
        _proxyable_classes.setdefault(cls)
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls)  # type: ignore[call-overload]

        found_proxies = []

        def check_proxy(a):
            if isinstance(a, Proxy):
                found_proxies.append(a)

        map_aggregate(args, check_proxy)
        map_aggregate(kwargs, check_proxy)

        if len(found_proxies) != 0:
            tracer = found_proxies[0].tracer
            return tracer.create_proxy("call_function", cls, args, kwargs)
        else:
            cls.__init__(instance, *args, **kwargs)  # type: ignore[misc]
            return instance


def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    co_args: tuple
    if hasattr(co, "co_qualname"):
        # Python-3.11+ code signature
        co_args = (
            nargs,
            0,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_qualname,  # type: ignore[attr-defined]
            co.co_firstlineno,
            co.co_lnotab,
            co.co_exceptiontable,  # type: ignore[attr-defined]
            co.co_freevars,
            co.co_cellvars,
        )
    elif hasattr(co, "co_posonlyargcount"):
        co_args = (
            nargs,
            0,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_freevars,
            co.co_cellvars,
        )
    else:
        co_args = (
            nargs,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_freevars,
            co.co_cellvars,
        )
    new_code = CodeType(*co_args)  # type: ignore[arg-type]
    return FunctionType(
        new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )

    # we need to insert placeholder nodes for *args and **kwargs
    # we can't call this function normally, otherwise it would try to unpack them
    # instead, let's make python think that args and kwargs are normal variables


@compatibility(is_backward_compatible=False)
class PHBase:
    """
    Object representing an input placeholder to `constant_args`
    """

    def __repr__(self):
        return "PH"


PH = PHBase()


@compatibility(is_backward_compatible=False)
class PHWithMeta(PHBase):
    """
    Object representing an input placeholder to `constant_args`
    """

    def __init__(self, ph_key: Optional[str] = None):
        super().__init__()

        # Provide a hey for user to identify placeholder node during analysis
        self.ph_key = ph_key


@compatibility(is_backward_compatible=True)
class Tracer(TracerBase):
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `math`s path from the
    # build environment (e.g. `<module 'math' from '/leaked/path').

    """Tracer(autowrap_modules=(math,), autowrap_functions=())

    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.

    Tracer can be subclassed to override various behaviors of the tracing
    process. The different behaviors that can be overridden are described
    in the docstrings of the methods on this class.
    """

    # Not checking BC on this API because the default value for `autowrap_modules`
    # includes the local filepath to the `math` module, which would jitter
    # across machines.
    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        autowrap_modules: Tuple[ModuleType] = (math,),
        autowrap_functions: Tuple[Callable, ...] = (),
        param_shapes_constant: bool = False,
        name="main",
    ) -> None:
        # This method's signature is overridden by the first line of this class'
        # docstring. If this method's signature is modified, the signature that
        # overrides it also should be modified accordingly.

        """
        Construct a Tracer object.

        Args:

            autowrap_modules (Tuple[ModuleType]): defaults to `(math, )`,
                Python modules whose functions should be wrapped automatically
                without needing to use fx.wrap(). Backward-compatibility for
                this parameter is guaranteed.

            autowrap_functions (Tuple[Callable, ...]): defaults to `()`,
                Python functions that should be wrapped automatically without
                needing to use fx.wrap(). Backward compatibility for this
                parameter is guaranteed.

            param_shapes_constant (bool): When this flag is set,  calls to shape,
                size and a few other shape like attributes of a module's parameter
                will be evaluated directly, rather than returning a new Proxy value
                for an attribute access. Backward compatibility for this parameter
                is guaranteed.
        """

        super().__init__()

        # Functions we will eagerly wrap when we see them while tracing
        # this captures both `math.sqrt()` and `from math import sqrt` automatically
        self._autowrap_function_ids: Set[int] = {
            id(value)
            for name, value in chain(*[m.__dict__.items() for m in autowrap_modules])
            if not name.startswith("_") and callable(value)
        }
        self._autowrap_function_ids.update({id(f) for f in autowrap_functions})

        # Python modules to apply autowrap to at the start, in addition to
        # modules we see while tracing
        self._autowrap_search: List[ModuleType] = list(autowrap_modules)
        self.param_shapes_constant = param_shapes_constant

        # self.submodule_paths: Optional[Dict[torch.nn.Module, str]] = None
        self.root_module_name: str = ""
        # Maps the containing module's name to the operator name
        self.scope = Scope("", None)
        # Records the module call stack
        self.module_stack = collections.OrderedDict()
        # Mapping of node name to module scope
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}

        self.subgraphs = []
        self.name = name

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> "Argument":
        """
        A method to specify the behavior of tracing when preparing values to
        be used as arguments to nodes in the ``Graph``.

        By default, the behavior includes:

        #. Iterate through collection types (e.g. tuple, list, dict) and recursively
           call ``create_args`` on the elements.
        #. Given a Proxy object, return a reference to the underlying IR ``Node``
        #. Given a non-Proxy Tensor object, emit IR for various cases:

            * For a Parameter, emit a ``get_attr`` node referring to that Parameter
            * For a non-Parameter Tensor, store the Tensor away in a special
              attribute referring to that attribute.

        This method can be overridden to support more types.

        Args:

            a (Any): The value to be emitted as an ``Argument`` in the ``Graph``.


        Returns:

            The value ``a`` converted into the appropriate ``Argument``
        """
        # The base tracer is used to construct Graphs when there is no associated
        # module hierarchy, so it can never create parameter references.
        # The default tracer adds the ability to refer to parameters when
        # tracing modules.
        # if not isinstance(a, IvyProxy) and (
        #     ivy.is_array(a) or ivy.is_ivy_container(a) #or ivy.is_native_dtype(a)
        # ):
        #     return Constant(a)
        """TODO (yusha): remove this piece of code if its unneeded"""

        # if isinstance(a, torch.nn.Parameter):
        #     for n, p in self.root.named_parameters():
        #         if a is p:
        #             return self.create_node("get_attr", n, (), {})
        #     raise NameError("parameter is not a member of this module")
        # elif isinstance(a, torch.Tensor):
        #     for n_, p_ in self.root.named_buffers():
        #         if a is p_:
        #             return self.create_node("get_attr", n_, (), {})
        # elif isinstance(a, torch.nn.Module):
        #     for n_, p_ in self.root.named_modules():
        #         if a is p_:
        #             return self.create_node("get_attr", n_, (), {})
        # For NamedTuple instances that appear literally as args, we emit
        # a node to construct the NamedTuple and use that Node as the argument.
        if isinstance(a, tuple) and hasattr(a, "_fields"):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node("call_function", a.__class__, args, {})

        # Tensors do not have a reliable string repr() from which they can be
        # constructed (and we probably don't want to rely on that, either), so
        # for any constant Tensor values we encounter, first search for if they
        # are an attribute of some module in the module hierarchy. If so, emit
        # a get_attr to retrieve that tensor. Otherwise, we'll store away the
        # tensor value into a special attribute on the Module s.t. we can
        # retrieve it with a get_attr.
        # if isinstance(a, (torch.Tensor, ScriptObject)):
        #     qualname: Optional[str] = self.tensor_attrs.get(a)

        #     # Tensor was not found in the Module hierarchy, stow it away in a
        #     # special attribute and set the qualname to refer to that
        #     if not qualname:
        #         i = 0
        #         while True:
        #             qualname = f"_tensor_constant{i}"
        #             if not hasattr(self.root, qualname):
        #                 break
        #             i += 1
        #         self.tensor_attrs[a] = qualname
        #         setattr(self.root, qualname, a)

        #     return self.create_node("get_attr", qualname, (), {})

        if type(a) in _proxyable_classes:
            # This is an instance of a proxyable class for which we did not
            # witness its construction. Intern this as a constant attribute

            # TODO: binary search
            i = 0
            while True:
                qualname = f"_{a.__class__.__name__}_constant_{i}"
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})

        return super().create_arg(a)

    """TODO (yusha): Unify this logic to work with ivy.Module instead of torch.nn.Module"""

    @compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:

            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        return False
        return (
            m.__module__.startswith("torch.nn")
            or m.__module__.startswith("torch.ao.nn")
        ) and not isinstance(m, torch.nn.Sequential)

    @compatibility(is_backward_compatible=True)
    def path_of_module(self, mod: torch.nn.Module) -> str:
        """
        Helper method to find the qualified name of ``mod`` in the Module hierarchy
        of ``root``. For example, if ``root`` has a submodule named ``foo``, which has
        a submodule named ``bar``, passing ``bar`` into this function will return
        the string "foo.bar".

        Args:

            mod (str): The ``Module`` to retrieve the qualified name for.
        """
        # Prefer the O(1) algorithm
        if self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                raise NameError("module is not installed as a submodule")
            assert isinstance(path, str)
            return path
        # O(N^2) fallback in the case that we didn't store the submodule
        # paths.
        else:
            for n, p in self.root.named_modules():
                if mod is p:
                    return n
            raise NameError("module is not installed as a submodule")

    @compatibility(is_backward_compatible=True)
    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when it encounters
        a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf module
        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to
        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through
        the operations in its ``forward`` function.

        This method can be overridden to--for example--create nested traced
        GraphModules, or any other behavior you would want while tracing across
        ``Module`` boundaries.

        Args:

            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Return:

            The return value from the Module call. In the case that a ``call_module``
            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever
            value was returned from the ``Module`` invocation.
        """
        module_qualified_name = self.path_of_module(m)
        with ScopeContextManager(
            self.scope, Scope(module_qualified_name, type(m))
        ) as _scope:
            # module_stack is an ordered dict so writing then deleting the
            # entry is equivalent to push/pop on a list
            self.module_stack[_scope.module_path] = _scope.module_type
            if not self.is_leaf_module(m, module_qualified_name):
                ret_val = forward(*args, **kwargs)
            else:
                ret_val = self.create_proxy(
                    "call_module", module_qualified_name, args, kwargs
                )
            key, _ = self.module_stack.popitem(last=True)
            assert key == _scope.module_path, f" Unexpected key {key}"

        return ret_val

    @compatibility(is_backward_compatible=False)
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        """
        Method that specifies the behavior of this ``Tracer`` when we call getattr
        on a call to an ``nn.Module`` instance.

        By default, the behavior is to return a proxy value for the attribute. It
        also stores the proxy value in the ``parameter_proxy_cache``, so that future
        calls will reuse the proxy rather than creating a new one.

        This method can be overridden to --for example-- not return proxies when
        querying parameters.

        Args:

            attr (str): The name of the attribute being queried
            attr_val (Any): The value of the attribute
            parameter_proxy_cache (Dict[str, Any]): A cache of attr names to proxies

        Return:

            The return value from the getattr call.
        """
        """TODO (yusha): remove this piece of code as well if its unneeded"""
        def maybe_get_proxy_for_attr(
            attr_val, collection_to_search, parameter_proxy_cache
        ):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if (
                            "proxy_factory_fn"
                            in inspect.signature(self.create_proxy).parameters
                        ):
                            kwargs["proxy_factory_fn"] = (
                                None
                                if not self.param_shapes_constant
                                else lambda node: ParameterProxy(
                                    self, node, n, attr_val
                                )
                            )
                        val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_parameters(), parameter_proxy_cache
            )
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_buffers(), parameter_proxy_cache
            )
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        return attr_val

    # This method will be refactored
    @compatibility(is_backward_compatible=False)
    def create_args_for_root(
        self,
        root_fn,
        is_module,
        constant_args=None,
        frontend=None,
        *concr_args,
        **concr_kwargs,
    ):
        """
        Create ``placeholder`` nodes corresponding to the signature of the ``root``
        Module. This method introspects root's signature and emits those
        nodes accordingly, also supporting ``*args`` and ``**kwargs``.
        """
        # In some cases, a function or method has been decorated with a wrapper
        # defined via ``functools.wraps``. In this case, the outer code object
        # will likely not contain the actual parameters we care about, so unwrap
        # the function to get to the innermost callable.
        fn_for_analysis = inspect.unwrap(root_fn)
        co = fn_for_analysis.__code__
        total_args = co.co_argcount + co.co_kwonlyargcount
        orig_args = list(co.co_varnames)
        names_iter = iter(co.co_varnames)
        args: List[Any] = []
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError(
                    "``self`` argument cannot be part of *args expansion!"
                )
            skip_arg_idx = 1
            next(names_iter)  # skip self
            args.append(self.root)

        sig = inspect.signature(fn_for_analysis)

        def proxy_placeholder(name: str):
            if constant_args is not None and name in constant_args:
                cnt = 0

                def replace_ph(x):
                    nonlocal cnt
                    cnt += 1
                    param = sig.parameters[name]
                    default = (
                        ()
                        if param.default is inspect.Parameter.empty
                        else (param.default,)
                    )
                    out = self.create_proxy(
                        "placeholder", f"{name}_{str(cnt)}", default, {}
                    )
                    if isinstance(x, PHBase):

                        def transfer_attrs(fr, to):
                            for attr_name in dir(fr):
                                attr_val = getattr(fr, attr_name)
                                if (
                                    not callable(attr_val)
                                    and not attr_name.startswith("__")
                                    and not hasattr(to, attr_name)
                                ):
                                    setattr(to, attr_name, attr_val)

                        if x != PH:
                            # Transfer attrs in the case where you're using a placeholder other
                            # than the singleton PH (PH has no attributes to transfer).
                            # Proxies were created out of the placeholders.
                            # Transfer any metadata (put on the placeholders in the form of
                            # attributes set by the user) from the placeholder to the
                            # underlying nodes (the proxy is unwrapped by the user, but
                            # the metadata should hold).
                            transfer_attrs(fr=x, to=out.node)

                        return out
                    # Union[int, bool] == bool in Python <= 3.6
                    if type(x) == bool or type(x) in base_types:
                        pass
                    elif type(x) == type(None):
                        pass
                    else:
                        warnings.warn(
                            f"Was not able to add assertion to guarantee correct input {name} to "
                            f"specialized function. It is up to the user to make sure that your inputs match the "
                            f"inputs you specialized the function with."
                        )

                    return x

                return pytree.tree_map(replace_ph, constant_args[name])

            if concrete_inputs is not None and name in concrete_inputs:
                concr_arg = concrete_inputs[name]
                proxy_data = concr_arg
            else:
                proxy_data = None
            # if name[0] == "*":
            #     default = ()
            # else:
            #     param = sig.parameters[name]
            #     default = () if param.default is inspect.Parameter.empty else (param.default,)  # type: ignore[assignment]
            return self.create_proxy(
                "placeholder",
                name,
                (),
                {},
                type_expr=fn_for_analysis.__annotations__.get(name, None),
                data=proxy_data,
                frontend=frontend,
            )

        arg_names = [next(names_iter) for idx in range(skip_arg_idx, total_args)]
        if isinstance(constant_args, tuple):
            if len(arg_names) != len(constant_args):
                raise RuntimeError(
                    f"Tracing expected {len(arg_names)} arguments but got {len(constant_args)} concrete arguments"
                )
            constant_args = dict(zip(arg_names, constant_args))
        concrete_inputs = {}
        concr_args_iter = iter(concr_args)
        for name in arg_names:
            if name in concr_kwargs:
                concrete_inputs[name] = concr_kwargs[name]
            else:
                try:
                    concrete_inputs[name] = next(concr_args_iter)
                except StopIteration:
                    pass

        args.extend(proxy_placeholder(names) for names in arg_names)

        if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
            varargs_proxies = []
            kwargs_proxies = {}
            # Create a new concrete_inputs dictionary that maps the generated names
            # for the concr_args and concr_kwargs values to their corresponding values
            concrete_inputs = {}
            for i, value in enumerate(concr_args):
                name = f"args{i}"
                concrete_inputs[name] = value
            for name, value in concr_kwargs.items():
                concrete_inputs[name] = value

            if co.co_flags & inspect.CO_VARARGS:
                varargs_proxies.extend(
                    proxy_placeholder(f"args{i}") for i, _ in enumerate(concr_args)
                )

            if co.co_flags & inspect.CO_VARKEYWORDS:
                kwargs_proxies.update(
                    {
                        k: root_fn.constants[k]
                        if k in root_fn.constants
                        else proxy_placeholder(k)
                        for k, v in concr_kwargs.items()
                    }
                )

            # root_fn = _patch_function(root_fn, len(args) + len(varargs_proxies) + len(kwargs_proxies))

            # Create a wrapper function that takes a single list of arguments and unpacks it
            # into separate arguments for the variable-length and keyword arguments
            def var_arg_n_kwargs_fn(*all_args):
                varargs = all_args[: len(varargs_proxies) + len(args)]
                kwargs = {
                    k: v
                    for k, v in zip(
                        kwargs_proxies.keys(), all_args[len(varargs_proxies) :]
                    )
                }
                return root_fn(*varargs, **kwargs)

            return var_arg_n_kwargs_fn, args + varargs_proxies + list(
                kwargs_proxies.values()
            )

        flat_args, in_spec = pytree.tree_flatten(tuple(args))
        if any(not isinstance(i, pytree.LeafSpec) for i in in_spec.children_specs):
            # In the case that we have pytree-flattened inputs in
            # `constant_args`, generate a flattening wrapper around the
            # original root function and return that.
            self.graph._codegen = _PyTreeCodeGen(
                _PyTreeInfo(orig_args[:total_args], in_spec, None)
            )

            def flatten_fn(*args):
                tree_args = pytree.tree_unflatten(list(args), in_spec)
                tree_out = root_fn(*tree_args)
                out_args, out_spec = pytree.tree_flatten(tree_out)
                assert isinstance(self.graph._codegen, _PyTreeCodeGen)
                self.graph._codegen.pytree_info = (
                    self.graph._codegen.pytree_info._replace(out_spec=out_spec)
                )
                return out_args

            return flatten_fn, flat_args
        return root_fn, args

    """TODO (yusha): Modify the logic here as well to work with ivy.Module instead of torch.nn.Module"""

    @compatibility(is_backward_compatible=True)
    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        constant_args: Optional[Dict[str, Any]] = None,
        args=[],
        frontend=None,
        **kwargs,
    ) -> Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``torch.nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``ivy.Module`` instance to use as the root and add embedded constants
        to.


        Args:

            root (Union[Module, Callable]): Either a ``Module`` or a function to be
                traced through. Backwards-compatibility for this parameter is
                guaranteed.
            constant_args (Optional[Dict[str, any]]): Concrete arguments that should
                not be treated as Proxies. This parameter is experimental and
                its backwards-compatibility is *NOT* guaranteed.

        Returns:

            A ``Graph`` representing the semantics of the passed-in ``root``.
        """
        global _is_fx_tracing_flag
        old_is_fx_tracing_flag = _is_fx_tracing_flag
        _is_fx_tracing_flag = True
        try:
            if isinstance(root, torch.nn.Module):
                self.root = root

                assert hasattr(
                    type(root), self.traced_func_name
                ), f"traced_func_name={self.traced_func_name} doesn't exist in {type(root).__name__}"

                fn = getattr(type(root), self.traced_func_name)
                self.root_module_name = root._get_name()
                self.submodule_paths = {mod: name for name, mod in root.named_modules()}
            else:
                self.root = torch.nn.Module()
                fn = root

            tracer_cls: Optional[Type["Tracer"]] = getattr(self, "__class__", None)
            self.graph = Graph(tracer_cls=tracer_cls)

            # When we encounter a Tensor value that's not a parameter, we look if it
            # is some other attribute on the model. Construct a dict mapping Tensor
            # values to the qualified name here for efficiency. This is used downstream
            # in create_arg
            # self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

            # def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
            #     for k, v in m.__dict__.items():
            #         if isinstance(v, (torch.Tensor, ScriptObject)):
            #             self.tensor_attrs[v] = ".".join(prefix_atoms + [k])
            #     for k, v in m.named_children():
            #         collect_tensor_attrs(v, prefix_atoms + [k])

            # collect_tensor_attrs(self.root, [])

            assert isinstance(fn, FunctionType)

            fn_globals = fn.__globals__  # run before it gets patched
            fn, args = self.create_args_for_root(
                fn,
                isinstance(root, torch.nn.Module),
                constant_args,
                frontend,
                *args,
                **kwargs,
            )

            parameter_proxy_cache: Dict[
                str, Proxy
            ] = {}  # Reduce number of get_attr calls

            # Method dispatch on parameters is not recorded unless it's directly used.
            # Thus, we need to insert a proxy when __getattr__ requests a parameter.
            @functools.wraps(_orig_module_getattr)
            def module_getattr_wrapper(mod, attr):
                attr_val = _orig_module_getattr(mod, attr)
                return self.getattr(attr, attr_val, parameter_proxy_cache)

            @functools.wraps(_orig_module_call)
            def module_call_wrapper(mod, *args, **kwargs):
                def forward(*args, **kwargs):
                    return _orig_module_call(mod, *args, **kwargs)

                _autowrap_check(
                    patcher,
                    getattr(getattr(mod, "forward", mod), "__globals__", {}),
                    self._autowrap_function_ids,
                )
                return self.call_module(mod, forward, args, kwargs)

            with _Patcher() as patcher:
                # allow duplicate patches to support the case of nested calls
                patcher.patch_method(
                    torch.nn.Module,
                    "__getattr__",
                    module_getattr_wrapper,
                    deduplicate=False,
                )
                patcher.patch_method(
                    torch.nn.Module, "__call__", module_call_wrapper, deduplicate=False
                )
                _patch_wrapped_functions(patcher)
                _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
                for module in self._autowrap_search:
                    _autowrap_check(
                        patcher, module.__dict__, self._autowrap_function_ids
                    )
                orig_ret = fn(*args)
                self.orig_ret = orig_ret
                self.create_node(
                    "output",
                    "output",
                    (self.create_arg(orig_ret),),
                    {},
                    type_expr=fn.__annotations__.get("return", None),
                )

            self.submodule_paths = None
        finally:
            _is_fx_tracing_flag = old_is_fx_tracing_flag
        return self.graph

    def __deepcopy__(self, memo):
        # _autowrap_search contains modules, which cannot be deepcopied.
        new_tracer = Tracer.__new__(Tracer)

        for k, v in self.__dict__.items():
            if k in {"_autowrap_search"}:
                new_obj = copy.copy(v)
            else:
                new_obj = copy.deepcopy(v, memo)

            new_tracer.__dict__[k] = new_obj

        return new_tracer


# List of pairs of (global dict, function name) functions
# to patch for the purposes of the wrap() API.
_wrapped_fns_to_patch: List[Tuple[dict, str]] = []

# List of methods on classes to wrap (class type, function name)
# this currently only works for Tensor.* methods that aren't traced properly
_wrapped_methods_to_patch: List[Tuple[type, str]] = []


def _find_proxy(*objects_to_search):
    """
    Recursively search a data structure for a Proxy() and return it,
    return None if not found.
    """
    proxy = None

    def find_proxy(x):
        nonlocal proxy
        if isinstance(x, Proxy):
            proxy = x

    map_aggregate(objects_to_search, find_proxy)
    return proxy


def _py_forloop(itr_, body, body_vars):
    for itr in itr_:
        body_vars = body(itr, *body_vars)


def _py_whileloop(test_fn, body_fn, body_vars):
    while test_fn(*body_vars):
        body_vars = body_fn(*body_vars)

    return body_vars


def _dummy_tracing_func(orig_fn):
    """
    Given an closed-over ``orig_function`` to invoke, search the args and kwargs for
    a Proxy object. If there is one, emit a ``call_function`` node to preserve the
    call to this leaf function directly. Otherwise, just return the results of
    this function call, as this function is not being traced.
    """
    is_already_wrapped = hasattr(orig_fn, "wrapped_for_tracing")

    # do not wrap default __init__
    if orig_fn is object.__init__:
        return orig_fn

    # Do not wrap the function:
    # (a) if it's a special method but not in ARRAY_BUILTINS
    # (b) if it's already wrapped
    if (
        (
            hasattr(orig_fn, "__name__")
            and (orig_fn.__name__[0] == "_" and orig_fn.__name__ not in glob.ARRAY_BUILTINS)
        )
        or is_already_wrapped
        
    ):
        return orig_fn
    
    if orig_fn.__name__ in ("if_else", "if_exp"):
        # Handle control flow ops
        @functools.wraps(orig_fn)
        def wrapped(*args, **kwargs):
            # Trace the callables passed as arguments to the control flow op
            ret = None
            subgs = []
            cond_vars = args[3]
            const_args, concr_args, proxy_inputs = {}, {}, []
            for k, v in cond_vars.items():
                if isinstance(v, Proxy):
                    proxy_inputs.append(v)
                    concr_args[k] = v.data
                else:
                    const_args[k] = v

            t = Tracer(name="pred_fn")
            pred_g = t.trace(args[0], constant_args=const_args, args=[], **concr_args)
            pred = t.orig_ret
            try:
                pred = pred[0] if isinstance(pred, (list, tuple)) else pred
            except IndexError:
                pred = False
            if not isinstance(pred, Proxy):  # static control flow
                # defer to pythonic if-else
                return args[1](**cond_vars) if pred else args[2](**cond_vars)
            else:
                # compile both branches since the control flow is dynamic
                subgs = [pred_g]
                for arg in args[1:]:
                    if inspect.isfunction(arg):
                        t = Tracer(name=f"{arg.__name__}")
                        graph = t.trace(
                            arg,
                            constant_args=const_args,
                            args=[],
                            **concr_args,
                        )
                        subgs.append(graph)

            ret = t.orig_ret
            proxy = _find_proxy(args, kwargs)
            if proxy is not None:
                return_proxy = proxy.tracer.create_proxy(
                    "call_function", orig_fn, args, kwargs, data=proxy._meta_tensor
                )
                return_proxy.node.meta["is_wrapped"] = True
                return_proxy.node.meta["orig_ret"] = ret
                return_proxy.node.meta["pred"] = pred
                return_proxy.node.meta["subgraphs"] = subgs
                return return_proxy
            return ret

    elif orig_fn.__name__ == "for_loop":
        """TODO (yusha): add tracing logic for 'for-loops'"""

        @functools.wraps(orig_fn)
        def wrapped(*args, **kwargs):
            pass

    elif orig_fn.__name__ == "while_loop":

        @functools.wraps(orig_fn)
        def wrapped(*args, **kwargs):
            # Trace the callables passed as arguments to the control flow op
            ret = None
            subgs = []
            test_fn, body_fn, loop_vars = args
            const_args, concr_args, proxy_inputs = {}, {}, []
            for k, v in loop_vars.items():
                if isinstance(v, Proxy):
                    proxy_inputs.append(v)
                    concr_args[k] = v.data
                else:
                    const_args[k] = v

            t = Tracer(name="test_fn")
            test_g = t.trace(test_fn, constant_args=const_args, args=[], **concr_args)
            test = t.orig_ret
            try:
                test = test[0] if isinstance(test, (list, tuple)) else test
            except IndexError:
                test = False
            if not isinstance(test, Proxy):  # static control flow
                # defer to pythonic while loop
                return _py_whileloop(test_fn, body_fn, loop_vars.values())
            else:
                # compile both callables since the control flow is dynamic
                subgs = [test_g]
                t = Tracer(name="body_fn")
                graph = t.trace(
                    body_fn, constant_args=const_args, args=[], **concr_args
                )
                subgs.append(graph)

            ret = t.orig_ret
            proxy = _find_proxy(args, kwargs)
            if proxy is not None:
                return_proxy = proxy.tracer.create_proxy(
                    "call_function", orig_fn, args, kwargs, data=proxy._meta_tensor
                )
                return_proxy.node.meta["is_wrapped"] = True
                return_proxy.node.meta["orig_ret"] = ret
                return_proxy.node.meta["test"] = test
                return_proxy.node.meta["subgraphs"] = subgs
                return return_proxy
            return ret

    else:
        @functools.wraps(orig_fn)
        def wrapped(*args, **kwargs):
            
            if _not_to_trace(orig_fn, *args, **kwargs):
                return orig_fn(*args, **kwargs)
            
            if orig_fn.__name__[0:2] == "__" :
                attr = orig_fn.__name__.strip('__')
                if attr in glob.ALL_DUNDER_METHODS:
                    # strip off the self argument
                    args = args[1:] 
            proxy = _find_proxy(args, kwargs)
            # Todo: search for the .data attribute inside _find_proxy
            if proxy is not None:
                return_proxy = proxy.tracer.create_proxy(
                    "call_function", orig_fn, args, kwargs, data=proxy._meta_tensor,  frontend=proxy.frontend
                )
                return_proxy.node.meta["is_wrapped"] = True
                return return_proxy
            return orig_fn(*args, **kwargs)

    wrapped.wrapped_for_tracing = True
    _wrap_numpy_ufuncs(wrapped, orig_fn)
    return wrapped


def _create_wrapped_method(cls, name):
    orig_fn = getattr(cls, name)

    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        """
        Search the args and kwargs for a Proxy object. If there is one,
        emit a ``call_method`` node to preserve the call to this method
        directly. Otherwise, just return the results of this function
        call, as this function is not being traced.
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return proxy.tracer.create_proxy(
                "call_method", name, args, kwargs, data=proxy._meta_tensor
            )
        return orig_fn(*args, **kwargs)

    return wrapped


class _PatchedFn(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any

    def revert(self):
        raise NotImplementedError()


class _PatchedFnSetItem(_PatchedFn):
    def revert(self):
        self.frame_dict[self.fn_name] = self.orig_fn


class _PatchedFnDel(_PatchedFn):
    def revert(self):
        del self.frame_dict[self.fn_name]


class _PatchedFnSetAttr(_PatchedFn):
    def revert(self):
        setattr(self.frame_dict, self.fn_name, self.orig_fn)


class _PatchedFnSetUfuncAttr(_PatchedFn):
    def revert(self):
        self.frame_dict[self.fn_name].func = self.orig_fn


class _Patcher:
    def __init__(self):
        super().__init__()
        self.patches_made: List[_PatchedFn] = []
        self.visited: Set[int] = set()

    def patch(
        self,
        frame_dict: Dict[str, Any],
        name: str,
        new_fn: Callable,
        deduplicate: bool = True,
    ):
        """
        Replace frame_dict[name] with new_fn until we exit the context manager.
        """
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        if name not in frame_dict and hasattr(builtins, name):
            self.patches_made.append(_PatchedFnDel(frame_dict, name, None))
        elif getattr(frame_dict[name], "__fx_already_patched", False):
            return  # already patched, no need to do it again
        else:
            self.patches_made.append(
                _PatchedFnSetItem(frame_dict, name, frame_dict[name])
            )
        frame_dict[name] = new_fn

    def patch_method(
        self, cls: type, name: str, new_fn: Callable, deduplicate: bool = True
    ):
        """
        Replace object_or_dict.name with new_fn until we exit the context manager.
        """
        # new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        orig_fn = getattr(cls, name)
        # if getattr(orig_fn, "__fx_already_patched", False):
        #     return  # already patched, no need to do it again
        if orig_fn.__class__.__name__ == "ufunc":
            self.patches_made.append(_PatchedFnSetUfuncAttr(cls, name, orig_fn))
            cls[name].func = new_fn
        else:
            self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn))
            setattr(cls, name, new_fn)

    def visit_once(self, thing: Any):
        """Return True on the first call to with thing, otherwise false"""
        idx = id(thing)
        if idx in self.visited:
            return False
        self.visited.add(idx)
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Undo all the changes made via self.patch() and self.patch_method()
        """
        while self.patches_made:
            # unpatch in reverse order to handle duplicates correctly
            self.patches_made.pop().revert()
        self.visited.clear()


def _patch_modules(
    patcher: _Patcher,
    modules: List[Dict[str, Any]],
    wrap_fn: Callable,
    wrap_all_fns=False,
    framework=None,
):
    """
    Iterate through the ``modules`` list and, for each object, wrap
    the listed functions using the `wrap_fn` wrapper.
    """
    for module in modules:
        for name in dir(module): 
            orig_func = getattr(module, name)
            if (
                name
                in FN_DECORATORS 
                + ["handle_numpy_arrays_in_specific_backend", "casting_modes_ops"]
                + glob.IVY_FUNCTIONS_NOT_TO_WRAP 
                or not _should_be_wrapped(orig_func)
                or name[0] == "_"
            ):
                continue
            elif name in glob.IVY_FUNCTIONS_NOT_TO_TRACK:
                if orig_func.__class__.__name__ == "ufunc":
                    patcher.patch_method(
                        module, name, convert_proxies_to_ivy_arrays(orig_func.func)
                    )
                else:
                    patcher.patch_method(module, name, convert_proxies_to_ivy_arrays(orig_func))
            else:
                if orig_func.__class__.__name__ == "ufunc":
                    patcher.patch_method(module, name, wrap_fn(orig_func.func))
                else:
                    patcher.patch_method(module, name, wrap_fn(orig_func))


def _patch_wrapped_functions(patcher: _Patcher):
    """
    Go through ``_wrapped_fn_patch_table`` and, for each frame object, wrap
    the listed global functions in the `_dummy_tracing_func` wrapper.
    """
    for frame_dict, name in _wrapped_fns_to_patch:
        if name not in frame_dict and hasattr(builtins, name):
            orig_fn = getattr(builtins, name)
        else:
            orig_fn = frame_dict[name]
        patcher.patch(frame_dict, name, _dummy_tracing_func(orig_fn))

    for cls, name in _wrapped_methods_to_patch:
        patcher.patch_method(cls, name, _create_wrapped_method(cls, name))


def _autowrap_check(
    patcher: _Patcher, frame_dict: Dict[str, Any], function_ids: Set[int]
):
    """
    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.
    This method searches a scope for them and patches them if found.
    """
    if patcher.visit_once(frame_dict):
        for name, value in frame_dict.items():
            if (
                not name.startswith("_")
                and callable(value)
                and id(value) in function_ids
            ):
                patcher.patch(frame_dict, name, _dummy_tracing_func(value))


@compatibility(is_backward_compatible=True)
def wrap(fn_or_name: Union[str, Callable], dynamic: bool = False):
    """
    This function can be called at module-level scope to register fn_or_name as a "leaf function".
    A "leaf function" will be preserved as a CallFunction node in the FX trace instead of being
    traced through::

        # foo/bar/baz.py
        def my_custom_function(x, y):
            return x * x + y * y

        torch.fx.wrap('my_custom_function')

        def fn_to_be_traced(x, y):
            # When symbolic tracing, the below call to my_custom_function will be inserted into
            # the graph rather than tracing it.
            return my_custom_function(x, y)

    This function can also equivalently be used as a decorator::

        # foo/bar/baz.py
        @torch.fx.wrap
        def my_custom_function(x, y):
            return x * x + y * y

    A wrapped function can be thought of a "leaf function", analogous to the concept of
    "leaf modules", that is, they are functions that are left as calls in the FX trace
    rather than traced through.

    Args:

        fn_or_name (Union[str, Callable]): The function or name of the global function to insert into the
            graph when it's called
    """
    if not callable(fn_or_name) and not isinstance(fn_or_name, str):
        raise RuntimeError(
            "Unsupported type for global function! Must be either a callable or "
            "string name"
        )

    if callable(fn_or_name):
        assert not isinstance(fn_or_name, str)  # to make mypy happy
        fn_name = fn_or_name.__name__
    else:
        assert isinstance(
            fn_or_name, str
        ), "fn_or_name must be a global function or string name"
        fn_name = fn_or_name

    currentframe = inspect.currentframe()
    assert currentframe is not None
    f = currentframe.f_back
    assert f is not None
    if f.f_code.co_name != "<module>" and f.f_code.co_name != "inner_factory":
        raise NotImplementedError("wrap must be called at the top level of a module")

    # consider implementing Callable version of this via _autowrap_function_ids / _autowrap_search
    # semantics would be slightly different, but would add support `from x import wrapped_function`
    if dynamic:
        patcher = _Patcher()
        patcher.patch(f.f_globals, fn_name, _dummy_tracing_func(fn_or_name))
    _wrapped_fns_to_patch.append((f.f_globals, fn_name))
    return fn_or_name


@compatibility(is_backward_compatible=True)
def symbolic_trace(
    root: Union[ivy.Module, Callable[..., Any]],
    args: Iterable[Union[ivy.Array, ivy.NativeArray]],
    constant_args: Optional[Dict[str, Any]] = None,
    frontend: Optional[str] = None,
    to_ivy: bool = False,
    with_numpy: bool = False,
    generate_source: bool = False,
    control_flow: bool = False,
    debug_mode: bool = False,
    stateful: Optional[List] = [],
    **kwargs,
) -> Graph:
    """
    Symbolic tracing API

    Given a ``ivy.Module`` or function instance ``root``, this function will return an FX Graph
    constructed by recording operations seen while tracing through ``root``.

    ``constant_args`` allows you to partially specialize your function, whether it's to remove control flow or data structures.

    For example::

        def f(a, b):
            if b == True:
                return a
            else:
                return a*2

    FX can typically not trace through this due to the presence of control
    flow. However, we can use `constant_args` to specialize on the value of
    `b` to trace through this::

        f = fx.symbolic_trace(f, constant_args={'b': False})
        assert f(3, False)  == 6

    Note that although you can still pass in different values of `b`, they will be ignored.

    We can also use `constant_args` to eliminate data-structure handling from
    our function. This will use pytrees to flatten your input. To avoid
    overspecializing, pass in `fx.PH` for values that shouldn't be
    specialized. For example::

        def f(x):
            out = 0
            for v in x.values():
                out += v
            return out
        f = fx.symbolic_trace(f, constant_args={'x': {'a': fx.PH, 'b': fx.PH, 'c': fx.PH}})
        assert f({'a': 1, 'b': 2, 'c': 4}) == 7


    Args:
        - root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted
            into a Graph representation.
        - constant_args (Optional[Dict[str, any]]): Constant inputs to be partially specialized
        - args Iterable[Union[ivy.Array, ivy.NativeArray]]: inputs to initialize the proxies with.
        - kwargs (Optional[Dict[str, any]]) : keyword inputs to initialize the proxies with.
        ``(Note):`` There is no real computation performed, the inputs are merely used to create specialized
        ``IvyProxies`` that are more stable when working with frontends/ivy functions.
        - frontend (Optional[str, None]): The frontend framework to wrap. Default is None
        - to_ivy (bool): Whether to trace into ivy functions or treat them as leaf nodes. Default is False.
        - to_numpy (bool): Whether to track numpy function calls. Default is False.
        - generate_source (bool): Whether to reload the sourcecode of the Ivy Graph. Default is False.
    Returns:
        Graph: an fx graph created from the recorded operations from ``root``.
    """

    """TODO(yusha): maybe move the backend framework wrapping to the Patcher as well in the future."""
    args = ivy.nested_map(args, lambda a: ivy.native_array(a) if ivy.is_array(a) else a, shallow=False)
    kwargs = ivy.nested_map(kwargs, lambda a: ivy.native_array(a) if ivy.is_array(a) else a, shallow=False)
    # wrap the native backend functions
    _wrap_functions_for_dummy_tracing(
        to_ivy=to_ivy,
        with_numpy=with_numpy,
        _wrap_fn= lambda fn: _dummy_tracing_func(fn),
        path=None,
    )

    with _Patcher() as patcher:
        framework = ivy.current_backend_str()
        ivy_modules = _load_modules_from(glob.MODULES_TO_WRAP["ivy"], add_path=None)
        backend_modules = _load_modules_from(
            glob.MODULES_TO_WRAP[framework], add_path="backend"
        )
        frontend_modules = (
            None
            if not frontend
            else _load_modules_from(glob.MODULES_TO_WRAP[frontend], add_path="frontend")
        )
        # custom wrap ivy functions
        _patch_modules(
            patcher,
            ivy_modules,
            lambda fn: add_custom_decorator(
                [proxies_to_native_arrays, native_arrays_to_proxies],
                fn,
                positions=(0, -1),
            ),
            framework="ivy",
        )
        # # custom wrap backend functions
        # _patch_modules(
        #     patcher,
        #     backend_modules,
        #     lambda fn: add_custom_decorator(
        #         [proxies_to_native_arrays, native_arrays_to_proxies],
        #         fn,
        #         positions=(0, -1),
        #     ),
        #     framework=framework,
        # )
        if frontend is not None:
            # custom wrap frontend functions
            _patch_modules(
                patcher,
                frontend_modules,
                lambda fn: add_custom_decorator(
                    [proxies_to_ivy_arrays, ivy_arrays_to_proxies],
                    fn,
                    positions=(0, -1),
                ),
                framework=frontend,
            )

        # explicitly wrap ivy control flow ops since we wont trace into tf.cond/jax.lax.cond etc.
        for f in [ivy.if_else, ivy.if_exp, ivy.for_loop, ivy.while_loop]:
            patcher.patch_method(ivy_modules[0], f.__name__, _dummy_tracing_func(f))

        # explicitly wrap all ast transform functions
        for name, f in cfe.transform_funcs.items():
            patcher.patch_method(cfe, name, _dummy_tracing_func(f))

        # transform the root function if control_flow = True 
        if control_flow:
            try:
                if isinstance(root, IvyGraph):
                    root._scripted_call = cfe.to_functional_form(root._scripted_call)
                else:
                    root = cfe.to_functional_form(root)
            except ASTTransformationError:
                raise ivy.utils.exceptions.IvyException(
                    message="Error while AST transforming the function."
                )
        try:
            tracer = Tracer()
            if isinstance(root, IvyGraph):
                root._scripted_call.constants = root.constants
                tracer_graph = tracer.trace(
                    root._scripted_call,
                    constant_args,
                    args=args,
                    frontend=frontend,
                    **kwargs,
                    **root.constants,
                )
            else:
                tracer_graph = tracer.trace(
                    root, constant_args, args=args, frontend=frontend, **kwargs
                )
        except SymTraceError:
            raise ivy.utils.exceptions.IvyException(
                message="Error while symbolically tracing the function."
            )
        if debug_mode:
            return tracer_graph, None
        # convert the tracer graph into an ivy graph
        try:
            ivy_graph = tracer_to_ivy_graph(
                tracer_graph, root, to_ivy=to_ivy, with_numpy=with_numpy, stateful=stateful,
            )
        except GraphConvertError:
            raise ivy.utils.exceptions.IvyException(
                message="Error while converting the tracer graph to an ivy graph."
            )
        if generate_source:
            glob.dummy_trace = True
            ivy_graph.reload_sourcecode()
            glob.dummy_trace = False

    # unwrap backend functions
    _unwrap_functions_from_dummy_tracing(
        to_ivy=to_ivy,
        with_numpy=with_numpy,
        _unwrap_fn=lambda fn: _unwrap_function_from_dummy_tracing(fn),
        path=None,
    )

    return tracer_graph, ivy_graph


# ----------------------------------------
# AST Transformed iterator/boolean helpers
# ----------------------------------------


def iter_proxy(iterable):
    if isinstance(iterable, (list, tuple)):
        return iterable

    if isinstance(iterable, Proxy):
        iterable.node.iter_proxy = True
        return (iterable,)


def dict_proxy(iterable):
    if isinstance(iterable, (dict)):
        return iterable
    if isinstance(iterable, Proxy):
        iterable.node.dict_proxy = True
        return {iterable.__name__: iterable}
