import dis
import copy
import sys
import inspect
import operator
import collections
import functools
"""TODO: remove torch dependencies."""
import torch

from dataclasses import is_dataclass, fields


from .graph import magic_methods, reflectable_magic_methods, inplace_methods, Graph
from typing import (
    Tuple,
    Dict,
    OrderedDict,
    Optional,
    Iterable,
    Any,
    Iterator,
    Callable,
    Union,
)
from .node import Target, Node, Argument, base_types, map_aggregate
from ._compatibility import compatibility
from .immutable_collections import Constant
import ivy

# import frontend tensors
from ivy.functional.frontends.torch import Tensor
from ivy.functional.frontends.jax import DeviceArray
from ivy.functional.frontends.tensorflow import EagerTensor
from ivy.functional.frontends.numpy import ndarray


__all__ = [
    "TracerBase",
    "GraphAppendingTracer",
    "TraceError",
    "Proxy",
    "Attribute",
    "Scope",
    "ScopeContextManager",
    "IvyProxy",
]


@compatibility(is_backward_compatible=False)
class Scope:
    """Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example::

        class Sub(torch.nn.Module):
            def forward(self, x):
                # This will be a call_method Node in GraphModule,
                # scope for this would be (module_path="sub", module_type=Sub)
                return x.transpose(1, 2)

        class M(torch.nn.Module):
            def __init__(self):
                self.sub = Sub()

            def forward(self, x):
                # This will be a call_method Node as well,
                # scope for this would be (module_path="", None)
                x = x.transpose(1, 2)
                x = self.sub(x)
                return x

    """

    def __init__(self, module_path: str, module_type: Any):
        super().__init__()
        self.module_path = module_path
        self.module_type = module_type


@compatibility(is_backward_compatible=False)
class ScopeContextManager:
    """A context manager to track the Scope of Node during symbolic tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.
    """

    def __init__(
        self,
        scope: Scope,
        current_scope: Scope,
    ):
        super().__init__()
        # Keep a copy of prev scope to restore on exit
        self._prev_scope = copy.copy(scope)
        # Update scope to current scope
        scope.module_path = current_scope.module_path
        scope.module_type = current_scope.module_type
        # Save a reference so we can restore it
        self._scope = scope

    def __enter__(self):
        return self._scope

    def __exit__(self, *args):
        self._scope.module_path = self._prev_scope.module_path
        self._scope.module_type = self._prev_scope.module_type
        return


@compatibility(is_backward_compatible=True)
class TracerBase:
    graph: Graph

    check_mutable_operations: bool = False
    # Feature flag for assert tracing
    trace_asserts: bool = False
    # Feature flag for proxying accesses to buffer values
    proxy_buffer_attributes: bool = False

    # Name of the function to be traced. It will only be used when
    # ``root`` is an instance of ``nn.Module``
    traced_func_name: str = "forward"

    # Maps the containing module's name to the operator name
    scope: Scope

    # Records the module call stack
    module_stack: OrderedDict[str, str]

    # Mapping of node name to module scope
    node_name_to_scope: Dict[str, Tuple[str, type]]

    @compatibility(is_backward_compatible=True)
    def create_node(
        self,
        kind: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        """
        Inserts a graph node given target, args, kwargs, and name.

        This method can be overridden to do extra checking, validation, or
        modification of values used in node creation. For example, one might
        want to disallow in-place operations from being recorded.
        """

        node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
        # TODO node_name_to_scope will be depreciated in favor of
        # node.meta['nn_module_stack']
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )
        return node

    @compatibility(is_backward_compatible=True)
    def proxy(
        self,
        node: Node,
        data: Union[ivy.Array, ivy.NativeArray] = None,
        frontend: str = None,
    ) -> "Proxy":
        if frontend is not None:
            frontend_proxy = FRONTEND_PROXIES[frontend]
            return (
            frontend_proxy(node, self, data=data) if data is not None else Proxy(node, self)
        ) 
        assert not isinstance(data, ivy.Array), f"incompatible type({type(data)} passed when creating NativeProxies)"
        return (
            NativeProxy(node, self, native_data=data) if data is not None else Proxy(node, self)
        )

    @compatibility(is_backward_compatible=True)
    def create_proxy(
        self,
        kind: str,
        target: Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        proxy_factory_fn: Callable[[Node], "Proxy"] = None,
        data: Union[ivy.Array, ivy.NativeArray] = None,
        frontend: str = None,
    ):
        """
        Create a Node from the given arguments, then return the Node
        wrapped in a Proxy object.

        If kind = 'placeholder', then we're creating a Node that
        represents the parameter of a function. If we need to encode
        a default parameter, we use the ``args`` tuple. ``args`` is
        otherwise empty for ``placeholder`` Nodes.
        """

        args_ = self.create_arg(args)
        kwargs_ = self.create_arg(kwargs)
        assert isinstance(args_, tuple)
        assert isinstance(kwargs_, dict)

        node = self.create_node(kind, target, args_, kwargs_, name, type_expr)

        if not proxy_factory_fn:
            proxy = self.proxy(node, data, frontend=frontend)
        else:
            proxy = proxy_factory_fn(node)

        return proxy

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> Argument:
        """
        A method that lowers the objects seen as arguments during symbolic evaluation
        into Argument types that can be stored in IR.

        Can be override to support more trace-specific types.
        """
        if not isinstance(a, Proxy) and hasattr(a, "__fx_create_arg__"):
            return a.__fx_create_arg__(self)
        # aggregates
        elif isinstance(a, tuple) and hasattr(a, "_fields"):
            # NamedTuple constructors don't seem to like getting a generator
            # expression as an argument to their constructor, so build this
            # intermediate tuple and unpack it into the NamedTuple constructor
            args = tuple(self.create_arg(elem) for elem in a)
            return type(a)(*args)  # type: ignore[arg-type]
        elif isinstance(a, (tuple, list)):
            return type(a)(self.create_arg(elem) for elem in a)
        elif isinstance(a, dict):
            r = {}
            for k, v in a.items():
                # Check for invalid dict keys. We do not want a Proxy to appear
                # anywhere within the key. Since keys can be collection types,
                # we iterate through the key with map_aggregate
                k = self.create_arg(k)

                def no_node(arg):
                    if isinstance(arg, Node):
                        raise RuntimeError(
                            "Keys for dictionaries used as an argument cannot contain a "
                            f"Node. Got key: {k}"
                        )

                map_aggregate(k, no_node)

                r[k] = self.create_arg(v)
            return r
        elif isinstance(a, slice):
            return slice(
                self.create_arg(a.start),
                self.create_arg(a.stop),
                self.create_arg(a.step),
            )

        elif isinstance(a, range):
            return range(
                self.create_arg(a.start),
                self.create_arg(a.stop),
                self.create_arg(a.step),
            )

        if isinstance(a, Proxy):
            # base case: we unwrap the Proxy object
            return a.node

        if is_dataclass(a):
            kwargs = {
                field.name: self.create_arg(getattr(a, field.name))
                for field in fields(a)
            }
            return self.create_node("call_function", a.__class__, (), kwargs)

        if inspect.isfunction(a):
            return a.__name__
        elif isinstance(a, base_types) or a is None or a is ...:
            return a
        return Constant(a)
        raise NotImplementedError(f"argument of type: {type(a)}")

    @compatibility(is_backward_compatible=True)
    def to_bool(self, obj: "Proxy") -> bool:
        """Called when a proxy object is being converted to a boolean, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return a value.
        """
        raise TraceError(
            "symbolically traced variables cannot be used as inputs to control flow"
        )

    @compatibility(is_backward_compatible=True)
    def iter(self, obj: "Proxy") -> Iterator:
        """Called when a proxy object is being iterated over, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return an iterator.
        """
        raise TraceError(
            "Proxy object cannot be iterated. This can be "
            "attempted when the Proxy is used in a loop or"
            " as a *args or **kwargs function argument. "
            "See the torch.fx docs on pytorch.org for a "
            "more detailed explanation of what types of "
            "control flow can be traced, and check out the"
            " Proxy docstring for help troubleshooting "
            "Proxy iteration errors"
        )

    @compatibility(is_backward_compatible=True)
    def keys(self, obj: "Proxy") -> Any:
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an
        iterator it ** is suppose to work in your custom tracer.
        """
        return Attribute(obj, "keys")()


# used in Proxy object when just appending to the graph while not tracing.
@compatibility(is_backward_compatible=True)
class GraphAppendingTracer(TracerBase):
    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph
        self.scope = Scope("", None)
        self.module_stack = collections.OrderedDict()
        self.node_name_to_scope = {}


@compatibility(is_backward_compatible=False)
def assert_fn(x):
    assert x


@compatibility(is_backward_compatible=True)
class TraceError(ValueError):
    pass


@compatibility(is_backward_compatible=True)
class Proxy:
    """
    ``Proxy`` objects are ``Node`` wrappers that flow through the
    program during symbolic tracing and record all the operations
    (function calls, method calls, operators) that they touch
    into the growing FX Graph.

    If you're doing graph transforms, you can wrap your own ``Proxy``
    method around a raw ``Node`` so that you can use the overloaded
    operators to add additional things to a ``Graph``.

    ``Proxy`` objects cannot be iterated. In other words, the symbolic
    tracer will throw an error if a ``Proxy`` is used in a loop or as
    an ``*args``/``**kwargs`` function argument.

    There are two main ways around this:
    1. Factor out the untraceable logic into a top-level function and
    use ``fx.wrap`` on it.
    2. If the control flow is static (i.e. the loop trip count is
    based on some hyperparameter), the code can be kept in its original
    position and refactored into something like::

        for i in range(self.some_hyperparameter):
            indexed_item = proxied_value[i]

    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None):
        if tracer is None:
            # This allows you to create a Proxy object around a raw Node
            tracer = GraphAppendingTracer(node.graph)
        self.tracer = tracer
        self.node = node
        self._meta_tensor = data
        self.frontend = None 
        
    def __repr__(self) -> str:
        return f"Proxy({self.node.name})"

    def __getattr__(self, k) -> "Attribute":
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

    def __call__(self, *args, **kwargs) -> "Proxy":
        return self.tracer.create_proxy(
            "call_method", "__call__", (self,) + args, kwargs, data=self.data
        )

    def __iter__(self) -> Iterable["Proxy"]:
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        inst_list = list(dis.get_instructions(calling_frame.f_code))
        if sys.version_info >= (3, 11):
            from bisect import bisect_left

            inst_idx = bisect_left(
                inst_list, calling_frame.f_lasti, key=lambda x: x.offset
            )
        else:
            inst_idx = calling_frame.f_lasti // 2
        inst = inst_list[inst_idx]
        if inst.opname == "UNPACK_SEQUENCE":
            if self.node.target.__name__ in ("if_else", "while_loop", "for_loop"):
                idx = 0

                def _index_proxy(self, val):
                    nonlocal idx
                    if isinstance(val, Proxy):
                        val = self[idx]
                    idx += 1
                    return val

                return iter(
                    ivy.nested_map(
                        self.node.meta["orig_ret"], lambda x: _index_proxy(self, x)
                    )
                )
            return (self[i] for i in range(inst.argval))  # type: ignore[index]

        return self.tracer.iter(self)

    def __bool__(self) -> bool:
        if self.tracer.trace_asserts:
            # check if this boolean is used in an assertion, bytecode pattern for assertions
            # is pretty stable for Python 3.7--3.9
            frame = inspect.currentframe()
            assert frame is not None
            calling_frame = frame.f_back
            assert calling_frame is not None
            insts = list(dis.get_instructions(calling_frame.f_code))
            if sys.version_info >= (3, 11):
                from bisect import bisect_left

                cur = bisect_left(insts, calling_frame.f_lasti, key=lambda x: x.offset)
            else:
                cur = calling_frame.f_lasti // 2
            inst = insts[cur]

            if inst.opname == "POP_JUMP_IF_TRUE":
                first = insts[cur + 1]
                assert inst.arg is not None
                last = insts[inst.arg // 2 - 1]
                starts_with_assert = (
                    first.opname == "LOAD_GLOBAL"
                    and first.argval == "AssertionError"
                    or first.opname == "LOAD_ASSERTION_ERROR"
                )
                if starts_with_assert and last.opname == "RAISE_VARARGS":
                    self.tracer.create_proxy("call_function", assert_fn, (self,), {})
                    return True
        return self.tracer.to_bool(self)

    @compatibility(is_backward_compatible=True)
    def keys(self):
        return self.tracer.keys(self)

    # @property
    # def data(self):
    #     return self._data

    # @data.setter
    # def data(self, value):
    #     self._data = value

    def __len__(self):
        raise RuntimeError(
            "'len' is not supported in symbolic tracing by default. If you want "
            "this call to be recorded, please call fx.wrap('len') at "
            "module scope"
        )


@compatibility(is_backward_compatible=True)
class IvyProxy(ivy.Array, Proxy):
    """
    A special proxy which lets "shape","dtype","size", and a few other
    attribute accesses pass through to our underlying  Ivy API,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None):
        super(IvyProxy, self).__init__(data) 
        Proxy.__init__(self, node, tracer, data)
        self._ivy_data = data

    def __repr__(self):
        return f"IvyProxy({self.node.name})"

@compatibility(is_backward_compatible=True)
class NativeProxy(Proxy):
    """
    A special proxy which lets "shape","dtype","size", and a few other
    attribute accesses pass through to our underlying  Ivy API,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, native_data=None):
        super(NativeProxy, self).__init__(node, tracer, native_data) 
        self._native_data = native_data

    def __repr__(self):
        return f"NativeProxy({self.node.name})"
    
    def __getattr__(self, k) -> "Attribute":
        return getattr(self._native_data, k)



# Frontend Proxies
@compatibility(is_backward_compatible=True)
class Torch_FrontendProxy(Tensor, Proxy):
    """
    A special proxy which lets "shape","dtype","size", and a few other
    attribute accesses pass through to our underlying  Frontend API,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None):
        super(Torch_FrontendProxy, self).__init__(data)
        Proxy.__init__(self, node, tracer, data)
        self._ivy_data = self._ivy_array
        self.frontend = 'torch'

    def __repr__(self):
        return f"Torch_FrontendProxy({self.node.name})"


@compatibility(is_backward_compatible=True)
class JAX_FrontendProxy(DeviceArray, Proxy):
    """
    A special proxy which lets "shape","dtype","size", and a few other
    attribute accesses pass through to our underlying  Frontend API,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None):
        super(JAX_FrontendProxy, self).__init__(data)
        Proxy.__init__(self, node, tracer, data)
        self._native_data = self._data
        self.frontend = 'jax'

    def __repr__(self):
        return f"JAX_FrontendProxy({self.node.name})"


@compatibility(is_backward_compatible=True)
class TF_FrontendProxy(EagerTensor, Proxy):
    """
    A special proxy which lets "shape","dtype","size", and a few other
    attribute accesses pass through to our underlying  Frontend API,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None):
        super(TF_FrontendProxy, self).__init__(data)
        Proxy.__init__(self, node, tracer, data)
        self._native_data = self._data
        self.frontend = 'tensorflow'

    def __repr__(self):
        return f"TF_FrontendProxy({self.node.name})"


@compatibility(is_backward_compatible=True)
class Numpy_FrontendProxy(ndarray, Proxy):
    """
    A special proxy which lets "shape","dtype","size", and a few other
    attribute accesses pass through to our underlying  Frontend API,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None):
        super(Numpy_FrontendProxy, self).__init__(shape=data.shape, dtype=data.dtype)
        Proxy.__init__(self, node, tracer, data)
        self._native_data = self._data
        self.frontend = 'numpy'

    def __repr__(self):
        return f"Numpy_FrontendProxy({self.node.name})"


@compatibility(is_backward_compatible=True)
class Attribute(Proxy):
    @compatibility(is_backward_compatible=True)
    def __init__(self, root: Proxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Optional[Node] = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}, data=None
            ).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy(
            "call_method", self.attr, (self.root,) + args, kwargs, data=None
        )

@compatibility(is_backward_compatible=False)
class ParameterProxy(Proxy):
    """
    A special proxy which lets "shape", "size", "dim", and a few other
    attribute accesses pass through to the underlying  module parameter object,
    so that conditional tests on these attributes will not throw exception during tracing
    """
    def __init__(self, tracer: TracerBase, node: Node, name, param):
        super().__init__(node, tracer)
        assert(isinstance(param, torch.nn.Parameter))
        self.param = param
        self.name = name

    def __repr__(self) -> str:
        return f'ParameterProxy({self.name})'

    @property
    def shape(self):
        return self.param.shape

    def size(self):
        return self.param.size()

    def dim(self):
        return self.param.dim()

    @property
    def ndim(self):
        return self.param.ndim

    def numel(self):
        return self.param.numel()

    def nelement(self):
        return self.param.nelement()

# define dunder methods for the Proxy class     
for method in magic_methods:

    def _scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            return tracer.create_proxy(
                "call_function", target, args, kwargs, data=None,
            )

        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(Proxy, as_magic, impl) 

    _scope(method)


def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        target = getattr(operator, orig_method_name)
        return self.tracer.create_proxy(
            "call_function", target, (rhs, self), {}, data=None
        )

    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(Proxy, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)

for method in inplace_methods:

    def _scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            return tracer.create_proxy(
                "call_function", target, args, kwargs, data=None
            )

        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(Proxy, as_magic, impl)

    _scope(method)

# define dunder methods for the NativeProxy class     
for method in magic_methods:

    def _scope(method):
        def impl(*args, **kwargs):
            native_method =  getattr(args[0]._native_data, method)
            return native_method(*args,**kwargs)

        impl.__name__ = method
        setattr(NativeProxy, method, impl) 
    method = f'__{method.strip("_")}__'
    _scope(method)


def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        native_reflectable = getattr(self._native_data, orig_method_name)
        return native_reflectable(self, rhs)

    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(NativeProxy, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)

for method in inplace_methods:

    def _scope(method):
        def impl(*args, **kwargs):
            native_inp_method = getattr(args[0]._native_data, method)
            return native_inp_method(*args, **kwargs)

        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(NativeProxy, as_magic, impl)

    _scope(method)

FRONTEND_PROXIES = {
    "torch": Torch_FrontendProxy,
    "jax": JAX_FrontendProxy,
    "tensorflow": TF_FrontendProxy,
    "numpy": Numpy_FrontendProxy,
}
