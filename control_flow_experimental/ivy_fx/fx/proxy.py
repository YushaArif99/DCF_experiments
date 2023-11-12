import dis
import copy
import sys
import inspect
import operator
import collections
import functools
import builtins
from enum import Enum
from numbers import Number
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
import graph_compiler.globals as glob
from graph_compiler.conversion import _to_ND
import control_flow_experimental.dy2static as dy2s
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

class ProxyDataError(Exception):
    pass

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

class ProxyType(Enum):
    IVY_PROXY = 1
    FRONTEND_PROXY = 2
    NATIVE_PROXY = 3
    SHAPE_PROXY = 4
    DTYPE_PROXY = 5
    SCALAR_PROXY = 6
    LIST_PROXY = 8
    DICT_PROXY = 9
    TUPLE_PROXY = 10
    SET_PROXY = 11
    ENUMERATE_PROXY = 12
    ZIP_PROXY = 13
    RANGE_PROXY = 14
    NDARRAY_PROXY = 15
    UNDEFINED_PROXY = 16
    EAGER_PROXY = 17

class ProxyFactory:
    def __init__(self, to_ivy, frontend, with_numpy):
        self.to_ivy = to_ivy 
        self.frontend = frontend
        self.with_numpy = with_numpy 
    
    def create_scalar_proxy(self,node, tracer, data):
        return ScalarProxy(node, tracer, data) 
    
    def create_array_proxy(self,node, tracer, data):
        native_proxy = NativeProxy(node, tracer, native_data=data) 
        if self.to_ivy:
            return IvyProxy(node, tracer, data=data, native_proxy=native_proxy, to_ivy=True) 
        elif self.frontend: 
            ivy_proxy = IvyProxy(node, tracer, data=data, native_proxy=native_proxy, to_ivy=True)
            frontend_proxy = FRONTEND_PROXIES[self.frontend]
            return frontend_proxy(node, tracer, data=data, ivy_proxy=ivy_proxy, frontend=self.frontend)
        return native_proxy
    
    def create_shape_proxy(self,node, tracer, data):
        shape_proxy = ShapeProxy(node, tracer, shape=data)
        if self.to_ivy:
            return IvyShapeProxy(node, tracer, shape=data, shape_proxy=shape_proxy, to_ivy=True)  
    def create_dtype_proxy(self,node, tracer, data):
        dtype_proxy = DtypeProxy(node, tracer, dtype=data)
        if self.to_ivy:
            return IvyDtypeProxy(node, tracer, shape=data, dtype_proxy=dtype_proxy, to_ivy=True) 

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
        data: Union[ivy.Array, ivy.NativeArray, Number, Iterable[Number]] = None,
        proxy_type: ProxyType = None,
        frontend: str = None,
        to_ivy=False,
        with_numpy=True,
    ) -> "Proxy":
        # FP -> IP -> NP ; where
        # FP : FrontendProxy 
        # IP : IvyProxy 
        # NP : NativeProxy 
        # -> : has-a relation 
        
        factory = ProxyFactory(to_ivy, frontend, with_numpy)

        if proxy_type == ProxyType.SCALAR_PROXY:
            return factory.create_scalar_proxy(node, self, data)
        elif proxy_type == ProxyType.DTYPE_PROXY:
            return factory.create_dtype_proxy(node, self, data)
        elif proxy_type == ProxyType.SHAPE_PROXY:
            return factory.create_shape_proxy(node,self,data)
        elif proxy_type == ProxyType.UNDEFINED_PROXY:
            return Proxy(node, self, data)
        elif proxy_type == ProxyType.EAGER_PROXY:
            return EagerProxy(node, self, data)
        elif proxy_type == ProxyType.LIST_PROXY:
            return ListProxy(node, self, data)
        elif proxy_type == ProxyType.DICT_PROXY:
            return DictProxy(node, self, data)
        elif proxy_type == ProxyType.TUPLE_PROXY:
            return TupleProxy(node, self, data)
        elif proxy_type == ProxyType.SET_PROXY:
            return SetProxy(node, self, data)
        elif proxy_type == ProxyType.ENUMERATE_PROXY:
            data = (data,) if not isinstance(data, tuple) else data
            return EnumerateProxy(node, self, *data)
        elif proxy_type == ProxyType.ZIP_PROXY:
            return ZipProxy(node, self, *data)
        elif proxy_type == ProxyType.RANGE_PROXY:
            data = (data,) if not isinstance(data, tuple) else data
            return RangeProxy(node, self, *data)
        else:
            if data is None: 
                data = ivy.native_array([]) 

            if ivy.current_backend_str() == "numpy" or with_numpy or proxy_type == ProxyType.NDARRAY_PROXY:
                data = _to_ND(data) 
            
            return factory.create_array_proxy(node, self, data)
        

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
        proxy_type: ProxyType = None,
        frontend: str = None,
        to_ivy=False,
        with_numpy=True,
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
            proxy = self.proxy(node, data, proxy_type=proxy_type, frontend=frontend, to_ivy=to_ivy, with_numpy=with_numpy)
        else:
            proxy = proxy_factory_fn(node,data)

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

                # def no_node(arg):
                #     if isinstance(arg, Node):
                #         raise RuntimeError(
                #             "Keys for dictionaries used as an argument cannot contain a "
                #             f"Node. Got key: {k}"
                #         )

                # map_aggregate(k, no_node)

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
        return Constant(None) if isinstance(a, dy2s.UndefinedVar) else Constant(a)
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
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None, frontend=None, to_ivy=False):
        if tracer is None:
            # This allows you to create a Proxy object around a raw Node
            tracer = GraphAppendingTracer(node.graph)
        self.tracer = tracer
        self.node = node
        self._meta_tensor = data
        self._meta_frontend = frontend 
        self._meta_to_ivy = to_ivy

    def __repr__(self) -> str:
        return f"Proxy({self.node.name})"

    def __getattr__(self, k) -> "Attribute":
        return Attribute(self, k)

    def __call__(self, *args, **kwargs) -> "Proxy":
        return self.tracer.create_proxy(
            "call_method", "__call__", (self,) + args, kwargs, data=self.data
        )

    def __iter__(self, frame_=None) -> Iterable["Proxy"]:
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame_ if frame_ else frame.f_back 
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
                        lambda x: _index_proxy(self, x), self.node.meta["orig_ret"],
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

    def __len__(self):
        raise RuntimeError(
            "'len' is not supported in symbolic tracing by default. If you want "
            "this call to be recorded, please call fx.wrap('len') at "
            "module scope"
        )
    
    def _to_ivy_proxy(self, x):
        if isinstance(x, NativeProxy):
            return IvyProxy(node=x.node, tracer=x.tracer, data=x._native_data, native_proxy=x)
        elif hasattr(x, "ivy_array"):
            return x.ivy_array
        return x

# Native Proxies
@compatibility(is_backward_compatible=True)
class NativeProxy(Proxy):
    """
    A specialized proxy designed to mimic the behavior of native arrays during symbolic tracing.
    This proxy holds an instance of the underlying native array (e.g., a PyTorch tensor) as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of the native array, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, native_data=None, frontend=None, to_ivy=False):
        super(NativeProxy, self).__init__(node, tracer, native_data, frontend, to_ivy) 
        self._native_data = native_data

    def __repr__(self):
        return f"NativeProxy({self.node.name})"
    
    def __getattr__(self, k) -> "NativeAttribute":
        native_attr = getattr(self._native_data, k)
        if k in glob.GRAPH_ATTRIBUTES[ivy.current_backend_str()]:
            def _factory_fn(node,data):
                if k == "shape":
                    return ShapeProxy(node=node, tracer=self.tracer, shape=data)
                return self.tracer.proxy(node, data)
            return self.tracer.create_proxy(
                "call_function", getattr, (self, k), {}, data=native_attr, proxy_factory_fn=_factory_fn,
            )
        if callable(native_attr):
            return NativeAttribute(self, getattr(type(self._native_data), k))
        else:
            return native_attr

@compatibility(is_backward_compatible=True)
class NativeAttribute(NativeProxy):
    @compatibility(is_backward_compatible=True)
    def __init__(self, root: NativeProxy, attr: Callable):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Optional[Node] = None

    def __getattr__(self, k):
        return getattr(self.attr, k) 
    
    def __repr__(self):
        return f"NativeAttribute({self.attr.__name__})"
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
        nargs = (self.root,) + args 
        return self.attr(*nargs, **kwargs)
        
        
@compatibility(is_backward_compatible=True)
class NativeShapeProxy(Proxy):
    """
    A special proxy which lets "shape","dtype","size", and a few other
    attribute accesses pass through to our underlying  Ivy API,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, shape=None, frontend=None, to_ivy=False):
        super(NativeShapeProxy, self).__init__(node, tracer, shape, frontend, to_ivy) 
        self._native_shape = shape 

    def __getattr__(self, item):
        shape_attr = getattr(self._native_shape, item)

        if callable(shape_attr):
            return NativeAttribute(self, getattr(type(self._native_shape), item))
        else:
            return shape_attr
    
    def __iter__(self):
        return iter(self._native_shape)
    
    def __len__(self):
        return len(self._native_shape)
    
    def __repr__(self):
        return f"NativeShapeProxy({self.node.name})"


@compatibility(is_backward_compatible=True)
class NativeShapeProxy(Proxy):
    """
    A special proxy which lets "shape","dtype","size", and a few other
    attribute accesses pass through to our underlying  Ivy API,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, shape=None, frontend=None, to_ivy=False):
        super(NativeShapeProxy, self).__init__(node, tracer, shape, frontend, to_ivy) 
        self._native_shape = shape 

    def __getattr__(self, item):
        shape_attr = getattr(self._native_shape, item)

        if callable(shape_attr):
            return NativeAttribute(self, getattr(type(self._native_shape), item))
        else:
            return shape_attr
    
    def __iter__(self):
        return iter(self._native_shape)
    
    def __len__(self):
        return len(self._native_shape)
    
    def __repr__(self):
        return f"NativeShapeProxy({self.node.name})"

# Ivy Proxies
@compatibility(is_backward_compatible=True)
class IvyProxy(ivy.Array, Proxy):
    """
    A specialized proxy designed to mimic the behavior of ivy arrays during symbolic tracing.
    It also acts as a wrapper around the NativeProxy class storing the instance as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of ivy.Arrays, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None, native_proxy=None, frontend=None, to_ivy=False):
        super(IvyProxy, self).__init__(data) 
        Proxy.__init__(self, node, tracer, data, frontend,to_ivy)
        self._ivy_data = self._data
        self._data = native_proxy # override the _data to return a NativeProxy

    def __getattr__(self, item):
        return getattr(self._data, item)
    
    @property
    def shape(self):
        native_shape = ShapeProxy(node=self.node, tracer=self.tracer,shape=self._ivy_data.shape)
        return IvyShapeProxy(node=self.node, tracer=self.tracer, shape=self._ivy_data.shape, shape_proxy=native_shape)
    
    @property
    def data(self) -> NativeProxy:
        """The native proxy being wrapped in self."""
        return self._data
    
    @data.setter
    def data(self, data):
        self._data = data

    def __repr__(self):
        return f"IvyProxy({self.node.name})"

# Frontend Proxies
@compatibility(is_backward_compatible=True)
class Torch_FrontendProxy(Tensor, Proxy):
    """
    A specialized proxy designed to mimic the behavior of a frontend Torch.Tensor during symbolic tracing.
    It also acts as a wrapper around the IvyProxy class storing the instance as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of frontend tensor, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None, ivy_proxy=None, frontend=None, to_ivy=False, **kwargs):
        super(Torch_FrontendProxy, self).__init__(array=data, **kwargs)
        Proxy.__init__(self, node, tracer, data, frontend, to_ivy)
        self.frontend = 'torch'
        self._ivy_data = self._ivy_array
        self._ivy_array = ivy_proxy  #override the _ivy_array to return an IvyProxy
    
    @property
    def ivy_array(self) -> IvyProxy:
        """The ivy proxy being wrapped in self."""
        return self._ivy_array
    
    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            IvyProxy(node=array.node, tracer=array.tracer, data=array._native_data, native_proxy=array ) if isinstance(array, NativeProxy) else array
        )
    def __getitem__(self, query, /):
        ivy_args = ivy.nested_map(lambda a:Torch_FrontendProxy._to_ivy_proxy(self,a),[self, query] )
        ret = ivy.get_item(*ivy_args)
        return Torch_FrontendProxy(node=ret.node, tracer=ret.tracer, data=ret._ivy_data, ivy_proxy=ret)

    def __setitem__(self, key, value, /):
        key, value = ivy.nested_map(lambda a:Torch_FrontendProxy._to_ivy_proxy(self,a),[key, value] )
        self.ivy_array[key] = value

    def __repr__(self):
        return f"Torch_FrontendProxy({self.node.name})"


@compatibility(is_backward_compatible=True)
class JAX_FrontendProxy(DeviceArray, Proxy):
    """
    A specialized proxy designed to mimic the behavior of a frontend DeviceArray during symbolic tracing.
    It also acts as a wrapper around the IvyProxy class storing the instance as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of frontend tensor, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None,ivy_proxy=None, frontend=None, to_ivy=False, **kwargs):
        super(JAX_FrontendProxy, self).__init__(array=data, **kwargs)
        Proxy.__init__(self, node, tracer, data, frontend, to_ivy)
        self.frontend = 'jax'
        self._ivy_data = self._ivy_array
        self._ivy_array = ivy_proxy  #override the _ivy_array to return an IvyProxy

    def __repr__(self):
        return f"JAX_FrontendProxy({self.node.name})"


@compatibility(is_backward_compatible=True)
class TF_FrontendProxy(EagerTensor, Proxy):
    """
    A specialized proxy designed to mimic the behavior of a frontend EagerTensor during symbolic tracing.
    It also acts as a wrapper around the IvyProxy class storing the instance as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of frontend tensor, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None, ivy_proxy=None, frontend=None, to_ivy=False):
        super(TF_FrontendProxy, self).__init__(array=data)
        Proxy.__init__(self, node, tracer, data, frontend, to_ivy)
        self.frontend = 'tensorflow'
        self._ivy_data = self._ivy_array
        self._ivy_array = ivy_proxy  #override the _ivy_array to return an IvyProxy

    def __getitem__(self, slice_spec, var=None, name="getitem"):
        ivy_args = ivy.nested_map(lambda a:TF_FrontendProxy._to_ivy_proxy(self,a),[self, slice_spec])
        ret = ivy.get_item(*ivy_args)
        return TF_FrontendProxy(node=ret.node, tracer=ret.tracer, data=ret._ivy_data, ivy_proxy=ret)

    def __repr__(self):
        return f"TF_FrontendProxy({self.node.name})"


@compatibility(is_backward_compatible=True)
class Numpy_FrontendProxy(ndarray, Proxy):
    """
    A specialized proxy designed to mimic the behavior of a frontend NDarray during symbolic tracing.
    It also acts as a wrapper around the IvyProxy class storing the instance as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of frontend tensor, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, shape=None, dtype="float32", ivy_proxy=None,frontend=None, to_ivy=False, **kwargs):
        super(Numpy_FrontendProxy, self).__init__(shape=shape, dtype=dtype, **kwargs)
        Proxy.__init__(self, node, tracer, self.ivy_array, frontend, to_ivy)
        self.frontend = 'numpy'
        self._ivy_data = self._ivy_array
        self._ivy_array = ivy_proxy  #override the _ivy_array to return an IvyProxy
    
    @property
    def ivy_array(self):
        return self._ivy_array
    
    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            IvyProxy(node=array.node, tracer=array.tracer, data=array._native_data, native_proxy=array) if not isinstance(array, IvyProxy) else array
        )

    def __getitem__(self, key, /):
        ivy_args = ivy.nested_map(lambda a: Numpy_FrontendProxy._to_ivy_proxy(self, a), [self, key])
        ret = ivy.get_item(*ivy_args)
        return Numpy_FrontendProxy(node=ret.node, tracer=ret.tracer, data=ret._ivy_data, ivy_proxy=ret, _init_overload=True)

    def __setitem__(self, key, value, /):
        key, value = ivy.nested_map(lambda a: Numpy_FrontendProxy._to_ivy_proxy(self, a),[key, value])
        self.ivy_array[key] = value

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
    
    def __repr__(self):
        return f"Attribute({self.attr})"

# Shape and Dtype Proxies
@compatibility(is_backward_compatible=True)
class ShapeProxy(Proxy):
    """
    A specialized proxy designed to mimic the behavior of native shapes during symbolic tracing.
    This proxy holds an instance of the underlying native shape (e.g. torch.Size) as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of the native shapes, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, shape=None, frontend=None, to_ivy=False):
        super(ShapeProxy, self).__init__(node, tracer, shape, frontend, to_ivy) 
        self._native_shape = shape 

    def __getattr__(self, item):
        shape_attr = getattr(self._native_shape, item)

        if callable(shape_attr):
            return NativeAttribute(self, getattr(type(self._native_shape), item))
        else:
            return shape_attr
    
    def __iter__(self):
        return iter(self._native_shape)
    
    def __len__(self):
        return len(self._native_shape)
    
    def __repr__(self):
        return f"ShapeProxy({self.node.name})"

@compatibility(is_backward_compatible=True)
class IvyShapeProxy(ivy.Shape, Proxy):
    """
    A specialized proxy designed to mimic the behavior of ivy.Shape during symbolic tracing.
    This proxy acts as a wrapper around the NativeShape class and holds its instance as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of the ivy.Shape, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, shape=None, shape_proxy=None, frontend=None, to_ivy=False):
        super(IvyShapeProxy, self).__init__(shape) 
        Proxy.__init__(self, node, tracer, shape, frontend, to_ivy)
        self._ivy_shape = self._shape
        self._shape = shape_proxy # override the _shape to return a ShapeProxy

    def __getattr__(self, item):
        return getattr(self._shape, item)
    
    def __repr__(self):
        return f"IvyShapeProxy({self.node.name})"

@compatibility(is_backward_compatible=True)
class DtypeProxy(Proxy):
    """
    A specialized proxy designed to mimic the behavior of native dtypes during symbolic tracing.
    This proxy holds an instance of the underlying native dtype (e.g., torch.dtype) as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of the native array, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, dtype=None, frontend=None, to_ivy=False):
        super(DtypeProxy, self).__init__(node, tracer, dtype, frontend, to_ivy) 
        self._native_dtype = dtype 

    def __getattr__(self, item):
        dtype_attr = getattr(self._native_dtype, item)

        if callable(dtype_attr):
            return NativeAttribute(self, getattr(type(self._native_dtype), item))
        else:
            return dtype_attr
    
    def __repr__(self):
        return f"DtypeProxy({self.node.name})"

@compatibility(is_backward_compatible=True)
class IvyDtypeProxy(ivy.Dtype, Proxy):
    """
    A specialized proxy designed to mimic the behavior of ivy.Dtype during symbolic tracing.
    This proxy acts as a wrapper around the NativeDtype class and holds its instance as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of the ivy.Dtype, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """
    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, dtype=None, dtype_proxy=None, frontend=None, to_ivy=False):
        super(IvyDtypeProxy, self).__init__(dtype) 
        Proxy.__init__(self, node, tracer, dtype, frontend, to_ivy)
        self._ivy_dtype = self._dtype
        self._dype = dtype_proxy

    def __getattr__(self, item):
        return getattr(self._dtype, item)
    
    def __repr__(self):
        return f"IvyDtypeProxy({self.node.name})"


@compatibility(is_backward_compatible=True)
class ScalarProxy(Proxy):
    """
    A specialized proxy designed to mimic the behavior of scalars during symbolic tracing.
    This proxy holds an instance of the underlying scalar (e.g., int, float, bool, str) as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of the scalar, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None, frontend=None, to_ivy=False):
        super(ScalarProxy, self).__init__(node, tracer, data) 
        self._scalar = data

    def __repr__(self):
        return f"ScalarProxy({self.node.name})"
    
    def __getattr__(self, k) -> "NativeAttribute":
        native_attr = getattr(self._scalar, k)
        if callable(native_attr):
            return NativeAttribute(self, getattr(type(self._scalar), k))
        else:
            return native_attr

@compatibility(is_backward_compatible=True)
class IteratorProxy(Proxy):
    """
    A specialized proxy designed to mimic the behavior of iterables during symbolic tracing.
    This proxy holds an instance of the underlying iterable (e.g., zip, enumerate, range) as an attribute.
    However, it's important to note that while the proxy can mimic the behavior of the collection, it does not hold any actual values. 
    Any computations performed during symbolic tracing will not yield meaningful results as they are operating on dummy data.
    """

    def __init__(self, node: Node, tracer: "Optional[TracerBase]" = None, data=None, frontend=None, to_ivy=False):
        super(IteratorProxy, self).__init__(node, tracer, data)  
        self._collection = data

    def __iter__(self):
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
            idx = 0

            res = []
            for idx,item in enumerate(self._collection):
                proxy = self.tracer.create_proxy("call_function", operator.getitem, (self,idx),{},data=item, proxy_type=ProxyType.EAGER_PROXY) 
                res.append(proxy)
            return iter(res)

        return (self[i] for i in range(inst.argval))  # type: ignore[index]
     
    def __getattr__(self, k):
        return getattr(self._collection, k)
        
    def __repr__(self):
        return f"IteratorProxy({self.node.name})"


class TupleProxy(IteratorProxy):
    def __init__(self, node, tracer, data=None):
        assert not isinstance(data, type(None))
        if isinstance(data, Proxy):
            if not isinstance(data, ITERABLE_PROXIES):
                raise ProxyDataError("Cannot convert a Proxy object to a tuple when using the `symbolic` mode.")
        super().__init__(node, tracer, data=data)
        self._collection = tuple(_get_concrete_val(data))
        assert isinstance(self._collection, tuple)
        
    def __getitem__(self, index):
        idx = map_aggregate(index, lambda a: _get_concrete_val(a))
        if _find_proxy(idx) is not None:
            raise ProxyDataError("cannot index a tuple with unknown index")
        
        value = self._collection[idx]
        return self.tracer.create_proxy("call_function", tuple.__getitem__, (self,index),{},data=value, proxy_type=ProxyType.EAGER_PROXY) 
    
    def __len__(self):
        return self.tracer.create_proxy("call_function", tuple.__len__, (self,),{},data=len(self._collection), proxy_type=ProxyType.SCALAR_PROXY)

    def __repr__(self):
        return f"TupleProxy({self.node.name})"
    
    def __contains__(self, value):
        return self.tracer.create_proxy("call_function", tuple.__contains__, (self,value),{},data=value in self._collection, proxy_type=ProxyType.EAGER_PROXY)

    def len(self):
        return self.__len__()
        
    def index(self, value, start=0, end=None):
        v,s,e = map_aggregate((value,start,end), lambda a: _get_concrete_val(a))
        if _find_proxy((v,s,e)) is not None:
            raise ProxyDataError("cannot index a tuple with unknown index")
        ret = self._collection.index(v,s,e)
        proxy = self.tracer.create_proxy("call_function", tuple.index, (self,value, start, end),{},data=ret, proxy_type=ProxyType.EAGER_PROXY)
        return proxy
    
    def count(self, value):
        val = map_aggregate(value, lambda a: _get_concrete_val(a))
        if _find_proxy(val) is not None:
            raise ProxyDataError("cannot index a tuple with unknown index")
        res = self._collection.count(val)
        return self.tracer.create_proxy("call_function", tuple.count, (self,value),{},data=res, proxy_type=ProxyType.EAGER_PROXY)


class SetProxy(IteratorProxy):
    def __init__(self, node, tracer, data=None):
        assert not isinstance(data, type(None))
        if isinstance(data, Proxy):
            if not isinstance(data, ITERABLE_PROXIES):
                raise ProxyDataError("Cannot convert a Proxy object to a set when using the `symbolic` mode.")
        super().__init__(node, tracer, data=data)
        self._collection = set(_get_concrete_val(data))
        assert isinstance(self._collection, set)
        
    def __len__(self):
        return self.tracer.create_proxy("call_function", set.__len__, (self,),{},data=len(self._collection), proxy_type=ProxyType.EAGER_PROXY)

    def __contains__(self, value):
        return self.tracer.create_proxy("call_function", set.__contains__, (self,value),{},data=value in self._collection, proxy_type=ProxyType.EAGER_PROXY)

    def __repr__(self):
        return f"SetProxy({self.node.name})"
    
    def len(self):
        return self.__len__()
        
    def add(self, value):
        self._collection.add(value)
        _ = self.tracer.create_proxy("call_function", set.add, (self,value),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)

    def remove(self, value):
        v = map_aggregate(value, lambda a: _get_concrete_val(a))
        if _find_proxy(v) is not None:
            raise ProxyDataError("cannot index a set with unknown index")
    
        self._collection.remove(v)
        _ = self.tracer.create_proxy("call_function", set.remove, (self,value),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)

    def discard(self,value):
        v = map_aggregate(value, lambda a: _get_concrete_val(a))
        if _find_proxy(v) is not None:
            raise ProxyDataError("cannot index a set with unknown index")
        
        self._collection.discard(value)
        _ = self.tracer.create_proxy("call_function", set.discard,(self,value),{},data=None ,proxy_type=ProxyType.UNDEFINED_PROXY)

    def pop(self):
        val = self._collection.pop()
        return self.tracer.create_proxy("call_function", set.pop,(self,),{},data=val ,proxy_type=ProxyType.EAGER_PROXY)
       
    def clear(self):
        self._collection.clear()
        _ = self.tracer.create_proxy("call_function", set.clear,(self,),{},data=None ,proxy_type=ProxyType.UNDEFINED_PROXY)

    def union(self, *others):
        others_ = map_aggregate(others, lambda a: _get_concrete_val(a))
        if _find_proxy(others_) is not None:
            raise ProxyDataError("Cannot convert a Proxy object to a set when using the `symbolic` mode.")   

        result = self._collection.union(*others_)
        return self.tracer.create_proxy("call_function", set.union, (self,) + others,{},data=result, proxy_type=ProxyType.SET_PROXY)

    def intersection(self, *others):
        others_ = map_aggregate(others, lambda a: _get_concrete_val(a))
        if _find_proxy(others_) is not None:
            raise ProxyDataError("Cannot convert a Proxy object to a set when using the `symbolic` mode.")   
        
        result = self._collection.intersection(*others_)
        return self.tracer.create_proxy("call_function", set.intersection, (self,) + others,{},data=result, proxy_type=ProxyType.SET_PROXY)
        
    

class DictProxy(IteratorProxy):
    def __init__(self, node, tracer, data=None):
        assert not isinstance(data, type(None))
        if isinstance(data, Proxy):
            if not isinstance(data, ITERABLE_PROXIES):
                raise ProxyDataError("Cannot convert a Proxy object to a dict when using the `symbolic` mode.")
        super().__init__(node, tracer, data=data)
        self._collection = dict(_get_concrete_val(data))
        assert isinstance(self._collection, dict)

    def __getitem__(self, key):
        k = map_aggregate(key, lambda a: _get_concrete_val(a))
        if _find_proxy(k) is not None:
            raise ProxyDataError("cannot index a dict with unknown key")
        
        val = self._collection[k]
        return self.tracer.create_proxy("call_function", dict.__getitem__, (self,key),{},data=val, proxy_type=ProxyType.EAGER_PROXY) 
        
    def __setitem__(self, key, value):
        k = map_aggregate(key, lambda a: _get_concrete_val(a))
        if _find_proxy(k) is not None:
            raise ProxyDataError("cannot index a dict with unknown key")
        
        self._collection[k] = value
        _ =  self.tracer.create_proxy("call_function", dict.__setitem__, (self,key,value),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)

    def __delitem__(self, key):
        k = map_aggregate(key, lambda a: _get_concrete_val(a))
        if _find_proxy(k) is not None:
            raise ProxyDataError("cannot index a dict with unknown key")
        
        del self._collection[key]
        _ =  self.tracer.create_proxy("call_function", dict.__delitem__, (self,key),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)
        
    def __contains__(self, key):
        val =  key in self._collection
        return self.tracer.create_proxy("call_function", dict.__contains__, (self,key),{},data=val, proxy_type=ProxyType.EAGER_PROXY)
        
    def __len__(self):
        return self.tracer.create_proxy("call_function", dict.__len__, (self,),{},data=len(self._collection), proxy_type=ProxyType.EAGER_PROXY)
    
    def __repr__(self):
        return f"DictProxy({self.node.name})"
    
    def len(self):
        return self.__len__()
        
    def keys(self):
        val = list(self._collection.keys())
        return self.tracer.create_proxy("call_function", dict.keys, (self,),{},data=val, proxy_type=ProxyType.EAGER_PROXY)
    
    def values(self):
        val = list(self._collection.items())
        return self.tracer.create_proxy("call_function", dict.values, (self,),{},data=val, proxy_type=ProxyType.EAGER_PROXY)
        
    def items(self):
        val = list(self._collection.items())
        return self.tracer.create_proxy("call_function", dict.items, (self,),{},data=val, proxy_type=ProxyType.EAGER_PROXY)
        
    def get(self, key, default=None):
        k= map_aggregate(key, lambda a: _get_concrete_val(a))
        if _find_proxy(k) is not None:
            raise ProxyDataError("cannot index a dict with unknown key")
        val = self._collection.get(k, default)
        return self.tracer.create_proxy("call_function", dict.get, (self,key,default),{},data=val, proxy_type=ProxyType.EAGER_PROXY)
        
    def pop(self, key, default=None):
        k = map_aggregate(key, lambda a: _get_concrete_val(a))
        if _find_proxy(k) is not None:
            raise ProxyDataError("cannot index a dict with unknown key")
        value = self._collection.pop(k, default)
        return self.tracer.create_proxy("call_function", dict.pop, (self,key,default),{},data=value, proxy_type=ProxyType.EAGER_PROXY)

    def popitem(self):
        key_value = self._collection.popitem()
        return self.tracer.create_proxy("call_function", dict.popitem, (self,),{},data=key_value, proxy_type=ProxyType.EAGER_PROXY)

    def update(self, other={}, **kwargs):
        if isinstance(other, Proxy):
            if isinstance(other, ITERABLE_PROXIES):
                self._collection.update(other._collection, **kwargs)
            else:
                raise ProxyDataError("Cannot convert a Proxy object to a dict when using the `symbolic` mode.")   
        else:
            self._collection.update(other, **kwargs)
        _ = self.tracer.create_proxy("call_function", dict.update, (self,),{"other": other},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)

    def clear(self):
        self._collection.clear()
        _ = self.tracer.create_proxy("call_function", dict.clear, (self,),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)

class ListProxy(IteratorProxy):
    def __init__(self, node, tracer, data=None):
        if isinstance(data, Proxy):
            if not isinstance(data, ITERABLE_PROXIES):
                raise ProxyDataError("Cannot convert a Proxy object to a list when using the `symbolic` mode.")
        super().__init__(node, tracer, data=data)
        self._collection= list(_get_concrete_val(data))
        assert isinstance(self._collection, list)

    def __getitem__(self, index):
        idx = map_aggregate(index, lambda a: _get_concrete_val(a))
        if _find_proxy(idx) is not None:
            raise ProxyDataError("cannot index a list with unknown index")
        
        value = self._collection[idx]
        return self.tracer.create_proxy("call_function", list.__getitem__, (self,index),{},data=value, proxy_type=ProxyType.EAGER_PROXY) 
    
    def __setitem__(self, index, value):
        idx = map_aggregate(index, lambda a: _get_concrete_val(a))
        if _find_proxy(idx) is not None:
            raise ProxyDataError("cannot index a list with unknown index")
        
        self._collection[idx] = value
        _ =  self.tracer.create_proxy("call_function", list.__setitem__, (self,index,value),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)
        
    def __delitem__(self, index):
        idx = map_aggregate(index, lambda a: _get_concrete_val(a))
        if _find_proxy(idx) is not None:
            raise ProxyDataError("cannot index a list with unknown index")
        
        del self._collection[idx]
        _ =  self.tracer.create_proxy("call_function", list.__delitem__, (self,index),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)

    def __len__(self):
        return self.tracer.create_proxy("call_function", list.__len__, (self,),{},data=len(self._collection), proxy_type=ProxyType.EAGER_PROXY)
     
    def __repr__(self):
        return f"ListProxy({self.node.name})"
    
    def len(self):
        return self.__len__()
        
    def append(self, value):
        self._collection.append(value)
        _ = self.tracer.create_proxy("call_function", list.append, (self,value),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)
    
    def extend(self, iterable):
        if isinstance(iterable, Proxy):
            if isinstance(iterable, ITERABLE_PROXIES):
                self._collection.extend(iterable._collection)
            else:
                raise ProxyDataError("Cannot convert a Proxy object to a list.extend when using the `symbolic` mode.")
        else:
            self._collection.extend(iterable)
        _ = self.tracer.create_proxy("call_function", list.extend, (self,iterable),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)

    def insert(self, index, value):
        idx = map_aggregate(index, lambda a: _get_concrete_val(a))
        if _find_proxy(idx) is not None:
            raise ProxyDataError("cannot index a list with unknown index")
        
        self._collection.insert(idx,value)
        _ = self.tracer.create_proxy("call_function", list.insert, (self,index,value),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)
       
    def remove(self, value):
        val = map_aggregate(value, lambda a: _get_concrete_val(a))
        if _find_proxy(val) is not None:
            raise ProxyDataError("cannot index a list with unknown index")
        
        self._collection.remove(val)
        _ = self.tracer.create_proxy("call_function", list.remove, (self,value),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)
    
    def pop(self, index=-1):
        idx = map_aggregate(index, lambda a: _get_concrete_val(a))
        if _find_proxy(idx) is not None:
            raise ProxyDataError("cannot index a list with unknown index")
        
        value = self._collection.pop(idx)
        return self.tracer.create_proxy("call_function", list.pop, (self,index),{},data=value, proxy_type=ProxyType.EAGER_PROXY)
        
    def clear(self):
        self._collection.clear()
        _ = self.tracer.create_proxy("call_function", list.clear, (self,),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)

    def index(self, value, start=0, end=None):
        val,s,e = map_aggregate((val,s,e), lambda a: _get_concrete_val(a))
        if _find_proxy((val,s,e)) is not None:
            raise ProxyDataError("cannot index a list with unknown index")
        value = self._collection.index(val,s,e)
        return self.tracer.create_proxy("call_function", list.index, (self,value, start, end),{},data=value, proxy_type=ProxyType.EAGER_PROXY)
        
    def count(self, value):
        val,s,e = map_aggregate((val,s,e), lambda a: _get_concrete_val(a))
        if _find_proxy((val,s,e)) is not None:
            raise ProxyDataError("cannot index a list with unknown index")
        
        return self.tracer.create_proxy("call_function", list.count, (self,value),{},data=self._collection.count(value), proxy_type=ProxyType.EAGER_PROXY)

    def sort(self, key=None, reverse=False):
        k,rev = map_aggregate((key,reverse), lambda a: _get_concrete_val(a))
        if _find_proxy((k,rev)) is not None:
            raise ProxyDataError("cannot index a list with unknown index")
        self._collection.sort(key=k, reverse=rev)
        _ = self.tracer.create_proxy("call_function", list.sort, (self,),{"key": key, "reverse": reverse},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)

    def reverse(self):
        self._collection.reverse()
        _ = self.tracer.create_proxy("call_function", list.reverse, (self,),{},data=None, proxy_type=ProxyType.UNDEFINED_PROXY)

class EnumerateProxy(IteratorProxy):
    def __init__(self, node, tracer, iterable_proxy, start=0):
        assert isinstance(iterable_proxy, ITERABLE_PROXIES)
        super().__init__(node, tracer, data=(iterable_proxy,start))
        iterable,s = _get_concrete_val(iterable_proxy,start)
        if _find_proxy((iterable,s)) is not None:
            raise ProxyDataError("cannot create an enumerate object with unknown iterable/start value.")
        self._collection = enumerate(iterable,s)

    def __repr__(self):
        return f"EnumerateProxy({self.node.name})"

class ZipProxy(IteratorProxy):
    def __init__(self, node, tracer, *iterable_proxies):
        for iterable_proxy in iterable_proxies:
            if isinstance(iterable_proxy, Proxy):
                assert isinstance(iterable_proxy, ITERABLE_PROXIES)
        super().__init__(node, tracer,data=iterable_proxies)
        iterables = map_aggregate(iterable_proxies, lambda a:_get_concrete_val(a))
        self._collection = zip(*iterables)
    
    def __repr__(self):
        return f"ZipProxy({self.node.name})"

class RangeProxy(Proxy):
    def __init__(self, node, tracer, start, stop=None, step=None):
        for arg in (start,stop,step):
            if isinstance(arg, Proxy):
                assert isinstance(arg, ScalarProxy), f"invalid input of type({type(arg)}) when calling range"
        super().__init__(node, tracer,data=(start,stop,step))

    def __getitem__(self, index):
        return self.tracer.create_proxy("call_function", range.__getitem__, (self,index),{},data=0, proxy_type=ProxyType.SCALAR_PROXY) 
    
    def __iter__(self):
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
        return (self[i] for i in range(inst.argval))  # type: ignore[index]
    
    def __repr__(self):
        return f"RangeProxy({self.node.name})"

@compatibility(is_backward_compatible=False)
class EagerProxy(Proxy):
    def __init__(self, node: Node, tracer: TracerBase, concr_val: Any):
        assert not isinstance(concr_val, Proxy), f"EagerProxies cannot be initialized with Proxies. Got an input of type {type(concr_val)}"
        super().__init__(node, tracer, concr_val)
        self._concrete_val = concr_val 

    def __getattr__(self, k) -> "EagerAttribute":
        try:
            attr = getattr(self._concrete_val,k)
            return EagerAttribute(self, getattr(type(self._concrete_val),k))
        except AttributeError:
            raise AttributeError(f"{type(self._concrete_val)} has no attribute '{k}'")

    def __call__(self, *args, **kwargs) -> "Proxy":
        return self.tracer.create_proxy(
            "call_method", "__call__", (self,) + args, kwargs, data=self.data
        )

    def __bool__(self):
        return bool(self._concrete_val)

    def __iter__(self) -> Iterable["EagerProxy"]:
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
            res = []
            for idx,item in enumerate(self._concrete_val):
                proxy = self.tracer.create_proxy("call_function", operator.getitem, (self,idx),{},data=item, proxy_type=ProxyType.EAGER_PROXY) 
                res.append(proxy)
            return iter(res)
        elif inst.opname == "DICT_UPDATE":
            return iter(self._concrete_val)
        elif inst.opname == "EXTENDED_ARG":
            while inst_list[inst_idx].opname == "EXTENDED_ARG":
                inst_idx += 1
            inst = inst_list[inst_idx]
        
        if inst.opname == 'UNPACK_EX':
            before_starred = inst.argval & 0xFF  # Lower 8 bits
            after_starred = inst.argval >> 8  # Upper 8 bits

            res_before = []
            for idx in range(before_starred):
                proxy = self.tracer.create_proxy("call_function", operator.getitem, (self,idx),{},data=self._concrete_val[idx], proxy_type=ProxyType.EAGER_PROXY) 
                res_before.append(proxy)

            res_after = []
            for idx in range(len(self._concrete_val) - after_starred, len(self._concrete_val)):
                proxy = self.tracer.create_proxy("call_function", operator.getitem, (self,idx),{},data=self._concrete_val[idx], proxy_type=ProxyType.EAGER_PROXY) 
                res_after.append(proxy)

            res_starred = self._concrete_val[before_starred:len(self._concrete_val)-after_starred]
            res_starred = [self.tracer.create_proxy("call_function", operator.getitem, (self,slice(before_starred,self.__len__()-after_starred)),{},data=res_starred, proxy_type=ProxyType.EAGER_PROXY)]
            
            res_before.append(res_starred)
            return iter(res_before + res_after)
    
        return self.tracer.create_proxy("call_function", self._concrete_val.__iter__, (self,),{},data=iter(self._concrete_val), proxy_type=ProxyType.EAGER_PROXY) 
    
    def __len__(self) -> 'EagerProxy':
        return self.tracer.create_proxy("call_function", self._concrete_val.__len__, (self,),{},data=len(self._concrete_val), proxy_type=ProxyType.EAGER_PROXY) 

    def len(self) -> 'EagerProxy':
        return self.__len__()
    
    def __repr__(self) -> str:
        return f"EagerProxy({self.node.name})"

class EagerAttribute(EagerProxy):
    @compatibility(is_backward_compatible=True)
    def __init__(self, root: EagerProxy, attr: Callable):
        assert isinstance(root, EagerProxy)
        assert callable(attr)
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Optional[Node] = None

    def __getattr__(self, k):
        return getattr(self.attr, k) 
    
    def __repr__(self):
        return f"EagerAttribute({self.attr.__name__})"
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
        nargs = (self.root,) + args
        if hasattr(self.attr, '__objclass__') and self.attr.__objclass__ in [list,tuple,set,dict]:
            concr_args = map_aggregate(nargs, lambda a: _get_concrete_val(a))
            concr_kwargs = map_aggregate(kwargs, lambda a: _get_concrete_val(a))
            val = self.attr(*concr_args, **concr_kwargs)
            return self.tracer.create_proxy(
                "call_function", self.attr, nargs, kwargs, data=val, proxy_type=ProxyType.EAGER_PROXY,
            )
        return self.attr(*nargs, **kwargs)

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

# define dunder methods for the EagerProxy/Iterable Proxy classes 
ITERABLE_PROXIES = (ListProxy, DictProxy, TupleProxy, SetProxy, EnumerateProxy, ZipProxy, EagerProxy)

for cls in ITERABLE_PROXIES:   
    for method in magic_methods:

        def _scope(method):
            def impl(*args, **kwargs):
                nargs = map_aggregate(args, lambda a: _get_concrete_val(a))
                nkwargs = map_aggregate(kwargs, lambda a: _get_concrete_val(a))
                tracer = args[0].tracer
                data = _get_concrete_val(args[0])
                method_ = f"__{method.strip('_')}__"
                try:
                    target = getattr(type(data), method_)
                except AttributeError:
                    target = getattr(operator, method_)
                concrete_result = target(*nargs, **nkwargs)
                return tracer.create_proxy(
                    "call_function", target, args, kwargs, data=concrete_result, proxy_type=ProxyType.EAGER_PROXY
                )

            impl.__name__ = method
            as_magic = f'__{method.strip("_")}__'
            setattr(cls, as_magic, impl) 

        _scope(method)


    def _define_reflectable(orig_method_name):
        def impl(self, rhs):
            rhs_val = map_aggregate(rhs,lambda a: _get_concrete_val(a))
            data = _get_concrete_val(self)
            method_ = f'__{orig_method_name.strip("_")}__'
            try:
                target = getattr(type(data), method_)
            except AttributeError:
                    target = getattr(operator, method_)
            concrete_result = target(rhs_val, data)
            return self.tracer.create_proxy(
                "call_function", target, (rhs, self), {}, data=concrete_result, proxy_type=ProxyType.EAGER_PROXY
            )

        method_name = f'__r{orig_method_name.strip("_")}__'
        impl.__name__ = method_name
        impl.__qualname__ = method_name
        setattr(cls, method_name, impl)


    for orig_method_name in reflectable_magic_methods:
        _define_reflectable(orig_method_name)

    for method in inplace_methods:
        def _scope(method):
            def impl(*args, **kwargs):
                nargs = map_aggregate(args, lambda a: _get_concrete_val(a))
                nkwargs = map_aggregate(kwargs, lambda a: _get_concrete_val(a))
                tracer = args[0].tracer
                data = _get_concrete_val(args[0]) 
                method_ = f"__{method.strip('_')}__"
                try:
                    target = getattr(type(data), method_)
                except AttributeError:
                    target = getattr(operator, method_)
                concrete_result = target(*nargs, **nkwargs)
                return tracer.create_proxy(
                    "call_function", target, args, kwargs, data=concrete_result, proxy_type=ProxyType.EAGER_PROXY
                )

            impl.__name__ = method
            as_magic = f'__{method.strip("_")}__'
            setattr(cls, as_magic, impl)

        _scope(method)

# define dunder methods for the Proxy class     
for method in magic_methods:

    def _scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            data = args[0]._meta_tensor
            target = getattr(operator, method)
            return tracer.create_proxy(
                "call_function", target, args, kwargs, data=data, proxy_type=_get_proxy_type(data)
            )

        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(Proxy, as_magic, impl) 

    _scope(method)


def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        target = getattr(operator, orig_method_name)
        data = self._meta_tensor
        return self.tracer.create_proxy(
            "call_function", target, (rhs, self), {}, data=data, proxy_type=_get_proxy_type(data)
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
            data = args[0]._meta_tensor 
            method_ = f"__{method.strip('_')}__"
            try:
                target = getattr(data, method_)
            except AttributeError:
                target = getattr(operator, method_)
            return tracer.create_proxy(
                "call_function", target, args, kwargs, data=data, proxy_type=_get_proxy_type(data)
            )

        impl.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(Proxy, as_magic, impl)

    _scope(method)

# define dunder methods for the NativeProxy class     
for method in magic_methods:

    def _scope(method):
        def impl(*args, **kwargs):
            native_method =  getattr(type(args[0]._native_data), method)
            return native_method(*args,**kwargs)

        impl.__name__ = method
        setattr(NativeProxy, method, impl) 
    method = f'__{method.strip("_")}__'
    _scope(method)


def _define_reflectable(orig_method_name):
    method_name = f'__r{orig_method_name.strip("_")}__'

    def impl(self, rhs):
        native_reflectable = getattr(type(self._native_data), orig_method_name)
        return native_reflectable(self, rhs)

    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(NativeProxy, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)

for method in inplace_methods:

    def _scope(method):
        def impl(*args, **kwargs):
            native_inp_method = getattr(type(args[0]._native_data), method)
            return native_inp_method(*args, **kwargs)

        impl.__name__ = method
        setattr(NativeProxy, method, impl)
    method = f'__{method.strip("_")}__'
    _scope(method)

def _get_concrete_val(obj):
    if isinstance(obj, EagerProxy):
        return obj._concrete_val
    elif isinstance(obj, (ListProxy,SetProxy,TupleProxy,DictProxy)):
        return obj._collection  
    else:
        return obj

def _find_proxy(*objects_to_search):
    """
    Recursively search a data structure for a Proxy() and return it,
    return None if not found.
    """
    proxy = None

    def find_proxy(x):
        nonlocal proxy
        if isinstance(x, Proxy) and proxy is None:
            proxy = x

    map_aggregate(objects_to_search, find_proxy)
    return proxy

def _is_native_shape(val):
    return val.__class__.__name__ in ("Size", "TensorShape")

def _get_proxy_type(data):
    if isinstance(data, Proxy):
        assert data.__class__.__name__ in class_to_enum
        return class_to_enum[data.__class__.__name__]
    if ivy.is_array(data) or data is None:
        return ProxyType.NATIVE_PROXY
    elif ivy.is_native_dtype(data):
        return ProxyType.DTYPE_PROXY
    elif _is_native_shape(data):
        return ProxyType.SHAPE_PROXY             
    elif ivy.isscalar(data):
        return ProxyType.SCALAR_PROXY
    elif isinstance(data, list):
        return ProxyType.LIST_PROXY
    elif isinstance(data, dict):
        return ProxyType.DICT_PROXY
    elif isinstance(data, tuple):
        return ProxyType.TUPLE_PROXY
    elif isinstance(data, set):
        return ProxyType.SET_PROXY
    elif isinstance(data, enumerate):
        return ProxyType.ENUMERATE_PROXY
    elif isinstance(data, range):
        return ProxyType.RANGE_PROXY
    elif isinstance(data, zip):
        return ProxyType.ZIP_PROXY
    else:
        return ProxyType.UNDEFINED_PROXY
    
FRONTEND_PROXIES = {
    "torch": Torch_FrontendProxy,
    "jax": JAX_FrontendProxy,
    "tensorflow": TF_FrontendProxy,
    "numpy": Numpy_FrontendProxy,
}

class_to_enum = {
    'IvyProxy': ProxyType.IVY_PROXY,
    'FrontendProxy': ProxyType.FRONTEND_PROXY,
    'NativeProxy': ProxyType.NATIVE_PROXY,
    'ShapeProxy': ProxyType.SHAPE_PROXY,
    'DtypeProxy': ProxyType.DTYPE_PROXY,
    'ScalarProxy': ProxyType.SCALAR_PROXY,
    'ListProxy': ProxyType.LIST_PROXY,
    'DictProxy': ProxyType.DICT_PROXY,
    'TupleProxy': ProxyType.TUPLE_PROXY,
    'SetProxy': ProxyType.SET_PROXY,
    'EnumerateProxy': ProxyType.ENUMERATE_PROXY,
    'ZipProxy': ProxyType.ZIP_PROXY,
    'RangeProxy': ProxyType.RANGE_PROXY,
    'NdarrayProxy': ProxyType.NDARRAY_PROXY,
    'UndefinedProxy': ProxyType.UNDEFINED_PROXY,
    'EagerProxy': ProxyType.EAGER_PROXY
}
