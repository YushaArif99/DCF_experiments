
# High Level Overview #

FX is a toolkit for developers to use to transform functions or ``nn.Module``
instances. It primarily consists of three main components: a **symbolic tracer,**
an **intermediate representation**, and **Python code generation**. 
However, we only make use of the symbolic tracing component in isolation to capture a DAG. The tracing mechanism makes use of the dynamic nature of Python to intercept call-sites for various entities ( native operations, Module invocations, and Arraylike method invocations). The **symbolic tracer** performs "symbolic execution" of the Python code. It feeds fake values, called **Proxies**, through the code. Operations on theses Proxies are recorded to form an intermediate representation (IR) called an fx graph. This graph consists of a doubly-linked list of Nodes. Although the Nodes are represented as a doubly-linked list, the use-def relationships form an acyclic graph (DAG) and can be traversed as such. Using this graph as the input, the **tracer_to_ivy_graph** function traverses the nodes of the fx graph, effectively applying a mapping function:

$$
F(n) \xrightarrow{\mathcal{T}} I(n)
$$

where:
- $F(n)$ represents a node in the FX graph.
- $I(n)$ represents the corresponding node in the Ivy Graph.
- $\mathcal{T}$ represents the transformation function (i.e [`log_ivy_fn`](https://github.com/unifyai/graph-compiler/blob/ivy_symbolic_tracing/ivy_fx/fx/graph_converter.py#L83)) that maps a node in the FX graph to a corresponding node in the Ivy Graph.



```python
import jax.numpy as jnp
import control_flow_experimental.ivy_fx.fx as fx 
import graph_compiler.globals as glob 

def foo(x,y, z=None,):
    res = torch.sin(x) + torch.cos(y)
    if z is not None: 
        res += torch.tan(z) 
    return res + [1,2,3]

glob.do_dummy_trace = True
ivy.set_backend('torch') 
x = ivy.array([1,2,3])
y = ivy.array([4,5,6])
z = ivy.array([7,8,9])

torch_tracer_g, torch_ivy_g = fx.symbolic_trace(foo, args=(x,y,z))
torch_ivy_g.reload_sourcecode(frontend='torch')
ivy.set_backend('jax', dynamic=True)
jax_tracer_g, jax_ivy_g = fx.symbolic_trace(torch_ivy_g, args=(x,y,z), frontend='torch', generate_source=True) 
```

The `foo` function above takes in two required arguments, `x` and `y`, and an optional argument `z`. The function calculates the sum of the sine of `x` and the cosine of `y`. It uses a conditional to check if the optional argument `z` is provided, and if so, the function also adds the tangent of `z` to the result. 
Running **symbolic tracing** twice on the above function yields the following Ivy Graph transpiled to JAX: 
```python
import jax
import jaxlib

def compiled_fn(*args, **kwargs):

    def pred_fn_139912365430080(*args, **kwargs):
        p139912365438816 = args[0]
        p139912365433152 = args[1]
        p139912365428976 = (p139912365433152 is not None)
        del p139912365433152
        p139912365433104 = bool(p139912365428976)
        del p139912365428976
        return bool(p139912365433104)

    def true_fn_139912365430080(*args, **kwargs):
        p139912365439152 = args[0]
        p139912365425760 = args[1]
        p139912365427632 = jax.numpy.tan(p139912365425760)
        p2079781071015729316 = p139912365425760
        del p139912365425760
        p139912365439872 = p139912365439152 + p139912365427632
        del p139912365439152, p139912365427632
        return p139912365439872, p2079781071015729316

    def false_fn_139912365430080(*args, **kwargs):
        p139912365437040 = args[0]
        p139912365431328 = args[1]
        p1264514798712448798 = p139912365437040
        del p139912365437040
        p6795413016519425161 = p139912365431328
        del p139912365431328
        return p1264514798712448798, p6795413016519425161
    p139912367216560 = args[0]
    p139912367216704 = args[1]
    p139912365428256 = args[2]
    c139912365725632 = kwargs['c139912365725632']
    p139912365426096 = jax.numpy.sin(p139912367216560)
    del p139912367216560
    p139912365437712 = jax.numpy.cos(p139912367216704)
    del p139912367216704
    p139912365428928 = p139912365426096 + p139912365437712
    del p139912365426096, p139912365437712
    with jax.disable_jit(): p139912365431472 = jax.lax.cond(lambda p139912365428928, p139912365428256: pred_fn_139912365430080(p139912365428928, p139912365428256),lambda p139912365428928, p139912365428256: true_fn_139912365430080(p139912365428928, p139912365428256), lambda p139912365428928, p139912365428256: false_fn_139912365430080(p139912365428928, p139912365428256), p139912365428928, p139912365428256)

    p139912365427920 = p139912365431472[0]
    del p139912365431472
    p139912365429888 = p139912365427920 + c139912365725632
    del p139912365427920
    return p139912365429888
```

# Internal Structure

## [Graph](https://github.com/unifyai/graph-compiler/blob/ivy_symbolic_tracing/ivy_fx/fx/graph.py#L671) ##
The `fx.Graph` is a core data structure in FX that represents the operations and their dependencies in a structured format. It consists of a List of `fx.Node` representing individual operations and their inputs and outputs. The Graph enables simple manipulation and analysis of the model structure, which is essential for implementing various transformations and optimizations.

## [Node](https://github.com/unifyai/graph-compiler/blob/ivy_symbolic_tracing/ivy_fx/fx/node.py#L116) ##
An `fx.Node` is a datastructure that represent individual operations within an `fx.Graph`, it maps to callsites such as operators, methods and modules. Each `fx.Node` keeps track of its inputs, the previous and next nodes, the stacktrace so you can map back the node to a line of code in your python file and some optional metadata stored in a `meta` dict.

# Tracing

## [Symbolic Tracer](https://github.com/unifyai/graph-compiler/blob/ivy_symbolic_tracing/ivy_fx/fx/_symbolic_trace.py#L571) ##

`Tracer` is the class that implements the symbolic tracing functionality of `fx.symbolic_trace`. A call to `symbolic_trace(m)` is equivalent to `Tracer().trace(m)`. Tracer can be subclassed to override various behaviors of the tracing process. The different behaviors that can be overridden are described in the docstrings of the methods on the class.

In the default implementation of `Tracer().trace`, the tracer first creates Proxy objects for all arguments in the `forward` function. (This happens in the call to `create_args_for_root`.) Next, the `forward` function is called with the new Proxy arguments. As the Proxies flow through the program, they record all the operations (function calls, method calls, and operators) that they touch into the growing FX Graph as Nodes.

## [Proxy](https://github.com/unifyai/graph-compiler/blob/ivy_symbolic_tracing/ivy_fx/fx/proxy.py#L274) ##

Proxy objects are Node wrappers used by the Tracer to record operations seen during symbolic tracing. The mechanism through which Proxy objects record computation is via the `_dummy_trace_func` wrapper. During execution, operations on Proxy are dispatched to the `_dummy_trace_func` wrapper, which records the operation in the Graph as a Node. The Node that was recorded in the Graph is then itself wrapped in a Proxy, facilitating further application of ops on that value.

# The FX IR Container

Tracing captures an intermediate representation (IR), which is represented as a doubly-linked list of Nodes.

Node is the data structure that represents individual operations within a Graph. For the most part, Nodes represent callsites to various entities, such as operators, methods, and Modules (some exceptions include Nodes that specify function inputs and outputs). Each Node has a function specified by its `op` property. The Node semantics for each value of `op` are as follows:

- `placeholder` represents a function input. The `name` attribute specifies the name this value will take on. `target` is similarly the name of the argument. `args` holds either: 1) nothing, or 2) a single argument denoting the default parameter of the function input. `kwargs` is don't-care. Placeholders correspond to the function parameters (e.g. `x`) in the graph printout.
- `get_attr` retrieves a parameter from the module hierarchy. `name` is similarly the name the result of the fetch is assigned to. `target` is the fully-qualified name of the parameter's position in the module hierarchy. `args` and `kwargs` are don't-care
- `call_function` applies a free function to some values. `name` is similarly the name of the value to assign to. `target` is the function to be applied. `args` and `kwargs` represent the arguments to the function, following the Python calling convention
- `call_module` applies a module in the module hierarchy's `forward()` method to given arguments. `name` is as previous. `target` is the fully-qualified name of the module in the module hierarchy to call. `args` and `kwargs` represent the arguments to invoke the module on, *including the self argument*.
- `call_method` calls a method on a value. `name` is as similar. `target` is the string name of the method to apply to the `self` argument. `args` and `kwargs` represent the arguments to invoke the module on, *including the self argument*
- `output` contains the output of the traced function in its `args[0]` attribute. This corresponds to the "return" statement in the Graph printout.

To facilitate easier analysis of data dependencies, Nodes have read-only properties `input_nodes` and `users`, which specify which Nodes in the Graph are used by this Node and which Nodes use this Node, respectively. 

## Sharp Bits: Symbolic Tracing + AST Transformations ##

Symbolic Tracing has limitations in that it can't deal with dynamic control flow and is limited to outputting a single graph at a time. To overcome the limitations, we basically employ AST Transformations to transform incompatible language constructs into a form that the tracer can handle. This includes (but is not limited to):
- transforming control flow such as if statements and while loops into `ivy.if_else` and `ivy.while_loop` respectively
- transforming python boolean operations such as `and`,`or` into an equivalent op (eg: [`fx.bool_and`](https://github.com/unifyai/graph-compiler/blob/ivy_symbolic_tracing/ivy_fx/fx/_symbolic_trace.py#L1799), [`fx.bool_or`](https://github.com/unifyai/graph-compiler/blob/ivy_symbolic_tracing/ivy_fx/fx/_symbolic_trace.py#L1798), etc.)
- transforming iterator/dictionary unpacking statements into a functional equivalent op (eg: [`fx.iter_proxy`](https://github.com/unifyai/graph-compiler/blob/ivy_symbolic_tracing/ivy_fx/fx/_symbolic_trace.py#L1779),[`fx.dict_proxy`](https://github.com/unifyai/graph-compiler/blob/ivy_symbolic_tracing/ivy_fx/fx/_symbolic_trace.py#L1788))


# WIP #

- ~~Adding support for symbolically tracing compiled Ivy Graphs (for the use case of transpilations).~~(DONE)
- Integrating symbolic tracing as a fallback mechanism in the `control_flow_v1` branch
- Rigorously testing symbolic tracing on the entire Ivy API to ensure compatibility.
- Adding support for tracing python for loops
- Finishing all the minor TODOs mentioned in the `_symbolic_tracer.py` file.

