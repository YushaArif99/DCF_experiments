# global
import time
import inspect

# local
from .node import Node as Proxy_Node
from .immutable_collections import Constant
import ivy
from ivy.functional.ivy.gradients import _is_variable
from graph_compiler.visualisation import _get_argument_reprs, _get_output_reprs
from graph_compiler.wrapping import Node, Graph
import graph_compiler.globals as glob
import graph_compiler.tracked_var_proxy as tvp
from graph_compiler.helpers import (
    _get_unique_id,
    _get_shape,
    _clone_param,
)


class IvyNode(Node):
    from_tracked_var = False
    from_tracked_var_iterators = False
    from_iterator_chain = False
    is_inplace_fw_fn = False
    inplace_fn = False
    prev_fn = None
    is_method = False


def is_tracked_slice(s):
    """Method to check whether a slice contains Proxy Nodes inside."""
    return isinstance(s, slice) and any(
        [isinstance((getattr(s, p)), Proxy_Node) for p in ["start", "step", "stop"]]
    )


def _find_parameter_indexes(nest, tracked_var_idxs=[]):
    """Find the indexes of the parameters in the args and kwargs."""
    tracked_idxs = (
        ivy.nested_argwhere(
            nest,
            lambda x: isinstance(x, Proxy_Node),
            check_nests=True,
            to_ignore=tvp._to_ignore,
        )
        + tracked_var_idxs
    )
    return tracked_idxs


def _record_parameters_info(args, tracked_var_idxs=[]):
    indexes = _find_parameter_indexes(args, tracked_var_idxs)
    parameters = ivy.multi_index_nest(args, indexes)
    ids = [_get_unique_id(p) for p in parameters]
    types = [p.__class__ for p in parameters]
    var_flags = [
        _is_variable(p, exclusive=True, to_ignore=tvp._to_ignore) for p in parameters
    ]
    shapes = [_get_shape(p) for p in parameters]
    iter_proxies = [
        _id
        for (_id, p) in zip(ids, parameters)
        if (hasattr(p, "iter_proxy") and p.iter_proxy == True)
    ]
    dict_proxies = [
        _id
        for (_id, p) in zip(ids, parameters)
        if (hasattr(p, "dict_proxy") and p.dict_proxy == True)
    ]
    return (
        indexes,
        parameters,
        ids,
        types,
        var_flags,
        shapes,
        iter_proxies,
        dict_proxies,
    )


def log_ivy_fn(graph, fn, ret, args, kwargs, to_ivy=False):
    """
    This is the main mapping function that processes an fx Node and maps it 
    to a corresponding IvyNode. Their is no computation performed here, but rather
    the function extracts the input(s) and output(s) along with some 
    other metadata and uses these to grow the Ivy Graph
    """
    glob.logging_stack.append(fn)

    # check if there are slices with Proxy Nodes inside
    arg_tracked_slices_idxs = ivy.nested_argwhere(args, is_tracked_slice)
    kwarg_tracked_slices_idxs = ivy.nested_argwhere(kwargs, is_tracked_slice)
    # convert slices to slice-lists
    args = ivy.map_nest_at_indices(args, arg_tracked_slices_idxs, tvp.slice_to_list)
    kwargs = ivy.map_nest_at_indices(
        kwargs, kwarg_tracked_slices_idxs, tvp.slice_to_list
    )

    node = IvyNode()
    (
        node.arg_tracked_idxs,
        arg_parameters,
        node.arg_param_ids,
        node.arg_param_types,
        node.arg_param_var_flags,
        _,
        node.iter_proxies,
        _,
    ) = _record_parameters_info(args)

    (
        node.kwarg_tracked_idxs,
        _,
        node.kwarg_param_ids,
        node.kwarg_param_types,
        node.kwarg_param_var_flags,
        _,
        _,
        node.dict_proxies,
    ) = _record_parameters_info(kwargs)

    # set the backend function
    backend_fn = fn
    # in the case where a method of a native class is called (eg: x.mean()), the fn logged will be the name
    # of the function (i.e an str) and *not* the function itself. This is because we are working with Proxy inputs
    # and hence dont know the class 'x' belongs to. Thus we create a placeholder function here to bypass any errors
    # the tracing logic might throw.
    if isinstance(fn, str):
        backend_fn = lambda *_: None
        backend_fn.__name__ = fn
        node.is_method = True  # this attribute will later be used during source gen

    input_parameter_ids = node.arg_param_ids + node.kwarg_param_ids

    # convert slice-lists to slices
    args = ivy.map_nest_at_indices(args, arg_tracked_slices_idxs, tvp.list_to_slice)
    kwargs = ivy.map_nest_at_indices(
        kwargs, kwarg_tracked_slices_idxs, tvp.list_to_slice
    )

    # convert return to list
    ret_listified = False
    if (
        isinstance(ret, tuple)
        and not hasattr(ret, "is_tracked_proxy")
        and not hasattr(ret, "_fields")
    ):
        tuple_type = type(ret)
        ret_list = list(ret)
    else:
        ret_list = [ret]
        ret_listified = True

    (
        node.output_tracked_idxs,
        _,
        node.output_param_ids,
        node.output_param_types,
        node.output_param_var_flags,
        node.output_param_shapes,
        _,
        _,
    ) = _record_parameters_info(ret_list)

    # return if there are no tracked outputs
    if not node.output_tracked_idxs:
        glob.logging_stack.pop()
        return ret_list[0] if ret_listified else tuple_type(ret_list)

    # find all those outputs which have the same id as one of the inputs
    # we will have to clone those outputs to preserve uniqueness in the graph
    duplicates = list()
    for i, ret_id in enumerate(node.output_param_ids):
        if ret_id in input_parameter_ids:
            duplicates.append(i)

    # clone all repeated return parameters to give unique parameter ids in the graph
    duplicate_tracked_idxs = [node.output_tracked_idxs[i] for i in duplicates]
    ret_list = ivy.map_nest_at_indices(
        ret_list, duplicate_tracked_idxs, lambda x: _clone_param(x, graph)
    )

    # get return param ids after cloning
    output_vals = ivy.multi_index_nest(ret_list, node.output_tracked_idxs)
    node.output_param_ids = [_get_unique_id(x) for x in output_vals]

    # find any dependent parameters and add it to the global dict
    with_dependent_parameters = any(
        [x in glob.dependent_ids for x in input_parameter_ids]
    )
    if with_dependent_parameters:
        [glob.dependent_ids.add(id_) for id_ in node.output_param_ids]
    # store info about this node
    node.backend_fn = backend_fn
    try:
        sig = inspect.signature(backend_fn)
        sig_keys = list(sig.parameters.keys())
    except ValueError:
        sig_keys = list()
    node.arg_n_kwarg_reprs = _get_argument_reprs(sig_keys, args, kwargs)
    node.output = ret_list
    node.remove_output_tuple = (
        isinstance(ret, tuple) and not isinstance(ret, tvp._to_ignore) and len(ret) == 1
    )
    node.output_reprs = _get_output_reprs(ret_list)

    node.timestamp = time.perf_counter()
    node.terminal = True
    node.is_constant = len(input_parameter_ids) == 0
    node.with_tracked_slices = arg_tracked_slices_idxs + kwarg_tracked_slices_idxs

    fns_in = [
        graph._id_to_function[id_]
        for id_ in input_parameter_ids
        if id_ in graph._id_to_function
    ]

    # add this function as the outgoing function of the incoming functions
    if node.output_param_ids:
        for fn_in in fns_in:
            fn_in.terminal = False
            if node not in fn_in.fns_out:
                fn_in.fns_out.append(node)

    node.fns_in = fns_in
    node.fns_out = list()
    node.id_ = id(node)
    # remove parameters from args and kwargs
    new_args = ivy.nested_map(
        args,
        lambda x: None if _get_unique_id(x) in glob.dependent_ids else x,
        include_derived=True,
        shallow=False,
    )
    new_kwargs = ivy.nested_map(
        kwargs,
        lambda x: None if _get_unique_id(x) in glob.dependent_ids else x,
        include_derived=True,
        shallow=False,
    )
    # Todo: remove this hacky patch if possible
    # this fix is needed because Proxy Nodes are of type immutable lists.
    # We therefore convert these to mutable lists to allow inplace updates

    to_mutable_list = (
        lambda t: list(to_mutable_list(x) for x in t)
        if isinstance(t, (tuple, list))
        else t
    )
    new_args = ivy.nested_map(
        new_args, to_mutable_list, include_derived=False, shallow=False
    )
    new_kwargs = ivy.nested_map(
        new_kwargs, to_mutable_list, include_derived=False, shallow=False
    )
    node.args = new_args
    node.kwargs = new_kwargs

    # assign the same name to `node` as it is in the backend
    node.__repr__ = lambda: node.__name__
    node.__name__ = fn if isinstance(fn, str) else fn.__name__

    # add this function to the graph for each output id

    for id_ in node.output_param_ids:
        if id_ not in graph._id_to_function:
            graph.add_fn_to_dict(id_, node)

    # remove function from stack, now logging has occurred
    glob.logging_stack.pop()

    # return the node
    return node


def _create_ivy_fn(node, to_ivy=False):
    def call_ivy_fn(graph):
        if hasattr(node.target, "__name__") and node.target.__name__ == "if_else":
            subgraphs = []
            if isinstance(node.meta["pred"], (bool, type(None))):
                pred_g = lambda: bool(node.meta["pred"])
                subgraphs.append(pred_g)
            for subg in node.meta["subgraphs"]:
                if callable(subg):
                    subgraphs.append(subg)
                else:
                    subgraphs.append(tracer_to_ivy_graph(subg))
            node.args = ivy.nested_map(
                node.args, lambda x: x.c if isinstance(x, Constant) else x
            )
            if_else_node = log_ivy_fn(
                graph,
                fn=node.target,
                ret=node,
                args=list(node.args[3].values()),
                kwargs={},
                to_ivy=to_ivy,
            )
            if_else_node.subgraphs = subgraphs
            if_else_node.pred = node.meta["pred"]
        elif hasattr(node.target, "__name__") and node.target.__name__ == "while_loop":
            subgraphs = []
            if isinstance(node.meta["test"], (bool, type(None))):
                test_g = lambda: bool(node.meta["test"])
                subgraphs.append(test_g)
            for subg in node.meta["subgraphs"]:
                if callable(subg):
                    subgraphs.append(subg)
                else:
                    subgraphs.append(tracer_to_ivy_graph(subg))
            node.args = ivy.nested_map(
                node.args, lambda x: x.c if isinstance(x, Constant) else x
            )
            while_loop_node = log_ivy_fn(
                graph,
                fn=node.target,
                ret=node,
                args=list(node.args[2].values()),
                kwargs={},
                to_ivy=to_ivy,
            )
            while_loop_node.subgraphs = subgraphs
            while_loop_node.test = node.meta["test"]
        else:
            node.args = ivy.nested_map(
                node.args, lambda x: x.c if isinstance(x, Constant) else x
            )
            node.kwargs = ivy.nested_map(
                node.kwargs, lambda x: x.c if isinstance(x, Constant) else x
            )
            _ = log_ivy_fn(
                graph,
                fn=node.target,
                ret=node,
                args=node.args,
                kwargs=node.kwargs,
                to_ivy=to_ivy,
            )

    return call_ivy_fn


def _register_output(graph, ret):
    """Record information about the final output `ret` of the forward pass."""
    graph._output = ret
    graph._output_tracked_idxs = ivy.nested_argwhere(
        ret, lambda x: isinstance(x, Proxy_Node)
    )
    graph._output_param_ids = [
        _get_unique_id(x)
        for x in ivy.multi_index_nest(list(ret), graph._output_tracked_idxs)
    ]

    # find any inputs which were fed directly to the output, and update id_ and add identity function
    for i, id_ in enumerate(graph._output_param_ids):
        if id_ in graph._arg_param_ids + graph._kwarg_param_ids:

            def input_to_output(a, _):
                return a

            if id_ in graph._arg_param_ids:
                index = graph._arg_param_ids.index(id_)
                arg = ivy.index_nest(graph._args, graph._arg_tracked_idxs[index])
            else:
                index = graph._kwarg_param_ids.index(id_)
                arg = ivy.index_nest(graph._kwargs, graph._kwarg_tracked_idxs[index])

            ret = log_ivy_fn(graph, input_to_output, ret=arg, args=(arg,), kwargs={})

            graph._output_param_ids[i] = ret.output_param_ids[0]


def _create_graph(
    fn,
    graph_args,
    graph_kwargs,
    graph_functions,
    graph_outputs,
    var_args=False,
    var_kwargs=False,
    to_ivy=False,
    with_numpy=False,
):
    # dummy inputs to initialize the graph
    args = [ivy.native_array([]) for _ in graph_args]
    kwargs = {k: ivy.native_array([]) for k, v in graph_kwargs.items()}
    ivy_graph = Graph(fn, to_ivy=to_ivy, with_numpy=with_numpy, *args, **kwargs)
    ivy_graph._args = graph_args
    ivy_graph._kwargs = graph_kwargs
    ivy_graph.var_args = var_args
    ivy_graph.var_kwargs = var_kwargs

    output_node = graph_outputs[0]

    ivy_graph._arg_tracked_idxs = ivy.nested_argwhere(
        graph_args, lambda x: isinstance(x, Proxy_Node)
    )
    ivy_graph._arg_param_ids = [
        _get_unique_id(x)
        for x in ivy.multi_index_nest(list(graph_args), ivy_graph._arg_tracked_idxs)
    ]

    ivy_graph._kwarg_tracked_idxs = ivy.nested_argwhere(
        graph_kwargs, lambda x: isinstance(x, Proxy_Node)
    )
    ivy_graph._kwarg_param_ids = [
        _get_unique_id(x)
        for x in ivy.multi_index_nest(graph_kwargs, ivy_graph._kwarg_tracked_idxs)
    ]
    # initialize the dependent param id
    [
        glob.dependent_ids.add(id_)
        for id_ in ivy_graph._arg_param_ids + ivy_graph._kwarg_param_ids
    ]

    # populate the id_to_function dict
    [fn(ivy_graph) for fn in graph_functions]

    # register the output to handle any duplicate nodes
    ret = (
        output_node.args[0]
        if isinstance(output_node.args[0], tuple)
        else (output_node.args[0],)
    )
    _register_output(ivy_graph, ret)

    if ivy_graph._id_to_function:
        # connect the graph
        ivy_graph.connect()

    return ivy_graph


def tracer_to_ivy_graph(tracer_graph, fn=None, to_ivy=False, with_numpy=False):
    graph_args, graph_kwargs, graph_functions, graph_outputs = [], {}, [], []
    var_args, var_kwargs = False, False
    for node in tracer_graph.nodes:
        if node.op == "placeholder":
            if node.target == "*args":
                var_args = True
                graph_args.append(node)
            elif node.target == "**kwargs":
                var_kwargs = True
                graph_kwargs[node.name] = node
            elif node.args:
                graph_kwargs[node.name] = node
            else:
                graph_args.append(node)
        elif node.op in ("call_function", "call_method"):
            graph_functions.append(_create_ivy_fn(node))
        else:
            graph_outputs.append(node)

    return _create_graph(
        fn,
        graph_args,
        graph_kwargs,
        graph_functions,
        graph_outputs,
        var_args=var_args,
        var_kwargs=var_kwargs,
        to_ivy=to_ivy,
        with_numpy=with_numpy,
    )
