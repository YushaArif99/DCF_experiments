# global
import time
import inspect

# local
from .node import Node as Proxy_Node
from .immutable_collections import Constant
import ivy
from ivy.functional.ivy.gradients import _is_variable
from graph_compiler.visualisation import _get_argument_reprs, _get_output_reprs
from graph_compiler.wrapping import Node, Graph, FUNC_TO_PATH
import graph_compiler.globals as glob
import graph_compiler.tracked_var_proxy as tvp
from graph_compiler.param import (
    get_ids,
    get_types,
    get_shapes,
    get_var_flags,
    store_unique_id,
    _get_unique_id
)


class IvyNode(Node):
    from_tracked_var = False
    from_tracked_var_iterators = False
    from_iterator_chain = False
    is_inplace_w_side_effects = False
    inplace_fn = False
    prev_fn = False
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
        )
        + tracked_var_idxs
    )
    return tracked_idxs


def _record_parameters_info(args, to_ivy, stateful_idxs=[]):
    indexes = _find_parameter_indexes(args, stateful_idxs)
    parameters = ivy.multi_index_nest(args, indexes)
    ids = get_ids(parameters, to_ivy)
    types = get_types(parameters)
    var_flags = get_var_flags(parameters)
    shapes = get_shapes(parameters)
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


def log_ivy_fn(graph, fn, ret, args, kwargs, arg_stateful_idxs=[], kwarg_stateful_idxs=[], to_ivy=False):
    """
    This is the main mapping function that processes an fx Node and maps it
    to a corresponding IvyNode. Their is no computation performed here, but rather
    the function extracts the input(s) and output(s) along with some
    other metadata and uses these to grow the Ivy Graph
    """
    glob.tracing_stack.append(fn)

    target_framework = "ivy" if to_ivy else ivy.current_backend_str()

    # check if there are slices with Proxy Nodes inside
    arg_tracked_slices_idxs = ivy.nested_argwhere(args, is_tracked_slice)
    kwarg_tracked_slices_idxs = ivy.nested_argwhere(kwargs, is_tracked_slice)
    # (preprocessing): convert slices to slice-lists
    args = ivy.map_nest_at_indices(args, arg_tracked_slices_idxs, tvp.slice_to_list)
    kwargs = ivy.map_nest_at_indices(
        kwargs, kwarg_tracked_slices_idxs, tvp.slice_to_list
    )

    ###################
    #  node creation  #
    ###################

    # 1). create the node
    node = IvyNode()

    # 2). populate arg ids and other meta data
    (
        node.arg_tracked_idxs,
        arg_parameters,
        node.arg_param_ids,
        node.arg_param_types,
        node.arg_param_var_flags,
        _,
        node.iter_proxies,
        _,
    ) = _record_parameters_info(args, to_ivy=False, stateful_idxs=arg_stateful_idxs)

    # 3). populate kwarg ids and other meta data
    (
        node.kwarg_tracked_idxs,
        _,
        node.kwarg_param_ids,
        node.kwarg_param_types,
        node.kwarg_param_var_flags,
        _,
        _,
        node.dict_proxies,
    ) = _record_parameters_info(kwargs,to_ivy=False, stateful_idxs=kwarg_stateful_idxs)

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

    # 4). populate output ids and other meta data
    (
        node.output_tracked_idxs,
        _,
        node.output_param_ids,
        node.output_param_types,
        node.output_param_var_flags,
        node.output_param_shapes,
        _,
        _,
    ) = _record_parameters_info(ret_list,to_ivy=False, stateful_idxs=[])

    # return if there are no tracked outputs
    if not node.output_tracked_idxs:
        glob.tracing_stack.pop()
        return ret_list[0] if ret_listified else tuple_type(ret_list)

    # 5). handle duplicates
    # find all those outputs which have the same id as one of the inputs
    # we will have to clone those outputs to preserve uniqueness in the graph
    input_parameter_ids = node.arg_param_ids + node.kwarg_param_ids
    duplicates = list()
    for i, ret_id in enumerate(node.output_param_ids):
        if ret_id in input_parameter_ids:
            duplicates.append(i)

    # clone all repeated return parameters to give unique parameter ids in the graph
    duplicate_tracked_idxs = [node.output_tracked_idxs[i] for i in duplicates]
    ret_list = ivy.map_nest_at_indices(
        ret_list, duplicate_tracked_idxs, lambda x: store_unique_id(x, graph)
    )

    # get return param ids after cloning
    output_vals = ivy.multi_index_nest(ret_list, node.output_tracked_idxs)
    node.output_param_ids = [_get_unique_id(x) for x in output_vals]

    # 6). add the input(s) and output(s) ids to the dependent parameter dict
    # find any dependent parameters and add it to the global dict
    with_dependent_parameters = any(
        [x in glob.dependent_ids for x in input_parameter_ids]
    )
    if with_dependent_parameters:
        [glob.dependent_ids.add(id_) for id_ in node.output_param_ids]

    # (post-processing):convert slice-lists back to slices
    args = ivy.map_nest_at_indices(args, arg_tracked_slices_idxs, tvp.list_to_slice)
    kwargs = ivy.map_nest_at_indices(
        kwargs, kwarg_tracked_slices_idxs, tvp.list_to_slice
    )

    ###################################
    #  handling additional attributes #
    ###################################
     
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

    # 1). handle generators 
    gen_fns = glob.GENERATOR_FUNCTIONS[target_framework]
    if graph._with_numpy and target_framework != "numpy":
        gen_fns = gen_fns + glob.GENERATOR_FUNCTIONS["numpy"]

    node.is_generator = fn.__name__ in gen_fns
    node.is_generator_to_include = (
        node.is_generator and graph._include_generators
    )
    if node.is_generator_to_include or with_dependent_parameters:
        [glob.dependent_ids.add(id_) for id_ in node.output_param_ids]

    # 2). handle numpy functions
    #determine if the function is a numpy function
    fn_is_numpy = False
    if graph._with_numpy:
        if hasattr(fn, "__qualname__"):
            fn_is_numpy = "ndarray" in fn.__qualname__

        if not fn_is_numpy:
            # check for method
            if hasattr(fn, "__self__") and fn.__self__ is not None:
                fn_is_numpy = "numpy" in str(fn.__self__.__class__)
            # check for function
            elif hasattr(fn, "__module__") and fn.__module__ is not None:
                fn_is_numpy = "numpy" in fn.__module__ and "jax" not in fn.__module__

    #3). handle inplace updates
    # added so tensorflow inplace variable updates work properly (return is set
    # to first arg since this is the variable updated inplace)
    # provide return value for __setattr__ and similar functions
    inplace_fn = False
    if (
        fn.__name__
        in ["__setattr__", "setitem"]
        + glob.INPLACE_METHODS_WITHOUT_RET[target_framework]
        + glob.INPLACE_FUNCTIONS_WITHOUT_RET[target_framework]
    ) or (
        fn_is_numpy
        and fn.__name__
        in glob.INPLACE_METHODS_WITHOUT_RET["numpy"]
        + glob.INPLACE_FUNCTIONS_WITHOUT_RET["numpy"]
    ):
        inplace_fn = True

    # 4). store info about this node
    node.backend_fn = backend_fn
    if backend_fn in FUNC_TO_PATH:
            node.path = FUNC_TO_PATH[backend_fn]
    try:
        sig = inspect.signature(backend_fn)
        sig_keys = list(sig.parameters.keys())
    except ValueError:
        sig_keys = list()
    node.arg_n_kwarg_reprs = _get_argument_reprs(sig_keys, args, kwargs)
    node.output = ret_list
    node.remove_output_tuple = (
        isinstance(ret, tuple) and not isinstance(ret) and len(ret) == 1
    )
    node.output_reprs = _get_output_reprs(ret_list)

    node.timestamp = time.perf_counter()
    node.terminal = True
    node.with_tracked_slices = arg_tracked_slices_idxs + kwarg_tracked_slices_idxs
    node.inplace_fn = inplace_fn
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
        lambda x: None if _get_unique_id(x) in glob.dependent_ids else x,
        args,
        include_derived=True,
        shallow=False,
    )
    new_kwargs = ivy.nested_map(
        lambda x: None if _get_unique_id(x) in glob.dependent_ids else x,
        kwargs,
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
        to_mutable_list, new_args, include_derived=False, shallow=False
    )
    new_kwargs = ivy.nested_map(
        to_mutable_list, new_kwargs, include_derived=False, shallow=False
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
    glob.tracing_stack.pop()

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
                    subgraphs.append(tracer_to_ivy_graph(subg, fn=subg._fn))
            node.args = ivy.nested_map(
                lambda x: x.c if isinstance(x, Constant) else x, node.args, include_derived=True
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
                    subgraphs.append(tracer_to_ivy_graph(subg, fn=subg._fn))
            node.args = ivy.nested_map(
                lambda x: x.c if isinstance(x, Constant) else x, node.args, include_derived=True
            )
            while_loop_node = log_ivy_fn(
                graph,
                fn=node.target,
                ret=node,
                args=[],
                kwargs=node.args[2],
                to_ivy=to_ivy,
            )
            while_loop_node.subgraphs = subgraphs
            while_loop_node.test = node.meta["test"]
        else:
            node.args = ivy.nested_map(
                lambda x: x.c if isinstance(x, Constant) else x, node.args, include_derived=True
            )
            node.kwargs = ivy.nested_map(
                lambda x: x.c if isinstance(x, Constant) else x, node.kwargs, include_derived=True
            )
            arg_stateful_idxs = node.meta['arg_stateful_idxs'] if node.meta.get('arg_stateful_idxs') else []
            kwarg_stateful_idxs = node.meta['kwarg_stateful_idxs'] if node.meta.get('kwarg_stateful_idxs') else []
            _ = log_ivy_fn(
                graph,
                fn=node.target,
                ret=node,
                args=node.args,
                kwargs=node.kwargs,
                arg_stateful_idxs=arg_stateful_idxs,
                kwarg_stateful_idxs=kwarg_stateful_idxs,
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
    stateful=[],
):
    # dummy inputs to initialize the graph
    args = [ivy.native_array([]) for _ in graph_args]
    kwargs = {k: ivy.native_array([]) for k, v in graph_kwargs.items()}
    stateful = stateful if isinstance(stateful,list) else [stateful]
    ivy_graph = Graph(fn, to_ivy=to_ivy, with_numpy=with_numpy, stateful=stateful, *args, **kwargs)
    ivy_graph._args = graph_args
    ivy_graph._kwargs = graph_kwargs
    ivy_graph.var_args = var_args
    ivy_graph.var_kwargs = var_kwargs

    output_node = graph_outputs[0]
    #TODO(yusha): this is duplicate logic and should be removed eventually.
    #  But it needs to be added for now because our Graph generates 
    # ids based on native arrays. Wheras we need to generate them 
    # based on Proxy Nodes.
    (
        ivy_graph._arg_tracked_idxs,
        _,
        ivy_graph._arg_param_ids,
        ivy_graph._arg_param_types,
        ivy_graph._arg_param_var_flags,
        ivy_graph._arg_param_shapes,
        _,
        _,
        ) = _record_parameters_info(
            graph_args, to_ivy=False, stateful_idxs=[]
        )
    
    (
        ivy_graph._kwarg_tracked_idxs,
        _,
        ivy_graph._kwarg_param_ids,
        ivy_graph._kwarg_param_types,
        ivy_graph._kwarg_param_var_flags,
        ivy_graph._kwarg_param_shapes,
        _,
        _,
        ) = _record_parameters_info(
            graph_kwargs, to_ivy=False, stateful_idxs=[]
        )
    
    # add tracked inputs to graph
    ids = ivy_graph._arg_param_ids + ivy_graph._kwarg_param_ids + ivy_graph._stateful_param_ids
    types = ivy_graph._arg_param_types + ivy_graph._kwarg_param_types + ivy_graph._stateful_classes
    var_flags = (
        ivy_graph._arg_param_var_flags
        + ivy_graph._kwarg_param_var_flags
        + ivy_graph._stateful_param_var_flags
    )
    shapes = (
        ivy_graph._arg_param_shapes
        + ivy_graph._kwarg_param_shapes
        + ivy_graph._stateful_param_shapes
    )
    ivy_graph.add_parameters(ids, types, var_flags, shapes)
    
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


def tracer_to_ivy_graph(tracer_graph, fn=None, stateful=[], to_ivy=False, with_numpy=False,):
    graph_args, graph_functions, graph_outputs, graph_kwargs = [], [], [], {}
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
        elif node.op == "get_attr":
            # convert a get_attr node as a call_function node
            node.op = "call_function"
            node.args = (stateful, node.target)
            # this is simply a placeholder function. the actual operation
            # will be generated during sourcegen
            nested_attrgetter = lambda *_: None
            setattr(nested_attrgetter,'__name__', 'nested_attrgetter')
            node.target = nested_attrgetter

            node.meta['arg_stateful_idxs'] = [[0]]
            graph_functions.append(_create_ivy_fn(node))
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
        stateful=stateful
    )
