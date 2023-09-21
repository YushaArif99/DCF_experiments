import inspect
from collections.abc import Sequence

from .helpers import gast

from .utils import ORIGI_INFO, unwrap

__all__ = []


class Location:
    """
    Location information of source code.
    """

    __slots__ = (
        "filepath",
        "lineno",
        "col_offset",
    )

    def __init__(self, filepath, lineno, col_offset=None):
        self.filepath = filepath
        self.lineno = lineno
        self.col_offset = col_offset

    def __str__(self):
        return "location: {}:{}:{}".format(
            self.filepath, self.lineno, self.col_offset
        )

    @property
    def line_location(self):
        return (self.filepath, self.lineno)


class OriginInfo:
    """
    Original information of source code.
    """

    __slots__ = (
        "location",
        "function_name",
        "source_code",
    )

    def __init__(self, location, function_name, source_code):
        self.location = location
        self.function_name = function_name
        self.source_code = source_code

    def __str__(self):
        return "{} \nsource_code: {}  in function {}\n  ".format(
            self.location, self.source_code, self.function_name
        )

    def formated_message(self):
        flag_for_origin_info = "(* user code *)"
        return '    File "{}", line {}, in {} {}\n\t{}'.format(
            self.location.filepath,
            self.location.lineno,
            self.function_name,
            flag_for_origin_info,
            self.source_code.lstrip(),
        )

    def as_frame(self):
        return (
            self.location.filepath,
            self.location.lineno,
            self.function_name,
            self.source_code.lstrip(),
        )


class OriginInfoAttacher(gast.NodeTransformer):
    """
    Attach original source information to AST node according corresponding function.
    """

    def __init__(self, root, func):
        self.root = root
        self.func = unwrap(func)
        self.filepath = inspect.getsourcefile(self.func)
        self.source_code = inspect.getsource(self.func)
        self.current_func = []

    def transform(self):
        source_lines, begin_lineno = inspect.getsourcelines(self.func)
        begin_line = source_lines[0]
        self.col_offset = len(begin_line) - len(begin_line.lstrip())
        self.source_lines = [line.strip("\n") for line in source_lines]
        self.lineno_offset = begin_lineno - 1
        self.visit(self.root)

    def visit(self, node):
        if isinstance(node, gast.FunctionDef):
            self.current_func.append(node)
        if hasattr(node, "lineno"):
            self._attach_origin_info(node)
        self.generic_visit(node)

        if isinstance(node, gast.FunctionDef):
            self.current_func.pop()
        return node

    def _attach_origin_info(self, node):
        assert isinstance(node, gast.AST)
        assert hasattr(node, "lineno")

        lineno = self._abs_lineno(node)
        col_offset = self._abs_col_offset(node)
        loc = Location(self.filepath, lineno, col_offset)
        func_name = self.current_func[-1].name
        code_line = self.source_lines[node.lineno - 1]

        origin_info = OriginInfo(loc, func_name, code_line)
        setattr(node, ORIGI_INFO, origin_info)

    def _abs_lineno(self, node):
        # NOTE:
        #   There are differences in ast_node.lineno between PY3.8+ and PY3.8-.
        #   If the first gast.FunctionDef has decorator, the lineno of gast.FunctionDef is differs.
        #       1. < PY3.8
        #           its lineno equals to the lineno of the first decorator node, which is not right.
        #       2. >= PY3.8
        #           its lineno is the actual lineno, which is right.

        return self.lineno_offset + node.lineno

    def _abs_col_offset(self, node):
        return self.col_offset + node.col_offset


global_origin_info_map = {}


def create_and_update_origin_info_map(
    transformed_node, static_func, is_global=True
):
    """
    Creates a original information map between transformed static function and original dygraph function.

    Args:
        transformed_node(gast.AST): The AST node of transformed dygraph function with attached source information of original dygraph function.
        static_func(Callable): The static function transformed by dygraph function corresponding to transformed_node.

    Returns:
        The original information map.
    """

    origin_info_map = {}
    static_source = inspect.getsource(static_func)
    static_node = gast.parse(static_source)
    static_node = attach_origin_info(static_node, static_func)

    for t_node, s_node in ast_walk(transformed_node, static_node):
        assert type(t_node) == type(
            s_node
        ), "The node types should be the same, but received type(t_node) is {}, and type(s_node) is {}.".format(
            type(t_node), type(s_node)
        )
        dygraph_info = getattr(t_node, ORIGI_INFO, None)
        static_info = getattr(s_node, ORIGI_INFO, None)

        if dygraph_info is None or static_info is None:
            continue
        static_loc = static_info.location.line_location
        exist_origin_info = origin_info_map.get(static_loc)

        if exist_origin_info is not None:
            if (
                exist_origin_info.location.lineno
                >= dygraph_info.location.lineno
            ):
                continue
            if (
                exist_origin_info.location.col_offset
                <= dygraph_info.location.col_offset
            ):
                continue

        origin_info_map[static_loc] = dygraph_info

    global_origin_info_map.update(origin_info_map)
    if is_global:
        return global_origin_info_map

    return origin_info_map


def attach_origin_info(ast_node, func):
    """
    Attach original source information to AST node according corresponding function.

    Args:
        ast_node(gast.AST): The AST node to attach original source information.
        func(Callable): The corresponding function of ast_node. Parse the original information from this function.

    Returns:
        An AST node attached original source information.
    """
    resolver = OriginInfoAttacher(ast_node, func)
    resolver.transform()
    return ast_node


def ast_walk(transformed_node, static_node):
    """
    Recursively yield all descendant nodes in the trees starting at transformed_node and static_node (including itself) in parallel.

    NOTE:
        Function ast.walk is not used because it yield all descendant nodes in no specified order.
    """

    def _as_list(x):
        if x is None:
            return []
        return list(x) if isinstance(x, Sequence) else [x]

    transformed_node_list = _as_list(transformed_node)
    static_node_list = _as_list(static_node)

    while transformed_node_list:
        assert len(transformed_node_list) == len(static_node_list)
        t_node = transformed_node_list.pop()
        s_node = static_node_list.pop()
        if type(t_node) != type(s_node):
            # NOTE(liym27):
            # Node types should be strictly required, but there is no strict distinction between gast.Load and gast.Param
            # in the ast transformation process.
            if isinstance(t_node, (gast.Load, gast.Param)) or isinstance(
                s_node, (gast.Load, gast.Param)
            ):
                continue

        assert type(t_node) == type(
            s_node
        ), "The node types should be the same, but received type(t_node) is {}, and type(s_node) is {}.".format(
            type(t_node), type(s_node)
        )

        yield t_node, s_node

        for field in t_node._fields:
            t_node_child = getattr(t_node, field)
            s_node_child = getattr(s_node, field)

            if isinstance(t_node_child, gast.AST):
                transformed_node_list.append(t_node_child)
                static_node_list.append(s_node_child)
            elif isinstance(t_node_child, (list, tuple)):
                assert len(t_node_child) == len(s_node_child)
                for d_item, s_item in zip(t_node_child, s_node_child):
                    if isinstance(d_item, gast.AST):
                        transformed_node_list.append(d_item)
                        static_node_list.append(s_item)


