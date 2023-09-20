import atexit
import builtins
import copy
import functools
import importlib.util
import inspect
import os
import shutil
import sys
import tempfile
import textwrap
import types
import warnings
from importlib.machinery import SourceFileLoader

import astor
import numpy as np

from ..utils import unique_name
from ..utils import gast

from .ast_utils import ast_to_source_code
from .static_analysis import StaticAnalysisVisitor
from .utils_helper import DYGRAPH_TO_STATIC_MODULE_PREFIX  # noqa: F401
from .utils_helper import IVY_MODULE_PREFIX  # noqa: F401
from .utils_helper import NodeVarType  # noqa: F401
from .utils_helper import _is_api_in_module_helper  # noqa: F401
from .utils_helper import index_in_list  # noqa: F401
from .utils_helper import is_api_in_module  # noqa: F401
from .utils_helper import is_dygraph_api  # noqa: F401
from .utils_helper import is_numpy_api  # noqa: F401;
from .utils_helper import is_ivy_api  # noqa: F401

import ivy 

__all__ = []

GET_ARGS_FUNC_PREFIX = 'get_args'
SET_ARGS_FUNC_PREFIX = 'set_args'
ALREADY_D2S = '__already_d2s'
ARGS_NAME = '__args'
# NOTE: Please use `getattr(ast_node, ORIGI_INFO)` instead of . operation to get the original information of ast node.
ORIGI_INFO = "Original information of source code for ast node."

DEL_TEMP_DIR = True  # A flag to avoid atexit.register more than once
FOR_ITER_INDEX_PREFIX = '__for_loop_var_index'
FOR_ITER_TUPLE_PREFIX = '__for_loop_iter_tuple'
FOR_ITER_TARGET_PREFIX = '__for_loop_iter_target'
FOR_ITER_ITERATOR_PREFIX = '__for_loop_iter_iterator'
FOR_ITER_TUPLE_INDEX_PREFIX = '__for_loop_iter_tuple_index'
FOR_ITER_VAR_LEN_PREFIX = '__for_loop_var_len'
FOR_ITER_VAR_NAME_PREFIX = '__for_loop_iter_var'
FOR_ITER_ZIP_TO_LIST_PREFIX = '__for_loop_iter_zip'

RE_PYNAME = '[a-zA-Z0-9_]+'
RE_PYMODULE = r'[a-zA-Z0-9_]+\.'

# Assign not support float64, use float32 value as magic number.
RETURN_NO_VALUE_VAR_NAME = "__no_value_return_var"
RETURN_NO_VALUE_MAGIC_NUM = 1.77113e27

PRED_FUNC_PREFIX = 'pred_fn'
TRUE_FUNC_PREFIX = 'true_fn'
FALSE_FUNC_PREFIX = 'false_fn'

WHILE_CONDITION_PREFIX = 'while_condition'
WHILE_BODY_PREFIX = 'while_body'
FOR_CONDITION_PREFIX = 'for_loop_condition'
FOR_BODY_PREFIX = 'for_loop_body'

GRAD_PREFIX = 'grad/'
GRAD_SUFFIX = '@GRAD'


class BaseNodeVisitor(gast.NodeVisitor):
    """
    Implement customized NodeVisitor inherited from gast.NodeVisitor.
    Ancestor nodes are traced to easily support more operations of currently
    visited node.
    """

    def __init__(self):
        self.ancestor_nodes = []

    def visit(self, node):
        """Visit a node."""
        self.ancestor_nodes.append(node)

        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()
        return ret


class UndefinedVar:
    def __init__(self, name):
        self.name = name

    def check(self):
        raise UnboundLocalError(
            "local variable '{}' should be created before using it."
        )


class Dygraph2StaticException(Exception):
    def __init__(self, message):
        super().__init__(message)


def saw(x):
    if isinstance(x, UndefinedVar):
        return x.check()
    else:
        return x


def parse_arg_and_kwargs(function):
    """
    Returns full argument names as list. e.g ['x', 'y', 'z']
    """
    fullargspec = inspect.getfullargspec(function)
    arg_names = fullargspec.args
    if arg_names and 'self' == arg_names[0]:
        arg_names = fullargspec.args[1:]

    # parse default kwargs
    default_kwargs = {}
    default_values = fullargspec.defaults
    if default_values:
        assert len(default_values) <= len(arg_names)
        default_kwarg_names = arg_names[-len(default_values) :]
        default_kwargs = dict(zip(default_kwarg_names, default_values))

    return arg_names, default_kwargs


def parse_varargs_name(function):
    """
    Returns varargs name string of function. e.g: 'input' from `foo(x, *input)`
    """
    fullargspec = inspect.getfullargspec(function)
    varargs = fullargspec.varargs
    return varargs


def type_name(v):
    return type(v).__name__


def make_hashable(x, error_msg=None):
    """
    Makes input `x` hashable.

    For some unhashable objects, such as `dict/list/set/np.ndarray`,applying hash function by using their values.
    """
    if isinstance(x, (tuple, list, set)):
        return tuple(map(make_hashable, x))

    try:
        hash(x)
    except TypeError:
        if isinstance(x, np.ndarray):
            # Note: `tostring()` will return the binary data from np.ndarray that
            # means different value will lead to different hash code.
            return hash(x.tostring())
        elif isinstance(x, dict):
            return tuple(map(make_hashable, x.values()))

        error_msg = error_msg or "Requires a hashable object."
        raise ValueError(error_msg + " But received type: %s" % type_name(x))

    return x


AS_NOT_INNER_FUNC_LIST = {""} # add ivy function's you'd like to transform


def is_ivy_func(func, ignore_white_list=True):
    """
    Return True if function is defined in ``ivy`` module.
    Skip to check APIs in white list if specifying ignore_white_list as True.
    """

    def in_white_list(module, func_name):
        if func_name is None:
            return False
        return (module.__name__ + '.' + func_name) in AS_NOT_INNER_FUNC_LIST

    try:
        if isinstance(func, functools.partial):
            func = func.func

        func_name = getattr(func, '__name__', None)
        if inspect.ismethod(func):
            func_name = func.__self__.__class__.__name__
            func = func.__func__
        elif hasattr(func, '__class__'):  
            func_name = func.__class__.__name__

        m = inspect.getmodule(func)
        flag = m is not None and m.__name__.startswith(IVY_MODULE_PREFIX)
        if ignore_white_list:
            flag = flag and not in_white_list(m, func_name)

        return flag
    except Exception:
        return False


def get_attribute_full_name(node):
    assert isinstance(
        node, gast.Attribute
    ), "Input non-Attribute node to get attribute full name"
    return astor.to_source(gast.gast_to_ast(node)).strip()


def generate_name_node(name_ids, ctx=gast.Load(), gen_tuple_if_single=False):
    """
    If name_ids is list or tuple or set with multiple strings, this function
    generates gast.Tuple of gast.Name.
    If the name_ids is single string or contains only 1 string, this function
    returns gast.Name if gen_tuple_if_single==False else returns gast.Tuple
    with only one gast.Name

    This function is used at several gast.Return statements.
    """
    if isinstance(name_ids, str):
        name_ids = [name_ids]
    if not isinstance(name_ids, (list, tuple, set)):
        raise TypeError(
            'name_ids must be list or tuple or set, but received %s'
            % type(type(name_ids))
        )

    def create_node_for_name(name):
        if '.' not in name:
            return gast.Name(
                id=name, ctx=ctx, annotation=None, type_comment=None
            )
        return gast.parse(name).body[0].value

    gast_names = [create_node_for_name(name_id) for name_id in name_ids]
    if len(gast_names) == 1 and not gen_tuple_if_single:
        name_node = gast_names[0]
    else:
        name_node = gast.Tuple(elts=gast_names, ctx=ctx)
    return name_node


def create_funcDef_node(nodes, name, input_args, return_name_ids):
    """
    Wrapper all statements of nodes into one ast.FunctionDef, which can be
    called by ast.Call.
    """
    nodes = copy.copy(nodes)
    # add return statement
    if return_name_ids:
        nodes.append(gast.Return(value=generate_name_node(return_name_ids)))
    elif PRED_FUNC_PREFIX not in name:
        nodes.append(gast.Return(value=None))
    func_def_node = gast.FunctionDef(
        name=name,
        args=input_args,
        body=nodes,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    return func_def_node


def create_assign_node(name, node):
    """
    Creates a `gast.Assign` node by given name_id as target and node as value.
    """
    targets = generate_name_node(name, ctx=gast.Store())
    assign_node = gast.Assign(targets=[targets], value=node)
    return targets, assign_node


def get_temp_dir():
    """
    Return @to_static temp directory.
    """
    dir_name = f"ivy/to_static_tmp/{os.getpid()}"
    temp_dir = os.path.join(os.getcwd(), '.cache', dir_name)#os.path.join(os.path.expanduser('~/.cache'), dir_name)
    is_windows = sys.platform.startswith('win')
    if is_windows:
        temp_dir = os.path.normpath(temp_dir)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    return temp_dir


def ast_to_func(ast_root, dyfunc, delete_on_exit=True):
    """
    Transform modified AST of decorated function into python callable object.
    TODO: If only decorate one of inner function instead of decorating the main
    function, the other inner functions are invisible for the decorated function.
    """

    def remove_if_exit(dir_path):
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    def func_prefix(func):
        pre_fix = func.__name__
        if hasattr(func, '__self__'):
            try:
                pre_fix = func.__self__.__class__.__name__ + '_' + func.__name__
            except:
                pass
        return "ivy_" + pre_fix + "__"

    source = ast_to_source_code(ast_root)
    source = _inject_import_statements() + source
    temp_dir = get_temp_dir()
    f = tempfile.NamedTemporaryFile(
        mode='w',
        prefix=func_prefix(dyfunc),
        suffix='.py',
        delete=False,
        dir=temp_dir,
        encoding='utf-8',
    )
    with f:
        module_name = os.path.basename(f.name[:-3])
        f.write(source)

    global DEL_TEMP_DIR
    if delete_on_exit and DEL_TEMP_DIR:
        # Clear temporary files in TEMP_DIR while exitting Python process
        atexit.register(remove_if_exit, dir_path=temp_dir)
        DEL_TEMP_DIR = False

    func_name = dyfunc.__name__
    loader = SourceFileLoader(module_name, f.name)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    # The 'forward' or 'another_forward' of 'TranslatedLayer' cannot be obtained
    # through 'func_name'. So set the special function name '__i_m_p_l__'.
    if hasattr(module, '__i_m_p_l__'):
        callable_func = module.__i_m_p_l__
        callable_func.__name__ = func_name
    elif hasattr(module, func_name):
        callable_func = getattr(module, func_name)
    else:
        raise ValueError(
            'Function: %s doesn\'t exist in the Module transformed from AST.'
            % func_name
        )
    # After transform dygraph function into callable_func saved in tmp file,
    # it lost the global variables from imported statements or defined in source file.
    # Recovers the necessary variables by `__globals__`.
    recover_globals_attribute(dyfunc, callable_func)

    return callable_func, f.name


def _inject_import_statements():
    backend_import = ivy.current_backend_str()
    import_statements = [
        "import ivy",
        "import control_flow_experimental.dygraph_ivy.dy2static as cfe",
        "import control_flow_experimental.ivy_fx.fx as fx",
        "from typing import *",
        "import numpy as np",
        "import warnings",
        "warnings.filterwarnings('ignore', category=DeprecationWarning)",
        "\n[fx.wrap(builtin) for builtin in ['min', 'max', 'sum', 'len']]\n\n"
    ]
    return '\n'.join(import_statements) + '\n'


def recover_globals_attribute(src_obj, dst_obj):
    attr_name = '__globals__'

    src_globals = getattr(src_obj, attr_name, {})
    dst_globals = getattr(dst_obj, attr_name, {})

    for k, v in src_globals.items():
        # ignore builtin attribute.
        if not (k.startswith('__') and k.endswith('__')):
            dst_globals[k] = v


def func_to_source_code(function, dedent=True):
    """
    Transforms function into raw string of source code.
    """
    if isinstance(function, functools.partial):
        function = function.func
    if not (inspect.isfunction(function) or inspect.ismethod(function)):
        raise TypeError(
            "The type of 'function' should be a function or method, but received {}.".format(
                type(function).__name__
            )
        )

    source_code_list, _ = inspect.getsourcelines(function)
    # Replace comments with blank lines so that error messages are not misplaced
    source_code_list = [
        line if not line.lstrip().startswith('#') else '\n'
        for line in source_code_list
    ]
    source_code = ''.join(source_code_list)

    if dedent:
        source_code = textwrap.dedent(source_code)

    return source_code


def is_candidate_node(node):
    """
    Nodes with specified type will be dependent on tensor.
    """
    is_compare_node = isinstance(
        node,
        (
            gast.Compare,
            gast.BoolOp,
            gast.UnaryOp,
            gast.For,
            gast.If,
            gast.While,
        ),
    )
    # TODO(Aurelius84): `.numpy()` may be an customized function,
    # and should consider a more elegant way to solve this problem.
    has_numpy_attr = ".numpy()" in ast_to_source_code(node)
    return is_compare_node or has_numpy_attr


def compare_with_none(node):
    """
    Whether the comparator of `gast.Compare` node is `None`.
    """
    if isinstance(node, gast.Compare):
        for child in [node.left, node.comparators]:
            # node.comparators is a list.
            if isinstance(child, list):
                child = child[0]
            if (isinstance(child, gast.Constant) and child.value is None) or (
                isinstance(child, gast.Name) and child.id == 'None'
            ):
                return True
    return False


class IsControlFlowVisitor(gast.NodeVisitor):
    """
    Judge whether the ast_node of control flow from Dygraph code dependent on a Tensor.
    `ast_node` can be gast.If, gast.For, gast.While, gast.If.test(gast.Compare, gast.BoolOp, gast.UnaryOp).

    If returns True,
    gast.If.test must meet at least one of the following requirements:
        1. involves at least one var whose type is Tensor.
        2. the Tensor var calls `.numpy()[]` interface or Tensor.shape is [1].
        3. involves Tensor.shape[i] and the shape[i] is unknown in compile time.
    gast.While must meet at least one of the requirements 1 to 5:
        4. has `break` statement.
        5. has `continue` statement.
    gast.For must meet at least one of the requirements 4 to 8:
        6. calls `range` function in `for` statement and the argument of range is Tensor.
        7. calls `enumerate` function in `for` statement and the argument of enumerate is Tensor.
        8. the iterable varaible in `for` statement is Tensor.
        TODO: Support non-range case

    The following examples should not be considered as control_flow_if:
        1. `if Tensor_var` or `if Tensor_var is None`
        2. if Tensor.shape[i] is determined with fixed value (not -1 or None)

    Note: pred in ConditionalBlock require variable, which means all vars should be Tensor
          or transformed into Tensor, like fill_constant(shape=[1], dtype='int32', value=Tensor.shape[i]).

    TODO: 1. need to deal with `tensor.shape[i]` which need to eval the data of shape[i],
             because reshape_op may be called before this statement.
    """

    def __init__(
        self, ast_node, static_analysis_visitor=None, node_var_type_map=None
    ):
        assert isinstance(
            ast_node, gast.AST
        ), "Type of input node should be gast.AST, but received %s." % type(
            ast_node
        )
        self.ast_root = ast_node
        if static_analysis_visitor is None:
            static_analysis_visitor = StaticAnalysisVisitor(ast_node)
        self.static_analysis_visitor = static_analysis_visitor
        self.node_to_wrapper_map = (
            self.static_analysis_visitor.get_node_to_wrapper_map()
        )
        self.node_var_type_map = node_var_type_map

        self.is_control_flow_num = 0
        self._compare_node_tenor_set = set()

    def transform(self):
        node = self.ast_root
        if isinstance(node, gast.If):
            self._visit_If(node)
        elif isinstance(node, gast.For):
            self._visit_For(node)
        elif isinstance(node, gast.While):
            self._visit_While(node)
        else:
            self.visit(node)
        return self.is_control_flow_num > 0

    def _visit_If(self, node):
        assert isinstance(node, gast.If)
        self.visit(node.test)

    def _visit_For(self, node):
        assert isinstance(node, gast.For)
        if isinstance(node.iter, gast.Call):
            # for in range(var[0]|var.numpy()[0]) or for in enumerate(var|var.numpy())
            if isinstance(node.iter.func, gast.Name):
                if (
                    node.iter.func.id == "range"
                    or node.iter.func.id == "enumerate"
                ):
                    for arg in node.iter.args:
                        self.visit(arg)
                else:
                    return
            # for in var.numpy()
            elif isinstance(node.iter.func, gast.Attribute):
                if node.iter.func.attr == 'numpy':
                    self._visit_Call(node.iter)
                else:
                    return
            else:
                return
        elif isinstance(node.iter, gast.Name):
            # for in var
            self.visit(node.iter)
        else:
            return

        for child_node in gast.walk(node):
            if isinstance(child_node, (gast.Continue, gast.Break)):
                self._visit_break_continue(child_node)
        return

    def _visit_While(self, node):
        assert isinstance(node, gast.While)
        test = node.test
        self.generic_visit(test)
        for child_node in gast.walk(node):
            if isinstance(child_node, (gast.Continue, gast.Break)):
                self._visit_break_continue(child_node)

    def _visit_break_continue(self, node):
        assert isinstance(node, (gast.Break, gast.Continue))
        wrapper_node = self.node_to_wrapper_map.get(node)
        if not wrapper_node:
            # Transformed node is not in node_to_wrapper_map
            return

        while wrapper_node.parent:
            parent_node = wrapper_node.parent.node
            if isinstance(parent_node, (gast.For, gast.While)):
                if parent_node is self.ast_root:
                    self.is_control_flow_num += 1
                    return
                else:
                    return

            wrapper_node = wrapper_node.parent

        return

    def visit_BoolOp(self, node):
        for i, child in enumerate(node.values):
            self.visit(child)
        return node

    def visit_Compare(self, node):
        pre_control_flow_num = self.is_control_flow_num
        if not compare_with_none(node):
            self.generic_visit(node)
            for child in gast.walk(node):
                if isinstance(child, gast.Subscript):
                    self._visit_Subscript(child)
        if self.is_control_flow_num > pre_control_flow_num:
            self._compare_node_tenor_set.add(node)
        return node

    def _visit_Subscript(self, node):
        self.generic_visit(node)
        if hasattr(node, 'value') and isinstance(node.value, gast.Call):
            self._visit_Call(node.value)
        return node

    def _visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if isinstance(node.func, gast.Attribute):
            attr_node = node.func
            if attr_node.attr == 'numpy':
                self.is_control_flow_num += 1

    def visit_Call(self, node):
        self._visit_Call(node)
        if is_ivy_api(node):
            self.is_control_flow_num += 1
        return node

    def visit_Name(self, node):
        if self._is_node_with_tensor(node, node.id):
            self.is_control_flow_num += 1
        return node

    def visit_Constant(self, node):
        if self._is_node_with_tensor(node, node.value):
            self.is_control_flow_num += 1
        return node

    def _is_node_with_tensor(self, node, name_id):
        # Look up the node_var_type_map by name_id.
        if self.node_var_type_map:
            if name_id and isinstance(name_id, str):
                var_type = self.node_var_type_map.get(name_id, None)
                if var_type and var_type & NodeVarType.TENSOR_TYPES:
                    return True
        # if not found, look up the node_to_wrapper_map by node.
        wrapper_node = self.node_to_wrapper_map.get(node, None)
        if wrapper_node is not None:
            if wrapper_node.node_var_type & NodeVarType.TENSOR_TYPES:
                return True

        return False

    def get_compare_nodes_with_tensor(self):
        return self._compare_node_tenor_set


# NOTE: inspect.unwrap() exits in PY3 but not in PY2.
def unwrap(func):
    """
    Returns the object wrapped by decorators.
    """

    def _is_wrapped(f):
        return hasattr(f, '__wrapped__')

    unwrapped_f = func
    while _is_wrapped(unwrapped_f):
        unwrapped_f = unwrapped_f.__wrapped__

    return unwrapped_f



class NameScope:
    def __init__(self):
        """
        A NameScope is a object which manager all the variable names.
        only FunctionDef and Controlflow node will have a namescope property.

        type can be "function" and "controlflow"

        we don't analyze the read only variable because they don't affect the analysis.
        """
        self.globals = set()
        self.nonlocals = set()
        self.args = set()
        self.father = None  # point to the nearest function name scope.
        self.w_vars = set()  # all qualified + normal names been stored
        self.created = set()  # useful for control flow compatibility
        # only valid in control_flow nodes
        # may be remove later.
        self.push_pop_vars = set()  # we call push and pop in the vars

    def set_father(self, father):
        self.father = father

    def existed_vars(self):
        """vars existing in current scope.
        they must not contain qualified names.
        """
        local_vars = self.w_vars - self.globals - self.nonlocals - self.args
        return set(filter(lambda x: '.' not in x, local_vars))

    def created_vars(self):
        return self.created

    def modified_vars(self):
        # may be globals / non-locals / args / qualified names and created_vars
        return self.w_vars

    def variadic_length_vars(self):
        """
        At present, we do not support global append, such as

        import numpy as np
        a = []
        def func():
            a.append() # global names `a`, we will raise a warning.
            p.append(a, 1) # global names `np`, we will raise a warning.
        """
        non_global_push_pop_names = []
        for var in self.push_pop_vars:
            if self._is_simple_name(var) and self.is_global_var(var):
                warnings.warn(
                    f"Find variable `{var}` defined in global scope"
                    f" and call `{var}.append() or {var}.pop()`"
                    f", which will be ignored and never be transfered into"
                    f" tensor array."
                )
            else:
                non_global_push_pop_names.append(var)
        return set(non_global_push_pop_names)

    def control_flow_vars(self):
        valid_names = self.w_vars
        tmp = (self.father.global_vars & valid_names,)
        return {"global": tmp, "nonlocal": self.w_vars - tmp}

    def _is_simple_name(self, name):
        if '.' in name or '[' in name:
            return False
        return True

    def is_global_var(self, name):
        """
        Return whether the name is a var created in global scope.
        Search from bottom to top. If it is not created or modified,
        it means global vars; otherwise, it means local vars.
        Only valid after FunctionNameLivenessAnalysis visitor.
        """
        assert self._is_simple_name(
            name
        ), "is_global_var accept a simple name, but get `{name}`."
        ancestor = self
        while ancestor is not None:
            if name in ancestor.globals:
                return True
            if name in (ancestor.nonlocals | ancestor.w_vars):
                return False
            ancestor = ancestor.father
        return True

    def is_local_var(self, name):
        return not self.is_global_var(name)

    def merge_from(self, name_scope):
        self.globals |= name_scope.globals
        self.nonlocals |= name_scope.nonlocals
        self.args |= name_scope.args
        self.w_vars |= name_scope.w_vars
        self.push_pop_vars |= name_scope.push_pop_vars


class FunctionNameLivenessAnalysis(gast.NodeVisitor):
    """analyze the liveness of a function.

    every variables stored in this scope will be collected,
    in addition with global/nonlocal information and
    push_pop information.

    1. global variable is stored in node.var_globals.
    2. nonlocal variable is stored in node.var_nonlocals.
    3. arguments is stored in node.var_args.
    4. if a variable's push and pop attribute is called,
       it will be collected in push_pop_vars. They are
       used for transformation to tensor_array.
       NOTE: push_pop_vars **may not** in w_vars.
       a.push(0) don't modify the variable a, but the content
       of a.

    For example:

    def func(*args, **kargs):
        a = 12
        global i,j
        nonlocal x,y
        print(a)
        i = k
        b = []
        c = [1,2,3]
        for m in range(10):
            q = 12
            b.push(1)
            c.pop()

    After this visitor we have:
    # node is the FunctionDef node with name: "func"
    node.ivy_scope = NameScope(
        globals = ['i', 'j'],
        nonlocals = ['x', 'y'],
        args = ['args', 'kargs'],
        wr_vars = ['a', 'i', 'q', 'm', 'c', 'b']
        push_pop_vars = ['b', 'c']
    )
    """

    def __init__(self, root_node):
        self.scope_node_stack = []  # controlflow, functiondef node
        self.visit(root_node)

    def _reset_name_scope(self, node):
        # always reset the node as empty namescope.
        node.ivy_scope = NameScope()

    def _get_name_scope(self, node):
        if not hasattr(node, "ivy_scope"):
            node.ivy_scope = NameScope()
        return node.ivy_scope

    def _current_name_scope(self):
        return self._get_name_scope(self.scope_node_stack[-1])

    def _father_name_scope(self):
        if len(self.scope_node_stack) == 1:
            return None
        return self._get_name_scope(self.scope_node_stack[-2])

    def _nearest_function_scope(self):
        if len(self.scope_node_stack) == 1:
            return None
        for node in self.scope_node_stack[-2::-1]:
            if isinstance(node, gast.FunctionDef):
                return self._get_name_scope(node)

    def visit_ListComp(self, node):
        """[ i for i in range(10) ]
        In this case, `i` will not created in FunctionScope.
        We don't collect `i` by not calling generic_visit.
        """
        pass

    def visit_DictComp(self, node):
        """the same as ListComp."""
        pass

    def visit_Name(self, node):
        self.generic_visit(node)
        write_context = (gast.Store, gast.AugStore, gast.Del)
        if isinstance(node.ctx, write_context):
            self._current_name_scope().w_vars.add(node.id)

    def visit_FunctionDef(self, node):
        def pre_func():
            self._current_name_scope().args |= set(
                self._get_argument_names(node)
            )

        def post_func():
            """NOTE: why we need merge w_vars and push_pop_vars here ?
            because we do ifelse_transformer after loop_transformer. Loops will changed into functioons. but we know this function will be called in if. so we add w_vars to father function scope.
            """
            control_flow_function_def = [
                WHILE_BODY_PREFIX,
                WHILE_BODY_PREFIX,
                FOR_CONDITION_PREFIX,
                FOR_BODY_PREFIX,
                TRUE_FUNC_PREFIX,
                FALSE_FUNC_PREFIX,
            ]

            def is_control_flow_def_node():
                for prefix in control_flow_function_def:
                    if node.name.startswith(prefix):
                        return True
                return False

            if self._father_name_scope() and is_control_flow_def_node():
                self._father_name_scope().w_vars |= (
                    self._current_name_scope().w_vars
                )
                self._father_name_scope().push_pop_vars |= (
                    self._current_name_scope().push_pop_vars
                )

        self._visit_scope_node(node, pre_func, post_func)

    def _visit_scope_node(self, node, pre_func, post_func):
        """scope node main visit logic.
        pre_func and post_func is callbacks
        """
        self._reset_name_scope(node)
        self.scope_node_stack.append(node)
        self._current_name_scope().set_father(self._nearest_function_scope())
        if pre_func:
            pre_func()
        self.generic_visit(node)
        if post_func:
            post_func()
        self.scope_node_stack.pop()

    def _visit_controlflow_node(self, node):
        def post_func():
            self._father_name_scope().merge_from(self._current_name_scope())
            self._nearest_function_scope().merge_from(
                self._current_name_scope()
            )
            self._current_name_scope().created = (
                self._nearest_function_scope().existed_vars()
                - node.before_created
            )
            # gather created vars into father and used in CreateUndefinedVarTransform
            self._nearest_function_scope().created |= (
                self._current_name_scope().created
            )

        def pre_func():
            node.before_created = self._nearest_function_scope().existed_vars()

        self._visit_scope_node(node, pre_func, post_func)

    def visit_For(self, node):
        self._visit_controlflow_node(node)

    def visit_While(self, node):
        self._visit_controlflow_node(node)

    def visit_If(self, node):
        self._visit_controlflow_node(node)

    def visit_Global(self, node):
        self._current_name_scope().globals |= set(node.names)

    def visit_Nonlocal(self, node):
        self._current_name_scope().nonlocals |= set(node.names)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        write_context = (gast.Store, gast.AugStore, gast.Del)
        if isinstance(node.ctx, write_context):
            name = ast_to_source_code(node).strip()
            self._current_name_scope().w_vars.add(name)

    def visit_Subscript(self, node):
        self.generic_visit(node)
        write_context = (gast.Store, gast.AugStore, gast.Del)
        if isinstance(node.ctx, write_context):
            while isinstance(node.value, gast.Subscript):
                node = node.value
            if isinstance(node.value, gast.Name):
                self._current_name_scope().w_vars.add(node.value.id)

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, gast.Attribute):
            return
        variadic_length_method = ['append', 'pop']
        if node.func.attr not in variadic_length_method:
            return
        # we don't treat push and pop as a write operator. such as a[i]=10 is not modify a.
        name = ast_to_source_code(node.func.value).strip()
        self._current_name_scope().push_pop_vars.add(name)

    def _get_argument_names(self, node):
        """get all arguments name in the functiondef node.
        this node is local to the function and shouldn't
        be created.
        """
        assert isinstance(
            node, gast.FunctionDef
        ), "Input node is not function define node"
        names = list(node.args.args)
        names.append(node.args.vararg)
        names.append(node.args.kwarg)
        names = [i.id for i in names if i is not None]
        return names


def create_get_args_node(names):
    """
    Create get_args function as follows:

        def get_args_0():
            nonlocal x, y
            return x, y
    """

    def empty_node():
        func_def = """
        def {func_name}():
            return
        """.format(
            func_name=unique_name.generate(GET_ARGS_FUNC_PREFIX)
        )
        return gast.parse(textwrap.dedent(func_def)).body[0]

    assert isinstance(names, (list, tuple))
    node = create_nonlocal_stmt_nodes(names)
    if not names:
        return empty_node()
    if node == []:
        nonlocal_vars = "\n"
    else:
        nonlocal_vars = ast_to_source_code(node[0])
    template = """
    def {func_name}():
        {nonlocal_vars}
        return {vars},
    """
    func_def = template.format(
        func_name=unique_name.generate(GET_ARGS_FUNC_PREFIX),
        nonlocal_vars=nonlocal_vars,
        vars=",".join(names),
    )
    return gast.parse(textwrap.dedent(func_def)).body[0]


def create_set_args_node(names):
    """
    Create set_args function as follows:

        def set_args_0(__args):
            nonlocal x, y
            x, y = __args
    """

    def empty_node():
        func_def = """
        def {func_name}({args}):
            pass
        """.format(
            func_name=unique_name.generate(SET_ARGS_FUNC_PREFIX), args=ARGS_NAME
        )
        return gast.parse(textwrap.dedent(func_def)).body[0]

    assert isinstance(names, (list, tuple))
    node = create_nonlocal_stmt_nodes(names)
    if not names:
        return empty_node()
    if node == []:
        nonlocal_vars = "\n"
    else:
        nonlocal_vars = ast_to_source_code(node[0])
    template = """
    def {func_name}({args}):
        {nonlocal_vars}
        {vars}, = {args}
    """
    func_def = template.format(
        func_name=unique_name.generate(SET_ARGS_FUNC_PREFIX),
        args=ARGS_NAME,
        nonlocal_vars=nonlocal_vars,
        vars=",".join(names),
    )
    return gast.parse(textwrap.dedent(func_def)).body[0]


def create_nonlocal_stmt_nodes(names):
    assert isinstance(names, (list, tuple))

    mapped = list(filter(lambda n: '.' not in n, names))
    mapped = list(filter(lambda n: '[' not in n, mapped))
    names = sorted(
        mapped, key=mapped.index
    )  # to keep the order, we can't use set() to unique
    if not names:
        return []
    func_code = "nonlocal {}".format(','.join(names))
    return [gast.parse(func_code).body[0]]
        
def create_dict_node(names):
    assert isinstance(names, (list, tuple))

    mapped = list(filter(lambda n: '.' not in n, names))
    mapped = list(filter(lambda n: '[' not in n, mapped))
    names = sorted(
        mapped, key=mapped.index
    )  # to keep the order, we can't use set() to unique
    if not names:
        return gast.Dict(keys=[], values=[])

    key_nodes = []
    value_nodes = []
    for var_name in names:
        key_node = gast.Constant(value=str(var_name), kind=None)
        value_node = gast.Name(
            id=str(var_name), ctx=gast.Load(), annotation=None, type_comment=None
        )
        key_nodes.append(key_node)
        value_nodes.append(value_node)

    dict_node = gast.Dict(keys=key_nodes, values=value_nodes)
    return dict_node


class GetterSetterHelper:
    """we have two classes of names in setter and getter function:
    w_vars(loop_vars) + push_pop_vars
    To simplify the setter logic in convert_while and convert_cond,
    we extract the helper class here.
    """

    def __init__(self, getter_func, setter_func, *name_lists):
        name_lists = ([] if x is None else x for x in name_lists)
        name_sets = (set(x) for x in name_lists)
        self._union = list(
            functools.reduce(lambda x, y: x | y, name_sets, set())
        )
        self._union.sort()
        self.getter = getter_func
        self.setter = setter_func
        self.name2id = {name: idx for idx, name in enumerate(self._union)}

    def union(self):
        return self._union

    def get(self, names):
        if names is None:
            names = []
        vars = self.getter()
        if vars is None:
            return ()
        for n in names:
            assert (
                n in self.name2id
            ), "the name `{}` not in name union set`{}`.".format(
                n, self.name2id.keys()
            )
        return tuple(vars[self.name2id[n]] for n in names)

    def set(self, names, values):
        if names is None:
            names = []
        if values is None:
            values = []
        vars = self.getter()
        if vars is None:
            return
        for n in names:
            assert (
                n in self.name2id
            ), "the name `{}` not in name union set`{}`.".format(
                n, self.name2id.keys()
            )
        vars = list(vars)
        indices = [self.name2id[n] for n in names]
        for i, v in zip(indices, values):
            vars[i] = v
        self.setter(vars)


def create_name_str(name_ids):
    """
    Return "('x', 'y')" for [x, y]
    """
    if not name_ids:
        return 'None'

    names_str = ["'%s'" % (name.replace("'", "\\'")) for name in name_ids]
    return "(%s, )" % ','.join(names_str)


def is_builtin(func, name=None):
    """predict whether a function is a builtin function with name={name}.
    if name == None, then any builtin function will return True
    """

    def name_judge():
        return name is None or func.__name__ == name

    if isinstance(func, types.BuiltinFunctionType) and name_judge():
        return True
    elif func in builtins.__dict__.values() and name_judge():
        return True
    else:
        return False

