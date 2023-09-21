import copy
from collections import defaultdict

from .helpers import unique_name
from control_flow_experimental.dy2static.utils import (
    FOR_ITER_INDEX_PREFIX,
    FOR_ITER_ITERATOR_PREFIX,
    FOR_ITER_TARGET_PREFIX,
    FOR_ITER_TUPLE_INDEX_PREFIX,
    FOR_ITER_TUPLE_PREFIX,
    FOR_ITER_VAR_LEN_PREFIX,
    FOR_ITER_VAR_NAME_PREFIX,
    FOR_ITER_ZIP_TO_LIST_PREFIX,
    FunctionNameLivenessAnalysis,
    GetterSetterHelper,
    ast_to_source_code,
    create_funcDef_node,
    create_get_args_node,
    create_name_str,
    create_nonlocal_stmt_nodes,
    create_dict_node,
    create_tuple_node,
    create_set_args_node,
)

# gast is a generic AST to represent Python2 and Python3's Abstract Syntax Tree(AST).
# It provides a compatibility layer between the AST of various Python versions,
# as produced by ast.parse from the standard ast module.
# See details in https://github.com/serge-sans-paille/gast/
from .helpers import gast

from .base_transformer import BaseTransformer
from .utils import FALSE_FUNC_PREFIX, TRUE_FUNC_PREFIX, PRED_FUNC_PREFIX

__all__ = []

GET_ARGS_FUNC_PREFIX = 'get_args'
SET_ARGS_FUNC_PREFIX = 'set_args'
ARGS_NAME = '__args'


class IfElseTransformer(BaseTransformer):
    """
    Transform if/else statement of Dygraph into Static Graph.
    """

    def __init__(self, root):
        self.root = root
        FunctionNameLivenessAnalysis(
            self.root
        )  # name analysis of current ast tree.

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)

    def visit_If(self, node):
        self.generic_visit(node)
        (   
            pred_func_node,
            true_func_node,
            false_func_node,
            cond_vars,
        ) = transform_if_else(node, self.root)

        new_node = create_convert_ifelse_node(
            cond_vars,
            pred_func_node,
            true_func_node,
            false_func_node,
        )

        return [
            pred_func_node,
            true_func_node,
            false_func_node,
        ] + [new_node]

    def visit_Call(self, node):
        # Remove `numpy()` statement, like `Tensor.numpy()[i]` -> `Tensor[i]`
        if isinstance(node.func, gast.Attribute):
            attribute = node.func
            if attribute.attr == 'numpy':
                node = attribute.value
        self.generic_visit(node)
        return node

    def visit_IfExp(self, node):
        """
        Transformation with `true_fn(x) if Tensor > 0 else false_fn(x)`
        """
        self.generic_visit(node)

        new_node = create_convert_ifelse_node(
            None, None, node.test, node.body, node.orelse, None, None, True
        )
        # Note: A blank line will be added separately if transform gast.Expr
        # into source code. Using gast.Expr.value instead to avoid syntax error
        # in python.
        if isinstance(new_node, gast.Expr):
            new_node = new_node.value

        return new_node


class NameVisitor(gast.NodeVisitor):
    def __init__(self, after_node=None, end_node=None):
        # The start node (exclusive) of the visitor
        self.after_node = after_node
        # The terminate node of the visitor.
        self.end_node = end_node
        # Dict to store the names and ctxs of vars.
        self.name_ids = defaultdict(list)
        # List of current visited nodes
        self.ancestor_nodes = []
        # True when in range (after_node, end_node).
        self._in_range = after_node is None
        self._candidate_ctxs = (gast.Store, gast.Load, gast.Param)
        self._def_func_names = set()

    def visit(self, node):
        """Visit a node."""
        if self.after_node is not None and node == self.after_node:
            self._in_range = True
            return
        if node == self.end_node:
            self._in_range = False
            return

        self.ancestor_nodes.append(node)
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        ret = visitor(node)
        self.ancestor_nodes.pop()

        return ret

    def visit_If(self, node):
        """
        For nested `if/else`, the created vars are not always visible for parent node.
        In addition, the vars created in `if.body` are not visible for `if.orelse`.

        Case 1:
            x = 1
            if m > 1:
                res = new_tensor
            res = res + 1   # Error, `res` is not visible here.

        Case 2:
            if x_tensor > 0:
                res = new_tensor
            else:
                res = res + 1   # Error, `res` is not visible here.

        In above two cases, we should consider to manage the scope of vars to parsing
        the arguments and returned vars correctly.
        """
        if not self._in_range or not self.end_node:
            self.generic_visit(node)
            return
        else:
            before_if_name_ids = copy.deepcopy(self.name_ids)
            body_name_ids = self._visit_child(node.body)
            # If traversal process stops early in `if.body`, return the currently seen name_ids.
            if not self._in_range:
                self._update_name_ids(before_if_name_ids)
            else:
                else_name_ids = self._visit_child(node.orelse)
                # If traversal process stops early in `if.orelse`, return the currently seen name_ids.
                if not self._in_range:
                    self._update_name_ids(before_if_name_ids)
                else:
                    # Blocks the vars in `if.body` and only inserts the vars both created in 'if/else' branch
                    # into name_ids.
                    new_name_ids = self._find_new_name_ids(
                        body_name_ids, else_name_ids
                    )
                    for new_name_id in new_name_ids:
                        before_if_name_ids[new_name_id].append(gast.Store())

                    self.name_ids = before_if_name_ids

    def visit_Attribute(self, node):
        if not self._in_range or not self._is_call_func_name_node(node):
            self.generic_visit(node)

    def visit_Name(self, node):
        if not self._in_range:
            self.generic_visit(node)
            return
        blacklist = {'True', 'False', 'None'}
        if node.id in blacklist:
            return
        if node.id in self._def_func_names:
            return
        if not self._is_call_func_name_node(node):
            if isinstance(node.ctx, self._candidate_ctxs):
                self.name_ids[node.id].append(node.ctx)

    def visit_Assign(self, node):
        if not self._in_range:
            self.generic_visit(node)
            return
        # Visit `value` firstly.
        node._fields = ('value', 'targets')
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # NOTE: We skip to visit names of get_args and set_args, because they contains
        # nonlocal statement such as 'nonlocal x, self' where 'self' should not be
        # parsed as returned value in contron flow.
        if (
            GET_ARGS_FUNC_PREFIX in node.name
            or SET_ARGS_FUNC_PREFIX in node.name
        ):
            return

        if not self._in_range:
            self.generic_visit(node)
            return
        self._def_func_names.add(node.name)
        if not self.end_node:
            self.generic_visit(node)
        else:
            before_name_ids = copy.deepcopy(self.name_ids)
            self.name_ids = defaultdict(list)
            self.generic_visit(node)

            if not self._in_range:
                self._update_name_ids(before_name_ids)
            else:
                self.name_ids = before_name_ids

    def _visit_child(self, node):
        self.name_ids = defaultdict(list)
        if isinstance(node, list):
            for item in node:
                if isinstance(item, gast.AST):
                    self.visit(item)
        elif isinstance(node, gast.AST):
            self.visit(node)

        return copy.deepcopy(self.name_ids)

    def _find_new_name_ids(self, body_name_ids, else_name_ids):
        def is_required_ctx(ctxs, required_ctx):
            for ctx in ctxs:
                if isinstance(ctx, required_ctx):
                    return True
            return False

        candidate_name_ids = set(body_name_ids.keys()) & set(
            else_name_ids.keys()
        )
        store_ctx = gast.Store
        new_name_ids = set()
        for name_id in candidate_name_ids:
            if is_required_ctx(
                body_name_ids[name_id], store_ctx
            ) and is_required_ctx(else_name_ids[name_id], store_ctx):
                new_name_ids.add(name_id)

        return new_name_ids

    def _is_call_func_name_node(self, node):
        white_func_names = {'append', 'extend'}
        if len(self.ancestor_nodes) > 1:
            assert self.ancestor_nodes[-1] == node
            parent_node = self.ancestor_nodes[-2]
            if isinstance(parent_node, gast.Call) and parent_node.func == node:
                # e.g: var_list.append(elem), var_list is also a name_id.
                should_skip = (
                    isinstance(node, gast.Attribute)
                    and node.attr in white_func_names
                )
                if not should_skip:
                    return True
        return False

    def _update_name_ids(self, new_name_ids):
        for name_id, ctxs in new_name_ids.items():
            self.name_ids[name_id] = ctxs + self.name_ids[name_id]


def _valid_nonlocal_names(return_name_ids, nonlocal_names):
    """
    All var in return_name_ids should be in nonlocal_names.
    Moreover, we will always put return_name_ids in front of nonlocal_names.

    For Example:

        return_name_ids: [x, y]
        nonlocal_names : [a, y, b, x]

    Return:
        nonlocal_names : [x, y, a, b]
    """
    assert isinstance(return_name_ids, list)
    for name in return_name_ids:
        if name not in nonlocal_names:
            raise ValueError(
                "Required returned var '{}' must be in 'nonlocal' statement '', but not found.".format(
                    name
                )
            )
        nonlocal_names.remove(name)

    return return_name_ids + nonlocal_names


def transform_if_else(node, root):
    """
    Transform ast.If into control flow statement of Ivy control flow op.
    """

    # TODO(liym27): Consider variable like `self.a` modified in if/else node.
    return_name_ids = sorted(node.ivy_scope.modified_vars())
    push_pop_ids = sorted(node.ivy_scope.variadic_length_vars())
    nonlocal_names = list(return_name_ids + push_pop_ids)
    nonlocal_names.sort()
    # NOTE: All var in return_name_ids should be in nonlocal_names.
    nonlocal_names = _valid_nonlocal_names(return_name_ids, nonlocal_names)

    # TODO(dev): Need a better way to deal this.
    # LoopTransformer will create some special vars, which is not visiable by users. so we can sure it's safe to remove them.
    filter_names = [
        ARGS_NAME,
        FOR_ITER_INDEX_PREFIX,
        FOR_ITER_TUPLE_PREFIX,
        FOR_ITER_TARGET_PREFIX,
        FOR_ITER_ITERATOR_PREFIX,
        FOR_ITER_TUPLE_INDEX_PREFIX,
        FOR_ITER_VAR_LEN_PREFIX,
        FOR_ITER_VAR_NAME_PREFIX,
        FOR_ITER_ZIP_TO_LIST_PREFIX,
    ]

    def remove_if(x):
        for name in filter_names:
            if x.startswith(name):
                return False
        return True

    nonlocal_names = list(filter(remove_if, nonlocal_names))
    cond_vars = nonlocal_names

    arg_node = gast.arguments(
        args=[gast.Name(id=name, ctx=gast.Param(), annotation=None, type_comment=None) for name in nonlocal_names],
        posonlyargs=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=None,
        kwarg=None,
        defaults=[],
    )

    
    pred_func_node = create_funcDef_node(
        [gast.Return(value=node.test)],
        name=unique_name.generate(PRED_FUNC_PREFIX),
        input_args=arg_node,
        return_name_ids=[],
    )
    true_func_node = create_funcDef_node(
        node.body,
        name=unique_name.generate(TRUE_FUNC_PREFIX),
        input_args=arg_node,
        return_name_ids=cond_vars,
    )
    false_func_node = create_funcDef_node(
        node.orelse,
        name=unique_name.generate(FALSE_FUNC_PREFIX),
        input_args=arg_node,
        return_name_ids=cond_vars,
    )

    return (
        pred_func_node,
        true_func_node,
        false_func_node,
        cond_vars,
    )


def create_convert_ifelse_node(
    cond_vars,
    pred_func,
    true_func,
    false_func,
    is_if_expr=False,
):
    """
    Create `control_flow_experimental.dygraph.dygraph_to_static.convert_ifelse(
            pred, true_fn, false_fn, dict_vars)`
    to replace original `python if/else` statement.
    """
    if is_if_expr:
        true_func_source = f"lambda : {ast_to_source_code(true_func)}"
        false_func_source = f"lambda : {ast_to_source_code(false_func)}"
    else:
        pred_func_source = pred_func.name
        true_func_source = true_func.name
        false_func_source = false_func.name

    convert_ifelse_fn = gast.parse(
        '{tuple_vars} = ivy.if_else('
        '{pred}, {true_fn}, {false_fn}, vars={dict_vars})'.format(
            tuple_vars =ast_to_source_code(create_tuple_node(cond_vars)).strip('\n'),
            pred=pred_func_source,
            true_fn=true_func_source,
            false_fn=false_func_source,
            dict_vars=ast_to_source_code(create_dict_node(cond_vars)),
        )
    ).body[0]

    return convert_ifelse_fn