from .helpers import gast

from .base_transformer import BaseTransformer
from .utils import ast_to_source_code

__all__ = []

cmpop_type_to_str = {
    gast.Eq: "==",
    gast.NotEq: "!=",
    gast.Lt: "<",
    gast.LtE: "<=",
    gast.Gt: ">",
    gast.GtE: ">=",
    gast.Is: "is",
    gast.IsNot: "is not",
    gast.In: "in",
    gast.NotIn: "not in",
}


def cmpop_node_to_str(node):
    return cmpop_type_to_str[type(node)]


class LogicalTransformer(BaseTransformer):
    """
    Transform python boolean op into functional ops.
    For example:
        a = x > 1 and y < 1

    Transformed code:
        a = dy2s.bool_and( dy2s.cmp_gt(x,1), dy2s.cmp_lt(y,1) )
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        return self.visit(self.root)

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, gast.And):
            new_node = self._create_bool_op_node(node.values, 'bool_and')
        elif isinstance(node.op, gast.Or):
            new_node = self._create_bool_op_node(node.values, 'bool_or')
        else:
            raise TypeError(
                "Only supports and/or syntax in control flow if statement."
            )
        return new_node

    def visit_Compare(self, node):
        self.generic_visit(node)
        ops_map = {
            gast.Gt: 'cmp_gt',
            gast.Lt: 'cmp_lt',
            gast.GtE: 'cmp_ge',
            gast.LtE: 'cmp_le',
            gast.Eq: 'cmp_eq',
            gast.NotEq: 'cmp_ne',
            gast.In: 'cmp_in',
            gast.NotIn: 'cmp_notin',
            gast.Is: 'cmp_is',
            gast.IsNot: 'cmp_isnot'
        }
        op_func = ops_map[type(node.ops[0])]
        new_node = self._create_cmp_op_node(node.left, node.comparators[0], op_func)
        for op, comparator in zip(node.ops[1:], node.comparators[1:]):
            op_func = ops_map[type(op)]
            new_node = self._create_cmp_op_node(new_node, comparator, op_func)
        return new_node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, gast.Not):
            arg = ast_to_source_code(node.operand)
            new_node_str = f"dy2s.unary_not({arg})"
            new_node = gast.parse(new_node_str).body[0].value
            return new_node
        return node

    def _create_bool_op_node(self, nodes, api_type):
        args = [ast_to_source_code(child) for child in nodes]
        new_node_str = "dy2s.{}({}, {})".format(api_type, args[0], args[1])
        new_node = gast.parse(new_node_str).body[0].value
        return new_node

    def _create_cmp_op_node(self, left, right, api_type):
        left_arg = ast_to_source_code(left)
        right_arg = ast_to_source_code(right)
        new_node_str = "dy2s.{}({}, {})".format(api_type, left_arg, right_arg)
        new_node = gast.parse(new_node_str).body[0].value
        return new_node
