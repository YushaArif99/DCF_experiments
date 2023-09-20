from ..utils import gast

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
    Transform python boolean op into Ivy logical op.

    For example:
        a = x > 1 and y < 1

    Transformed code:
        a = cfe.And(lambda:x>1, lambda:y<1)
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        return self.visit(self.root)

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, gast.Not):
            arg = ast_to_source_code(node.operand)
            new_node_str = f"cfe.Not({arg})"
            # NOTE: gast.parse returns Module(body=[expr(value=...)])
            new_node = gast.parse(new_node_str).body[0].value
            return new_node
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, gast.And):
            new_node = self._create_bool_op_node(node.values, 'And')
        elif isinstance(node.op, gast.Or):
            new_node = self._create_bool_op_node(node.values, 'Or')
        else:
            raise TypeError(
                "Only supports and/or syntax in control flow if statement."
            )
        return new_node

    def _create_bool_op_node(self, nodes, api_type):
        '''
        NOTE:
           The arguments of function convert_logical_XX should be callable so that they can be run
          according to the actual order. In `convert_logical_and(lambda:x>1, lambda:y<1)`, `lambda:y<1`
          must be run after `lambda:x>1`, If `x>1` is False, `y<1` should NOT be run.
        '''
        assert (
            len(nodes) > 1
        ), "The length of BoolOp should be at least 2, but received {}.".format(
            len(nodes)
        )
        if len(nodes) > 2:
            # Creates logic_and/logic_or node recursively.
            pre_logic_node = self._create_bool_op_node(nodes[:2], api_type)
            if len(nodes[2:]) == 1:
                post_logic_node = nodes[2]
            else:
                post_logic_node = self._create_bool_op_node(nodes[2:], api_type)
            nodes = [pre_logic_node] + [post_logic_node]

        args = [ast_to_source_code(child) for child in nodes]
        new_node_str = "cfe.{}(lambda:{}, lambda:{})".format(
            api_type, args[0], args[1]
        )
        # NOTE: gast.parse return Module(body=[expr(...)])
        new_node = gast.parse(new_node_str).body[0].value
        return new_node
