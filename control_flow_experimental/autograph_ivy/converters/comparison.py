"""Handles comparison expressions: lt, gt, is, isnot."""

import gast

from control_flow_experimental.autograph_ivy.core import converter
from control_flow_experimental.autograph_ivy.pyct import templates

class CompareTransformer(converter.Base):
    """Transforms comparisons like Is, IsNot."""

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) > 1:
            return node
        left = node.left
        op = node.ops[0]
        right = node.comparators[0]
        operator_fns = {
            gast.Is.__name__ : "ivy.cmp_is",
            gast.IsNot.__name__ : "ivy.cmp_isnot",
        }

        if type(op).__name__ not in operator_fns:
            return node

        prefix=operator_fns[type(op).__name__]
        
        template = f'''
            {prefix}(left, right)
        '''

        ret = templates.replace_as_expression(
            template,
            left=left,
            right=right,
        )
        return ret

def transform(node, ctx):
    node = CompareTransformer(ctx).visit(node)
    return node
