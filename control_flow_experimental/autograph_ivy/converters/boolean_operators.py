"""Lowers boolean operators into function calls.

Example:

    x or y or z

becomes

    fx.or_fn(x, y, z)

where fx.or_fn = lambda *args: any(args)
"""

import gast

from control_flow_experimental.autograph_ivy.core import converter
from control_flow_experimental.autograph_ivy.pyct import templates


# Define the lambda functions for the boolean operators
class BooleanTransformer(converter.Base):
    """Lowers boolean operators into function calls."""

    def visit_BoolOp(self, node):
        # A BoolOp node has an op attribute (which can be Or or And)
        # and a values attribute (which is a list of expressions)
        if isinstance(node.op, gast.Or):
            # Use the or_fn function for the Or operator
            template = """
                fx.or_fn(args)
            """
        elif isinstance(node.op, gast.And):
            # Use the and_fn function for the And operator
            template = """
                fx.and_fn(args)
            """
        else:
            raise NotImplementedError('unsupported boolean operator')

        # Recursively visit the values to transform any nested boolean operators
        values = [self.visit(v) for v in node.values]

        return templates.replace_as_expression(template, args=values)

    def visit_Compare(self, node):
        # A Compare node has a left attribute (which is an expression)
        # and a list of ops and comparators attributes (which are also expressions)
        # For example, x == y == z has left = x, ops = [Eq, Eq], comparators = [y, z]
        if len(node.ops) == 1:
            # For single comparison operators, just visit them normally
            return self.generic_visit(node)

        if any(isinstance(op, gast.Eq) for op in node.ops):
            # Use the eq_fn function for the Eq operator
            template = """
                fx.eq_fn(args)
            """
        elif any(isinstance(op, gast.NotEq) for op in node.ops):
            # Use the ne_fn function for the NotEq operator
            template = """
                fx.ne_fn(args)
            """
        elif any(isinstance(op, gast.Gt) for op in node.ops):
            # Use the gt_fn function for the Gt operator
            template = """
                fx.gt_fn(args)
            """
        elif any(isinstance(op, gast.GtE) for op in node.ops):
            # Use the ge_fn function for the GtE operator
            template = """
                fx.ge_fn(args)
            """
        elif any(isinstance(op, gast.Lt) for op in node.ops):
            # Use the lt_fn function for the Lt operator
            template = """
                fx.lt_fn(args)
            """
        elif any(isinstance(op, gast.LtE) for op in node.ops):
            # Use the le_fn function for the LtE operator
            template = """
                fx.le_fn(args)
            """
        else:
            raise NotImplementedError('unsupported comparison operator')

        # Recursively visit the left and comparators to transform any nested boolean operators
        left = self.visit(node.left)
        comparators = [self.visit(c) for c in node.comparators]

        return templates.replace_as_expression(template, args=[left] + comparators)


def transform(node, ctx):
    return BooleanTransformer(ctx).visit(node)
