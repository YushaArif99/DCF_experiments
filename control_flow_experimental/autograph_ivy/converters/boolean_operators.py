"""Lowers python boolean operators into function calls.

Example:

    x or y or z

becomes

    cfe.bool_or(lambda: cfe.bool_or(lambda: x, lambda: y), lambda: z)

"""

import gast

from control_flow_experimental.autograph_ivy.core import converter
from control_flow_experimental.autograph_ivy.pyct import templates


# Define a helper function to create lambda nodes from expressions
def create_lambda_node(expr):
    # Create a lambda node with no arguments and the given expression as the body
    return gast.Lambda(
        args=gast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=expr
    )


class BooleanTransformer(converter.Base):
    """Lowers boolean operators into function calls."""

    def visit_BoolOp(self, node):
        # A BoolOp node has an op attribute (which can be Or or And)
        # and a values attribute (which is a list of expressions)
        if isinstance(node.op, gast.Or):
            template = """
                cfe.bool_or(args)
            """
        elif isinstance(node.op, gast.And):
            template = """
                cfe.bool_and(args)
            """
        else:
            raise NotImplementedError('unsupported boolean operator')

        # Recursively visit the values to transform any nested boolean operators
        values = [self.visit(v) for v in node.values]

        # Convert all the values to lambdas to enable short-circuit evaluation
        values = [create_lambda_node(v) for v in values]

        return templates.replace_as_expression(template, args=values)

    def visit_Compare(self, node):
        # A Compare node has a left attribute (which is an expression)
        # and a list of ops and comparators attributes (which are also expressions)
        # For example, x == y == z has left = x, ops = [Eq, Eq], comparators = [y, z]
        
        # Recursively visit the left and comparators to transform any nested boolean operators
        left = self.visit(node.left)
        comparators = [self.visit(c) for c in node.comparators]

        # Convert all the left and comparators to lambdas to enable lazy evaluation
        left = create_lambda_node(left)
        comparators = [create_lambda_node(c) for c in comparators]

        # Use a template that matches the first operator and pass the left and the first comparator as arguments
        op = node.ops[0]
        if isinstance(op, gast.Eq):
            template = """
                cfe.cmp_eq(left, right)
            """
        elif isinstance(op, gast.NotEq):
            template = """
                cfe.cmp_ne(left, right)
            """
        elif isinstance(op, gast.Gt):
            template = """
                cfe.cmp_gt(left, right)
            """
        elif isinstance(op, gast.GtE):
            template = """
                cfe.cmp_ge(left, right)
            """
        elif isinstance(op, gast.Lt):
            template = """
                cfe.cmp_lt(left, right)
            """
        elif isinstance(op, gast.LtE):
            template = """
                cfe.cmp_le(left, right)
            """
        elif isinstance(op, gast.Is):
            template = """
                cfe.cmp_is(left, right)
            """
        elif isinstance(op, gast.IsNot):
            template = """
                cfe.cmp_isnot(left, right)
            """
        elif isinstance(op, gast.In):
            template = """
                cfe.cmp_in(left, right)
            """
        elif isinstance(op, gast.NotIn):
            template = """
                cfe.cmp_notin(left, right)
           """
        else:
            raise NotImplementedError('unsupported comparison operator')

        result = templates.replace_as_expression(template, left=left, right=comparators[0])

        # If there are more operators and comparators, use a loop to chain them with the result
        if len(node.ops) > 1:
            for i in range(1, len(node.ops)):
                op = node.ops[i]
                if isinstance(op, gast.Eq):
                    template = """
                        cfe.cmp_eq(result, arg)
                    """
                elif isinstance(op, gast.NotEq):
                    template = """
                        cfe.cmp_ne(result, arg)
                    """
                elif isinstance(op, gast.Gt):
                    template = """
                        cfe.cmp_gt(result, arg)
                    """
                elif isinstance(op, gast.GtE):
                    template = """
                        cfe.cmp_ge(result, arg)
                    """
                elif isinstance(op, gast.Lt):
                    template = """
                        cfe.cmp_lt(result, arg)
                    """
                elif isinstance(op, gast.LtE):
                    template = """
                        cfe.cmp_le(result, arg)
                    """
                elif isinstance(op, gast.Is):
                    template = """
                        cfe.cmp_is(result, arg)
                    """
                elif isinstance(op, gast.IsNot):
                    template = """
                        cfe.cmp_isnot(result, arg)
                    """
                elif isinstance(op, gast.In):
                    template = """
                        cfe.cmp_in(result, arg)
                    """
                elif isinstance(op, gast.NotIn):
                    template = """
                        cfe.cmp_notin(result, arg)
                   """
                else:
                    raise NotImplementedError('unsupported comparison operator')

                result = templates.replace_as_expression(template, result=result, arg=comparators[i])

        return result
    
    def visit_UnaryOp(self, node):
        # A UnaryOp node has an op attribute (which can be Not or other unary operators)
        # and an operand attribute (which is an expression)
        if isinstance(node.op, gast.Not):
            template = """
                cfe.unary_not(arg)
            """
        else:
            # For other unary operators, just visit them normally
            return self.generic_visit(node)

        # Recursively visit the operand to transform any nested boolean operators
        operand = self.visit(node.operand)

        # Convert the operand to a lambda to enable lazy evaluation
        operand = create_lambda_node(operand)

        return templates.replace_as_expression(template, arg=operand)

def transform(node, ctx):
    return BooleanTransformer(ctx).visit(node)
