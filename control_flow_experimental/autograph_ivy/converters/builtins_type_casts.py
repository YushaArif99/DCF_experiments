"""Lowers built-in type casts into function calls.

Example:

    int(x)

becomes

    cfe.cast_to_int(x)

"""

import gast

from control_flow_experimental.autograph_ivy.core import converter
from control_flow_experimental.autograph_ivy.pyct import templates


class BuiltinsTransformer(converter.Base):
    """Lowers built-in type casts into function calls."""

    def visit_Call(self, node):
        # A Call node has a func attribute (which is an expression)
        # and an args attribute (which is a list of expressions)
        
        # Check if the func is a Name node and if its id is one of the built-in type casts
        if isinstance(node.func, gast.Name) and node.func.id in ('int', 'float', 'bool', 'str', 'list', 'tuple', 'set', 'dict'):
            # Use a template that matches the type cast with the corresponding function from the ivy module
            template = f"""
                cfe.cast_to_{node.func.id}(args)
            """
            
            # Recursively visit the args to transform any nested expressions
            args = [self.visit(a) for a in node.args]

            return templates.replace_as_expression(template, args=args)

        else:
            # For other function calls, just visit them normally
            return self.generic_visit(node)


def transform(node, ctx):
    return BuiltinsTransformer(ctx).visit(node)
