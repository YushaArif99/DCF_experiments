
"""Handles assigns and unpacks."""

import gast

from control_flow_experimental.autograph_ivy.core import converter

class AssignsTransformer(converter.Base):
    """Transforms assign statements."""

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        targets = node.targets
        if len(targets) == 1 and isinstance(targets[0], gast.Tuple):
            node.targets[0].elts.append(gast.Starred(gast.Name(id='_', ctx=gast.Store(), annotation=None, type_comment=None), gast.Store()))
        return node


def transform(node, ctx):
    transformer = AssignsTransformer(ctx)
    node = transformer.visit(node)
    return node
