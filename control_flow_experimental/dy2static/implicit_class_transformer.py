from control_flow_experimental.dy2static.utils import ast_to_source_code
from .helpers import gast

from .base_transformer import BaseTransformer

__all__ = []


class ImplicitToExplicitTransformer(BaseTransformer):
    """
    This class transforms implicit class creations into explicit ones in Static Graph Ast.
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        self.visit(self.root)

    def visit_List(self, node):
        if isinstance(node.ctx, gast.Store):
            return node
        self.generic_visit(node)
        new_node = gast.Call(
            func=gast.Name(id='list', ctx=gast.Load(), annotation=None, type_comment=None),
            args=[node] if node.elts else [],
            keywords=[]
        )
        return new_node

    def visit_Dict(self, node):
        self.generic_visit(node)
        new_node = gast.Call(
            func=gast.Name(id='dict', ctx=gast.Load(), annotation=None, type_comment=None),
            args=[node] if node.keys and node.values else [],
            keywords=[]
        )
        return new_node

    def visit_Tuple(self, node):
        return node
        if isinstance(node.ctx, gast.Store):
            return node
        self.generic_visit(node)
        new_node = gast.Call(
            func=gast.Name(id='tuple', ctx=gast.Load(), annotation=None, type_comment=None),
            args=[node] if node.elts else [],
            keywords=[]
        )
        return new_node
