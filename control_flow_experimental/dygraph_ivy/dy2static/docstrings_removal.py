from .base_transformer import BaseTransformer
from ..utils import gast
__all__ = []

class DocstringTransformer(BaseTransformer):
    """
    A class to remove docstrings in gast.FunctionDef.
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        node.returns = None
        self.generic_visit(node)
        if isinstance(node.body[0], gast.Expr) and isinstance(node.body[0].value, gast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node
