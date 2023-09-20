from .base_transformer import BaseTransformer

__all__ = []


class TypeHintTransformer(BaseTransformer):
    """
    A class remove all the typehint in gast.Name(annotation).
    NOTE: Put it behind other transformers because other transformers (eg: static analysis) may rely on typehints.
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        node.annotation = None
        self.generic_visit(node)
        return node
