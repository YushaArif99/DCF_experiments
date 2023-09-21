from .base_transformer import BaseTransformer
from .utils import FunctionNameLivenessAnalysis
from .variable_trans_func import create_undefined_var

__all__ = []


class CreateVariableTransformer(BaseTransformer):
    """ """

    def __init__(self, root):
        self.root = root
        FunctionNameLivenessAnalysis(self.root)

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        # attributes = set(filter(lambda x: '.' in x, node.ivy_scope.modified_vars()))
        self.generic_visit(node)
        bodys = node.body
        names = sorted(node.ivy_scope.created_vars())
        for name in names:
            bodys[0:0] = [create_undefined_var(name)]
        return node
