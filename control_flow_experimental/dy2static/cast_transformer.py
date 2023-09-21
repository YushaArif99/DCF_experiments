from control_flow_experimental.dy2static.utils import ast_to_source_code
from .helpers import gast

from .base_transformer import BaseTransformer

__all__ = []


class CastTransformer(BaseTransformer):
    """
    This class transforms type casting into Static Graph Ast.
    """

    def __init__(self, root):
        self.root = root
        self._castable_type = {'bool', 'int', 'float', 'set', 'dict', 'tuple', 'list'}

    def transform(self):
        self.visit(self.root)

    def visit_Call(self, node):
        self.generic_visit(node)
        func_str = ast_to_source_code(node.func).strip()
        if func_str in self._castable_type and len(node.args) > 0:
            args_str = ast_to_source_code(node.args[0]).strip()
            new_func_str = f"cfe.cast_to_{func_str}({args_str})"
            new_node = gast.parse(new_func_str).body[0].value
            return new_node

        return node
