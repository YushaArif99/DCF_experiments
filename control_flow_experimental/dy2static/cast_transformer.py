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
        if func_str in self._castable_type:
            if node.args:
                args_str = ast_to_source_code(node.args[0]).strip()
            else:
                assert func_str in ("list","tuple","set","dict"), f"cannot have empty args when calling {func_str}"
                # Create an empty instance for lists/dicts/tuples etc.
                args_str = '[]' if func_str in {'list', 'set', 'tuple'} else '{}'
            new_func_str = f"dy2s.cast_to_{func_str}({args_str})"
            new_node = gast.parse(new_func_str).body[0].value
            return new_node

        return node
