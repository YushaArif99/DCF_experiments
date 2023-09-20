from .utils import ast_to_source_code, is_ivy_api
from ..utils import gast

from .base_transformer import BaseTransformer
from .utils import is_builtin  # noqa: F401

PDB_SET = "pdb.set_trace"

__all__ = []


class CallTransformer(BaseTransformer):
    """
    This class transforms function calls into Static Graph Ast.
    """

    def __init__(self, root):
        self.root = root

    def _no_need_convert_call(self, node):
        """
        Determines whether a function needs to be transformed by `convert_call`.
        It doesn't need to be transformed when a function satisfies the following conditions:
          1. It's an api of ivy
          2. It's a python builtin function not include `len`, `zip`, `range` and `enumerate`
        """
        assert isinstance(node, gast.Call)
        if is_ivy_api(node):
            return True

        func_str = ast_to_source_code(node.func).strip()
        try:
            need_convert_builtin_func_list = {
                'len',
                'zip',
                'range',
                'enumerate',
                'print',
            }
            is_builtin = eval(f"is_builtin({func_str})")  # noqa: F811
            need_convert = func_str in need_convert_builtin_func_list
            return is_builtin and not need_convert
        except Exception:
            return False

    def transform(self):
        self.visit(self.root)

    def visit_Call(self, node):
        self.generic_visit(node)

        if self._no_need_convert_call(node):
            return node

        func_str = ast_to_source_code(node.func).strip()

        # NOTE(liym27): Don't convert `pad.set_trace` even if the convertion doesn't work finally, because
        # it is clearer to see where it is called from.
        if PDB_SET in func_str:
            return node

        new_func_str = f"ivy.Call({func_str})"
        new_func_ast = gast.parse(new_func_str).body[0].value
        node.func = new_func_ast

        return node
