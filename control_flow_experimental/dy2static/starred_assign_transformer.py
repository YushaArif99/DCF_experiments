from .base_transformer import BaseTransformer
from .helpers import gast


class StarredToRegularTransformer(BaseTransformer):
    """
    This class transforms starred assignments into regular sequence unpackings in Static Graph Ast.
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        self.visit(self.root)

    def get_indexes(self, elts):
        before_vars = []
        starred_var = []
        after_vars = []
        for i, elt in enumerate(elts):
            if isinstance(elt, gast.Starred):
                starred_var.append(i)
            elif not starred_var:
                before_vars.append(i)
            else:
                after_vars.append(-len(elts) + i)
        return before_vars, starred_var, after_vars

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.targets[0], gast.Tuple):
            elts = node.targets[0].elts
            if any(isinstance(elt, gast.Starred) for elt in elts):
                new_elts = []
                new_values = []
                before_vars, starred_var, after_vars = self.get_indexes(elts)
                for i in before_vars + starred_var + after_vars:
                    elt = elts[i]
                    if isinstance(elt, gast.Starred):
                        new_elts.append(elt.value)
                        new_values.append(
                            gast.Call(
                                func=gast.Name(
                                    id="list",
                                    ctx=gast.Load(),
                                    annotation=None,
                                    type_comment=None,
                                ),
                                args=[
                                    gast.Subscript(
                                        value=node.value,
                                        slice=gast.Slice(
                                            lower=None
                                            if not before_vars
                                            else gast.Constant(value=i, kind=None),
                                            upper=None
                                            if not after_vars
                                            else gast.UnaryOp(
                                                op=gast.USub(),
                                                operand=gast.Constant(
                                                    value=abs(after_vars[0]), kind=None
                                                ),
                                            ),
                                            step=None,
                                        ),
                                        ctx=gast.Load(),
                                    )
                                ],
                                keywords=[],
                            )
                        )
                    else:
                        new_elts.append(elt)
                        new_values.append(
                            gast.Subscript(
                                value=node.value,
                                slice=gast.Constant(value=i, kind=None)
                                if i >= 0
                                else gast.UnaryOp(
                                    op=gast.USub(),
                                    operand=gast.Constant(value=abs(i), kind=None),
                                ),
                                ctx=gast.Load(),
                            )
                        )
                node.targets[0].elts = new_elts
                node.value = gast.Tuple(elts=new_values, ctx=gast.Load())
        return node
