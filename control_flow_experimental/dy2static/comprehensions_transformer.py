from .base_transformer import BaseTransformer
from .helpers import gast
__all__ = []

class ListCompTransformer(BaseTransformer):
    """
    A class to transform list comprehensions into standard control flow.
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        self.visit(self.root)

    def visit_Assign(self, node):
        if not isinstance(node.value, gast.ListComp):
            return self.generic_visit(node)
        if len(node.targets) > 1:
            raise NotImplementedError('multiple assignments')

        target, = node.targets
        list_comp_node = node.value

        # Create an empty list
        target_init = gast.Assign(
            targets=[target],
            value=gast.List(elts=[], ctx=gast.Load()),
        )

        # Create a for loop for each generator in the list comprehension
        body = []
        for gen in reversed(list_comp_node.generators):
            # Create an if statement for each condition in the generator
            for gen_if in reversed(gen.ifs):
                body = [
                    gast.If(
                        test=gen_if,
                        body=body,
                        orelse=[],
                    )
                ]
            # Append elements to the list inside the for loop
            body = [
                gast.For(
                    target=gen.target,
                    iter=gen.iter,
                    body=[
                        gast.Expr(
                            value=gast.Call(
                                func=gast.Attribute(
                                    value=gast.Name(id=target.id, ctx=gast.Load(), annotation=None, type_comment=None),
                                    attr='append',
                                    ctx=gast.Load(),
                                ),
                                args=[list_comp_node.elt],
                                keywords=[],
                            ),
                        ),
                    ] + body,
                    orelse=[],
                    type_comment=None,
                ),
            ]

        # Replace the original list comprehension with the new statements
        return [target_init] + body
