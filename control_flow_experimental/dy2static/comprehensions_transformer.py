from .base_transformer import BaseTransformer
from .helpers import gast

__all__ = []


class CompTransformer(BaseTransformer):
    """
    A class to transform list/dict/set comprehensions into standard control flow.
    """

    def __init__(self, root):
        self.root = root
        self.var_counter = 0
        self.created_vars = []

    def transform(self):
        self.visit(self.root)

    def visit_Assign(self, node):
        if not isinstance(node.value, (gast.DictComp, gast.ListComp, gast.SetComp)):
            return self.generic_visit(node)
        if len(node.targets) > 1:
            raise NotImplementedError("multiple assignments")

        (target,) = node.targets
        # Create an empty list or dict or set
        type_str = 'list' if isinstance(node.value, gast.ListComp) else ('set' if isinstance(node.value, gast.SetComp) else 'dict')
        target_init = gast.Assign(
            targets=[target],
            value=gast.Call(
                func=gast.Attribute(
                    value=gast.Name(id='dy2s', ctx=gast.Load(), annotation=None, type_comment=None),
                    attr='initialize_comprehension',
                    ctx=gast.Load(),
                ),
                args=[gast.Constant(value=type_str, kind=None)],
                keywords=[],
            ),
        )

        # Create a for loop for each generator in the comprehension
        body = self.transform_comprehension(node.value, target.id)

        # Initialize all created variables
        var_inits = [
            gast.Assign(
                targets=[
                    gast.Name(
                        id=var_name,
                        ctx=gast.Store(),
                        annotation=None,
                        type_comment=None,
                    )
                ],
                value=gast.List(elts=[], ctx=gast.Load())
                if isinstance(node.value, gast.ListComp)
                else gast.Call(
                    func=gast.Name(
                        id="set", ctx=gast.Load(), annotation=None, type_comment=None
                    ),
                    args=[],
                    keywords=[],
                )
                if isinstance(node.value, gast.SetComp)
                else gast.Dict(keys=[], values=[]),
            )
            for var_name in self.created_vars
        ]

        # Replace the original comprehension with the new statements
        return var_inits + [target_init] + body

    def transform_comprehension(self, comp_node, target_id):
        assert isinstance(
            comp_node, (gast.ListComp, gast.DictComp, gast.SetComp)
        ), f"expect a gast.Comp node but got node of type ({comp_node})"
        if isinstance(comp_node, (gast.ListComp, gast.SetComp)):
            update_func = "add" if isinstance(comp_node, gast.SetComp) else "append"
            update_args = [comp_node.elt]
            inner_node = comp_node.elt
        else:  # gast.DictComp
            update_func = "update"
            update_args = [gast.Dict(keys=[comp_node.key], values=[comp_node.value])]
            inner_node = comp_node.value

        # Check if the element being added is itself a comprehension or a conditional expression
        if isinstance(inner_node, (gast.ListComp, gast.DictComp, gast.SetComp)):
            # Create a new variable for the nested comprehension
            nested_target_id = self.new_variable()
            update_args = [
                gast.Name(
                    id=nested_target_id,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None,
                )
            ]

            # Transform the nested comprehension
            nested_body = self.transform_comprehension(inner_node, nested_target_id)

            # Add the transformed nested comprehension to the body of the outer comprehension
            body = nested_body + [
                gast.Expr(
                    value=gast.Call(
                        func=gast.Attribute(
                            value=gast.Name(
                                id=target_id,
                                ctx=gast.Load(),
                                annotation=None,
                                type_comment=None,
                            ),
                            attr=update_func,
                            ctx=gast.Load(),
                        ),
                        args=update_args,
                        keywords=[],
                    ),
                ),
            ]
        # Check if the element being added is a conditional expression
        elif isinstance(inner_node, gast.IfExp):
            # Transform each branch of the condition if it is a comprehension
            if isinstance(
                inner_node.body, (gast.ListComp, gast.DictComp, gast.SetComp)
            ):
                # Create a new variable for the true branch
                true_target_id = self.new_variable()
                body_true = self.transform_comprehension(
                    inner_node.body, true_target_id
                )
            else:
                body_true = inner_node.body

            if isinstance(
                inner_node.orelse, (gast.ListComp, gast.DictComp, gast.SetComp)
            ):
                # Create a new variable for the false branch
                false_target_id = self.new_variable()
                body_false = self.transform_comprehension(
                    inner_node.orelse, false_target_id
                )
            else:
                body_false = inner_node.orelse

            # Replace the original conditional expression with a new one
            inner_node = gast.IfExp(
                test=inner_node.test,
                body=gast.Name(
                    id=true_target_id,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None,
                )
                if isinstance(body_true, list)
                else body_true,
                orelse=gast.Name(
                    id=false_target_id,
                    ctx=gast.Load(),
                    annotation=None,
                    type_comment=None,
                )
                if isinstance(body_false, list)
                else body_false,
            )

            # Add the transformed branches to the body of the outer comprehension
            body = (
                (body_true if isinstance(body_true, list) else [])
                + (body_false if isinstance(body_false, list) else [])
                + [
                    gast.Expr(
                        value=gast.Call(
                            func=gast.Attribute(
                                value=gast.Name(
                                    id=target_id,
                                    ctx=gast.Load(),
                                    annotation=None,
                                    type_comment=None,
                                ),
                                attr=update_func,
                                ctx=gast.Load(),
                            ),
                            args=[inner_node],
                            keywords=[],
                        ),
                    ),
                ]
            )
        else:
            body = [
                gast.Expr(
                    value=gast.Call(
                        func=gast.Attribute(
                            value=gast.Name(
                                id=target_id,
                                ctx=gast.Load(),
                                annotation=None,
                                type_comment=None,
                            ),
                            attr=update_func,
                            ctx=gast.Load(),
                        ),
                        args=update_args,
                        keywords=[],
                    ),
                ),
            ]

        for gen in reversed(comp_node.generators):
            # Create an if statement for each condition in the generator
            for gen_if in reversed(gen.ifs):
                body = [
                    gast.If(
                        test=gen_if,
                        body=body,
                        orelse=[],
                    )
                ]
            # Add elements to the list or dict inside the for loop
            body = [
                gast.For(
                    target=gen.target,
                    iter=gen.iter,
                    body=body,
                    orelse=[],
                    type_comment=None,
                ),
            ]

        return body

    def new_variable(self):
        # Generate a new unique variable name
        self.var_counter += 1
        var_name = f"{self.var_counter}"

        # Add the new variable to the list of created variables
        self.created_vars.append(var_name)

        return var_name
