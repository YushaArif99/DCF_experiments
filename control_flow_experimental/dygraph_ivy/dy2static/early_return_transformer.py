from ..utils import gast

from .base_transformer import BaseTransformer

__all__ = []


class EarlyReturnTransformer(BaseTransformer):
    """
    Transform if/else return statement of Dygraph into Static Graph.
    """

    def __init__(self, root):
        self.root = root

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)

    def is_define_return_in_if(self, node):
        assert isinstance(
            node, gast.If
        ), "Type of input node should be gast.If, but received %s ." % type(
            node
        )
        for child in node.body:
            if isinstance(child, gast.Return):
                return True
        return False

    def visit_block_nodes(self, nodes):
        result_nodes = []
        destination_nodes = result_nodes
        for node in nodes:
            rewritten_node = self.visit(node)

            if isinstance(rewritten_node, (list, tuple)):
                destination_nodes.extend(rewritten_node)
            else:
                destination_nodes.append(rewritten_node)

            # append other nodes to if.orelse even though if.orelse is not empty
            if isinstance(node, gast.If) and self.is_define_return_in_if(node):
                destination_nodes = node.orelse
                # handle stmt like `if/elif/elif`
                while (
                    len(destination_nodes) > 0
                    and isinstance(destination_nodes[0], gast.If)
                    and self.is_define_return_in_if(destination_nodes[0])
                ):
                    destination_nodes = destination_nodes[0].orelse

        return result_nodes

    def visit_If(self, node):
        node.body = self.visit_block_nodes(node.body)
        node.orelse = self.visit_block_nodes(node.orelse)
        return node

    def visit_While(self, node):
        node.body = self.visit_block_nodes(node.body)
        node.orelse = self.visit_block_nodes(node.orelse)
        return node

    def visit_For(self, node):
        node.body = self.visit_block_nodes(node.body)
        node.orelse = self.visit_block_nodes(node.orelse)
        return node

    def visit_FunctionDef(self, node):
        node.body = self.visit_block_nodes(node.body)
        return node
