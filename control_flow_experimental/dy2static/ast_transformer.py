# gast is a generic AST to represent Python2 and Python3's Abstract Syntax Tree(AST).
# It provides a compatibility layer between the AST of various Python versions,
# as produced by ast.parse from the standard ast module.
# See details in https://github.com/serge-sans-paille/gast/

import os

from . import logging_utils
from .assert_transformer import AssertTransformer
from .base_transformer import BaseTransformer
from .break_continue_transformer import (
    BreakContinueTransformer,
    BreakTransformOptimizer,
)
from .call_transformer import CallTransformer
from .cast_transformer import CastTransformer
from .create_variable_transformer import CreateVariableTransformer
from .decorator_transformer import DecoratorTransformer
from .early_return_transformer import EarlyReturnTransformer
from .ifelse_transformer import IfElseTransformer
from .logical_transformer import LogicalTransformer
from .loop_transformer import LoopTransformer
from .return_transformer import ReturnTransformer
from .typehint_transformer import TypeHintTransformer
from .docstrings_removal import DocstringTransformer
from .comprehensions_transformer import ListCompTransformer
from .utils import ast_to_source_code

__all__ = []


def apply_optimization(transformers):
    """
    Judge wheter to apply optimized transformation, such as BreakTransformOptimizer.
    And not all optimized transformations are applied by default. It's controlled by
    'export FLAGS_optim_transformation=1'
    """
    flag = str(os.environ.get('FLAGS_optim_transformation')) in [
        '1',
        'True',
        'true',
    ]
    if flag:
        transformers.insert(3, BreakTransformOptimizer)


class DygraphToStaticAst(BaseTransformer):
    """
    Main class to transform Dygraph to Static Graph
    """

    def __init__(self):
        self.translator_logger = logging_utils.TranslatorLogger()

    def get_static_ast(self, root):
        self.root = root
        self.decorate_func_name = None

        # inplace transfer
        self.transfer_from_node_type(self.root)
        return self.root

    def _apply(self, transformer, node, log_level):
        transformer(node).transform()
        self.translator_logger.log_transformed_code(
            log_level, self.root, transformer.__name__
        )

    def transfer_from_node_type(self, node):
        self.translator_logger.log(
            1, f"Source code: \n{ast_to_source_code(self.root)}"
        )
        # Generic transformation
        self.visit(node)

        transformers = [
            DocstringTransformer, # remove all docstrings from function
            EarlyReturnTransformer,
            ListCompTransformer,
            BreakContinueTransformer,  # break/continue in loops
            ReturnTransformer,  # return in functions
            CreateVariableTransformer,  # create undefined var for if / while / for
            LoopTransformer,  # for/while -> while_op
            IfElseTransformer,  # if/else -> cond_op
            AssertTransformer,  # assert statement
            CallTransformer,  # transform call recursively
            LogicalTransformer,  # logical and/or/not
            CastTransformer,  # type casting statement
            DecoratorTransformer,  # transform decorators to function call
            TypeHintTransformer,  # remove all typehint in gast.Name
        ]

        apply_optimization(transformers)

        for index, transformer in enumerate(transformers):
            self._apply(transformer, node, log_level=index + 1)

        self.translator_logger.log_transformed_code(
            logging_utils.LOG_AllTransformer, self.root, "All Transformers"
        )

    def visit_FunctionDef(self, node):
        if self.decorate_func_name is None:
            self.decorate_func_name = node.name

        self.generic_visit(node)
        return node

    def get_module_name(self):
        """
        Return the main function name which will be used as module name
        in ast_to_func.
        """
        # Should consider BaseAPITransformer which add new module name in Yamei's PR.
        assert self.decorate_func_name, "decorate_func_name shall not be None."
        return self.decorate_func_name
