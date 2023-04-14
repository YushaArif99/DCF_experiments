# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts function definitions and lambdas by adding necessary boilerplate."""

import gast

from control_flow_experimental.autograph_ivy.core import converter
from control_flow_experimental.autograph_ivy.pyct import anno
from control_flow_experimental.autograph_ivy.pyct import parser
from control_flow_experimental.autograph_ivy.pyct import qual_names
from control_flow_experimental.autograph_ivy.pyct import templates
from control_flow_experimental.autograph_ivy.pyct.static_analysis import activity
from control_flow_experimental.autograph_ivy.pyct.static_analysis import annos


class _Function(object):

    def __init__(self):
        self.context_name = None


class FunctionTransformer(converter.Base):
    """Wraps function bodies around autograph-specific boilerplate."""

    def visit_Lambda(self, node):
        with self.state[_Function] as fn_scope:
            node = self.generic_visit(node)

            scope = anno.getanno(node, anno.Static.SCOPE)
            function_context_name = self.ctx.namer.new_symbol('lscope',
                                                                                                                scope.referenced)
            fn_scope.context_name = function_context_name
            anno.setanno(node, 'function_context_name', function_context_name)

            return node

    def visit_FunctionDef(self, node):
        with self.state[_Function] as fn_scope:
            scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)

            function_context_name = self.ctx.namer.new_symbol('fscope',
                                                                                                                scope.referenced)
            fn_scope.context_name = function_context_name
            anno.setanno(node, 'function_context_name', function_context_name)

            node = self.generic_visit(node)

            if fn_scope.level <= 2:
                # Top-level functions lose their decorator because the conversion is
                # always just-in-time and by the time it happens the decorators are
                # already set to be applied.
                node.decorator_list = []

            docstring_node = None
            if node.body:
                first_statement = node.body[0]
                if (isinstance(first_statement, gast.Expr) and
                        isinstance(first_statement.value, gast.Constant)):
                    docstring_node = first_statement
                    node.body = node.body[1:]

            if docstring_node is not None:
                node.body = [docstring_node] + node.body

            return node


def transform(node, ctx):
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx, None)

    return FunctionTransformer(ctx).visit(node)
