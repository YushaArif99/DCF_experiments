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
"""Converts the ternary conditional operator."""

import gast

from control_flow_experimental.autograph_ivy.core import converter
from control_flow_experimental.autograph_ivy.pyct import parser
from control_flow_experimental.autograph_ivy.pyct import templates

from control_flow_experimental.autograph_ivy.pyct import anno


class ConditionalExpressionTransformer(converter.Base):
    """Converts conditional expressions to functional form."""
    
    
    def visit_IfExp(self, node):
        node = self.generic_visit(node)

        # tuple(locals().values()) indicates to the graph compiler
        # that all local variables are considered parameters.
        template = '''
                ivy.if_else(
                        test,
                        lambda *_: true_expr,
                        lambda *_: false_expr,
                        tuple(locals().values()))
        '''
        ret = templates.replace_as_expression(
                template,
                test=node.test,
                true_expr=node.body,
                false_expr=node.orelse)
        return ret

def transform(node, ctx):
    node = ConditionalExpressionTransformer(ctx).visit(node)
    return node
