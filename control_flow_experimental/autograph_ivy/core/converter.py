# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Converter construction support.

This module contains a base class for all converters, as well as supporting
structures. These structures are referred to as contexts.

The class hierarchy is as follows:

        <your converter>
            [extends] converter.Base
                [extends] transformer.Base
                        [extends] gast.nodeTransformer
                    [uses] transformer.SourceInfo
                [uses] converter.EntityContext
                    [uses] converter.ProgramContext
                    [uses] transformer.SourceInfo

converter.Base is a specialization of transformer.Base for AutoGraph. It's a
very lightweight subclass that adds a `ctx` attribute holding the corresponding
EntityContext object (see below). Note that converters are not reusable, and
`visit` will raise an error if called more than once.

converter.EntityContext contains mutable state associated with an entity that
the converter processes.

converter.ProgramContext contains mutable state across related entities. For
example, when converting several functions that call one another, the
ProgramContext should be shared across these entities.

Below is the overall flow at conversion:

        program_ctx = ProgramContext(<entities to convert>, <global settings>, ...)
        while <program_ctx has more entities to convert>:
            entity, source_info = <get next entity from program_ctx>
            entity_ctx = EntityContext(program_ctx, source_info)
            for <each ConverterClass>:
                converter = ConverterClass(entity_ctx)

                # May update entity_ctx and program_ctx
                entity = converter.visit(entity)

            <add entity's dependencies to program_ctx>

Note that pyct contains a small number of transformers used for static analysis.
These implement transformer.Base, rather than converter.Base, to avoid a
dependency on AutoGraph.
"""


from control_flow_experimental.autograph_ivy.pyct import transformer


class Base(transformer.Base):
    """All converters should inherit from this class.

    Attributes:
        ctx: EntityContext
    """

    def __init__(self, ctx):
        super(Base, self).__init__(ctx)

        self._used = False
        self._ast_depth = 0

    def visit(self, node):
        if not self._ast_depth:
            if self._used:
                raise ValueError('converter objects cannot be reused')
            self._used = True

        self._ast_depth += 1
        try:
            return super(Base, self).visit(node)
        finally:
            self._ast_depth -= 1
