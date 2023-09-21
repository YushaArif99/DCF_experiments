import re
import warnings

from .helpers import gast

from .base_transformer import BaseTransformer
from .utils import RE_PYMODULE, RE_PYNAME, ast_to_source_code

from ivy.func_wrapper import FN_DECORATORS

__all__ = []

IGNORE_NAMES = [
    'wraps',
    'staticmethod',
    'classmethod',
    'decorator',
] + FN_DECORATORS


class DecoratorTransformer(BaseTransformer):
    """
    Transform decorators.
    """

    def __init__(self, root):
        self.root = root

        self.ancestor_nodes = []

    def transform(self):
        """
        Main function to transform AST.
        """
        self.visit(self.root)

    def visit_FunctionDef(self, node):
        assert isinstance(node, gast.FunctionDef)
        self.generic_visit(node)

        deco_list = node.decorator_list
        node.decorator_list = []

        # every decorator will append a node
        decofun_nodes = []
        # func to be decoed next time
        deco_target = '_orig_' + node.name
        # last decoed func
        decoed_func = ''

        for deco in reversed(deco_list):
            # skip INGNORE_NAMES
            deco_full_name = ast_to_source_code(deco).strip()
            if isinstance(deco, gast.Call):
                # match case like :
                # 1: @ivy.Call(a.b.c.d.deco)()
                # 2: @q.w.e.r.deco()
                re_tmp = re.match(
                    r'({module})*({name}\(){{0,1}}({module})*({name})(\)){{0,1}}\(.*$'.format(
                        name=RE_PYNAME, module=RE_PYMODULE
                    ),
                    deco_full_name,
                )
                deco_name = re_tmp.group(4)
            else:
                # match case like:
                # @a.d.g.deco
                re_tmp = re.match(
                    r'({module})*({name})$'.format(
                        name=RE_PYNAME, module=RE_PYMODULE
                    ),
                    deco_full_name,
                )
                deco_name = re_tmp.group(2)
            if deco_name in IGNORE_NAMES:
                continue
            elif deco_name == 'contextmanager':
                warnings.warn(
                    "Dy2Static : A context manager decorator is used, this may not work correctly after transform."
                )

            decoed_func = '_decoedby_' + deco_name

            # get function after decoration
            if isinstance(deco, gast.Call):
                if 'ivy.Call' in deco_full_name:
                    # in this case , the deco_full_name will be like:
                    # 'ivy.Call(deco)(5)'
                    rematch = re.match(
                        r'\ivy\.Call\((.+?)\)\((.*)\)', deco_full_name
                    )
                    re_name = rematch.group(1)
                    re_args = rematch.group(2)
                    re_args_with_func = deco_target + ', ' + re_args
                    decofun_str = 'try:\n\t{0} = ivy.Call({1})({2})\nexcept:\n\t{0} = ivy.Call({1})({3})({4})'.format(
                        decoed_func,
                        re_name,
                        re_args_with_func,
                        re_args,
                        deco_target,
                    )
                else:
                    # ivy api will not be transformed to 'ivy.Call'
                    rematch = re.match(r'(.+?)\((.*)\)', deco_full_name)
                    re_name = rematch.group(1)
                    re_args = rematch.group(2)
                    re_args_with_func = deco_target + ', ' + re_args
                    decofun_str = 'try:\n\t{0} = {1}({2})\nexcept:\n\t{0} = {1}({3})({4})'.format(
                        decoed_func,
                        re_name,
                        re_args_with_func,
                        re_args,
                        deco_target,
                    )

            else:
                decofun_str = '{} = cfe.Call({})({})'.format(
                    decoed_func, deco_full_name, deco_target
                )

            decofun_nodes.extend(gast.parse(decofun_str).body)
            deco_target = decoed_func

        if not decofun_nodes:
            return node

        orig_func_node = gast.FunctionDef(
            name='_orig_' + node.name,
            args=node.args,
            body=node.body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )

        args = [arg.id for arg in node.args.args]
        arg_str = ','.join(args)
        callfun_str = f'return {decoed_func}({arg_str})'
        callfun_node = gast.parse(callfun_str).body[0]

        node.body = [orig_func_node] + decofun_nodes + [callfun_node]

        return node
