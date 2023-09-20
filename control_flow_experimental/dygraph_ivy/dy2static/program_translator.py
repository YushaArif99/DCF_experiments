import inspect
import threading
import weakref
from .ast_transformer import DygraphToStaticAst
from .origin_info import (
    attach_origin_info,
    create_and_update_origin_info_map,
)
from ..utils import gast
from .utils import (
    ALREADY_D2S,
    ast_to_func,
    func_to_source_code,
    is_ivy_func,
    unwrap,
)

__all__ = []

# For each traced function, we set `max_traced_program_count` = 10 to consider caching performance.
# Once exceeding the threshold, we will raise warning to users to make sure the conversion is as expected.
MAX_TRACED_PROGRAM_COUNT = 10

CONVERSION_OPTIONS = "_ivy_not_to_static"


def synchronized(func):
    func.__lock__ = threading.Lock()

    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return lock_func


class FunctionCache:
    """
    Caches the transformed functions to avoid redundant conversions of the same function.
    """

    def __init__(self):
        # Caches the converted static functions. {dygraph_func: static_func}
        self._converted_static_func_caches = weakref.WeakKeyDictionary()
        # Caches the converted ast node for same source code. {source_code: ast_root}
        self._code_to_ast_caches = {}
        self._dygraph_to_static = DygraphToStaticAst()

    def convert_with_cache(self, func):
        """
        Returns the cached static function or converts it when first encounters the function.
        """
        # If hit cache, return it directly.
        static_func = self._converted_static_func_caches.get(func, None)

        if static_func is None:
            static_func = self._convert(func)
            self._converted_static_func_caches[func] = static_func

        return static_func

    def _convert(self, func):
        """
        Converts dygraph function into static function. For two functions with same dedent code,
        the second function will reuse the transformed ast node of previous one.

        For example:
            # A.py
            def foo(x, y):
                z = x + y
                return z

            # B.py
            def foo(x, y):
                z = x + y
                return z

        If the conversion of A.foo happens after B.foo, it will reuse the transformed ast node of B.foo
        to speed up the conversion.
        """
        # Note: In Python2, it will raise OSError when inspect function
        # with decorator directly and function.__wrapped__ holds the actual function.
        func = unwrap(func)
        source_code = func_to_source_code(func)

        # TODO(liym27):
        #  Consider this case: source_code in self._code_to_ast_caches,
        #  but actually they are methods in different classes.
        #  Maybe use (__class__, source_code) as key
        if source_code in self._code_to_ast_caches:
            root = self._code_to_ast_caches[source_code]
        else:
            root = gast.parse(source_code)
            root = attach_origin_info(root, func)
            root = self._dygraph_to_static.get_static_ast(root)
            self._code_to_ast_caches[source_code] = root

        # Get static function from AST
        static_func, file_name = ast_to_func(root, func, False)

        create_and_update_origin_info_map(root, static_func)
        return static_func

    def exist(self, func):
        return func in self._converted_static_func_caches


_CACHE_LOCK = threading.Lock()
_FUNCTION_CACHE = FunctionCache()


def convert_to_static(function):
    """
    Transforms function of dygraph into static function using the cache mechanism.

    Note(dev): It will return function.__func__ if encountering class method.

    Args:
        function(callable): The function with dygraph layers that will be converted into static layers.
    """
    if getattr(function, ALREADY_D2S, None):
        return function

    # Return directly if decorated with @not_to_static and DO NOT Cache it
    options = getattr(function, CONVERSION_OPTIONS, None)
    # or ignore ivy api
    need_skip = (options is not None and options.not_convert) or is_ivy_func(
        function
    )
    if need_skip:
        return function.__func__ if inspect.ismethod(function) else function

    with _CACHE_LOCK:
        static_func = _FUNCTION_CACHE.convert_with_cache(function)
        setattr(static_func, ALREADY_D2S, True)
        return static_func


