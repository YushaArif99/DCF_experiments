import re
from control_flow_experimental.ivy_fx.fx.proxy import Proxy 
import ivy 

from .utils import (
    RETURN_NO_VALUE_VAR_NAME,
    Dygraph2StaticException,
    GetterSetterHelper,
    UndefinedVar,
)
__all__ = []


def indexable(x, code=None):
    if isinstance(x, Proxy):
        return x
    elif hasattr(x, '__iter__'):
        return list(x)
    elif hasattr(x, '__len__') and hasattr(
        x, '__getitem__'
    ):  # used for customed type and non-iterable type.
        return x
    else:
        raise RuntimeError("X can't be convert into indexable.")


def unpack_by_structure(target, structure):
    """unified unpack interface for ivy and python."""
    if isinstance(target, Proxy):
        return _unpack_by_structure_ivy(target, structure)
    else:
        return _unpack_by_structure_python(target, structure)


def _unpack_by_structure_python(target, structure):
    """TODO(xiongkun): analysis the differences between python and ivy unpack."""
    return _unpack_by_structure_ivy(target, structure)


def _unpack_by_structure_ivy(target, structure):
    if structure == 1:
        return target
    ret = []
    for idx, ele in enumerate(structure):
        if ele == 1:
            ret.append(target[idx])
            continue
        if isinstance(ele, list):
            ret.append(unpack_by_structure(target[idx], ele))
            continue
        raise AssertionError("structure element must be 1 or list")
    return ret


# def convert_while_loop(
#     cond, body, getter, setter, return_name_ids=None, push_pop_names=None
# ):
#     """
#     A function representation of a Python ``while`` statement.

#     Args:
#         cond(Callable): A callable object that returns a boolean variable to control whether to execute the loop body. It takes ``loop_vars`` as arguments.
#         body(Callable): A callable object that returns a tuple or list of variables with the same arguments ``loops_vars`` as ``cond`` .
#         get_args(callable): Get all arguments that needed in true_fn and false_fn.
#         set_args(callable): Update arguments that modified in trure_fn and false_fn.
#         return_name_ids(list[string], optional): the returned names.
#         push_pop_names(list[string], optional): the names on which called .append() or .pop().

#     Returns:
#         A list or tuple of variables which returned by ``body``.
#     """

#     # NOTE: It may be slower if cond is very expensive, but usually cond is just O(1).
#     # If loop_vars is changed during cond callable, then it causes bug, but current logical_and/logical_not/... doesn't change the loop_vars.
#     pred = cond()
#     if isinstance(pred, Proxy):
#         _run_paddle_while(
#             cond, body, getter, setter, return_name_ids, push_pop_names
#         )
#     else:
#         _run_py_while(cond, body, getter, setter)


# def _convert_tensor_arrray_if_necessary(setterhelper, push_pop_names):
#     push_pop_vars = setterhelper.get(push_pop_names)
#     if push_pop_vars is None:
#         return

#     def maybe_to_ivy_array(v):
#         if isinstance(v, list):
#             return ivy.asarray(v)
#         else:
#             return v

#     setterhelper.set(
#         push_pop_names, [maybe_to_ivy_array(v) for v in push_pop_vars]
#     )


# def _run_paddle_while(
#     cond, body, getter, setter, return_name_ids, push_pop_names
# ):
#     # NOTE: loop_vars of `control_flow.while_loop` must be Tensors.
#     helper = GetterSetterHelper(getter, setter, return_name_ids, push_pop_names)
#     _convert_tensor_arrray_if_necessary(helper, push_pop_names)

#     def new_body_fn(*args):
#         """wrap the body() and add return value for `while_loop`
#         the args may be differ from getter().
#         """
#         mutable_loop_vars = args
#         helper.set(return_name_ids, mutable_loop_vars)
#         body()
#         return helper.get(return_name_ids)

#     def new_cond_fn(*args):
#         """cond is a zero-args function, which is not
#         compatible with `while_loop`.
#         """
#         return cond()

#     # UndefinedVar will become data layer not check variable with value=NO_VALUE_MAGIC.
#     loop_vars = [
#         to_static_variable(var) if not isinstance(var, UndefinedVar) else var
#         for var in helper.get(return_name_ids)
#     ]
#     helper.set(
#         return_name_ids, loop_vars
#     )  # change the non-local var to variable
#     # variable maybe modified to inner var. change it into
#     from paddle.static.nn import while_loop

#     loop_vars = while_loop(new_cond_fn, new_body_fn, loop_vars)
#     helper.set(return_name_ids, loop_vars)
#     return loop_vars


def _run_py_while(cond, body, getter, setter):
    while True:
        pred = cond()
        if isinstance(pred, Proxy):
            raise Dygraph2StaticException(
                "python while pred change from bool to variable."
            )
        if not pred:
            break
        body()


def convert_logical_and(x_func, y_func):
    """
    A function representation of a Python ``and`` statement.

    Args:
        x_func(callable): x_func() is the left hand operand of ``and`` operator. x_func() is bool or Tensor.
        y_func(callable): y_func() is the right hand operand of ``and`` operator.  y_func() is bool or Tensor.

    Returns:
        A python bool variable or a bool Tensor.

    NOTE(liym27):
        1) The operands are executed sequentially according to the running logic of Python. So here the arguments
        should be callable.
        2) If the left hand operand is False, the right hand operand should be executed.

        For example:
            a = x > 1 and y < 1
        Transformed code:
            a = control_flow_experimental.dygraph..dy2static.convert_logical_and(lambda:x>1, lambda:y<1)

          In `convert_logical_and(lambda:x>1, lambda:y<1)`, `lambda:y<1` must be run after `lambda:x>1`. And
        if `x>1` is False, `y<1` should NOT be run.
    """
    x_value = x_func()
    if not isinstance(x_value, Proxy):
        return _run_py_logical_and(lambda: x_value, y_func)

    y_value = y_func()
    if not isinstance(y_value, Proxy):
        return _run_py_logical_and(lambda: y_value, lambda: x_value)

    return _run_ivy_logical_and(x_value, y_value)


def _run_ivy_logical_and(x, y):
    return ivy.logical_and(x, y) 


def _run_py_logical_and(x_func, y_func):
    x_value = x_func()
    assert not isinstance(x_value, Proxy)

    # NOTE(liym27):
    #  1. Returns y_func() if x_value is False;
    #  2. If x_value is False, y_func() should not be run.
    return x_value and y_func()


def convert_logical_or(x_func, y_func):
    """
    A function representation of a Python ``or`` statement.

    Args:
        x_func(callable): x_func() is the left hand operand of ``or`` operator. x_func() is bool or Tensor.
        y_func(callable): y_func() is the right hand operand of ``or`` operator.  y_func() is bool or Tensor.

    Returns:
        A python bool variable or a bool Tensor.

    NOTE(liym27):
        1) The operands are executed sequentially according to the running logic of Python. So here the arguments
        should be callable.
        2) If the left hand operand is True, the right hand operand should be executed.

        For example:
            a = x > 1 or y < 1
        Transformed code:
            a = control_flow_experimental.dygraph.dy2static.convert_logical_or(lambda:x>1, lambda:y<1)

        In `convert_logical_or(lambda:x>1, lambda:y<1)`, `lambda:y<1` must be run after `lambda:x>1`. And
        if `x>1` is True, `y<1` should NOT be run.
    """
    x_value = x_func()
    if not isinstance(x_value, Proxy):
        return _run_py_logical_or(lambda: x_value, y_func)

    y_value = y_func()
    if not isinstance(y_value, Proxy):
        return _run_py_logical_or(lambda: y_value, lambda: x_value)

    return _run_ivy_logical_or(x_value, y_value)


def _run_ivy_logical_or(x, y):
    return ivy.logical_or(x, y)


def _run_py_logical_or(x_func, y_func):
    x_value = x_func()
    assert not isinstance(x_value, Proxy)

    # NOTE(liym27):
    #  1. Returns y_func() if x_value is False;
    #  2. If x_value is True, y_func() should not be run.
    return x_value or y_func()


def convert_logical_not(x):
    """
    A function representation of a Python ``not`` statement.

    Args:
        x(bool|Tensor): Operand of ``not`` operator.

    Returns:
        A python bool variable or a bool Tensor.
    """

    if isinstance(x, Proxy):
        return _run_ivy_logical_not(x)
    else:
        return _run_py_logical_not(x)


def _run_ivy_logical_not(x):
    return ivy.logical_not(x)


def _run_py_logical_not(x):
    return not x


# def convert_ifelse(
#     pred,
#     true_fn,
#     false_fn,
#     get_args,
#     set_args,
#     return_name_ids,
#     push_pop_names=None,
# ):
#     """
#     A function representation of a Python ``if/else`` statement.

#     Args:
#         pred(bool|Tensor): A boolean Tensor which determines whether to return the result of ``true_fn`` or ``false_fn`` .
#         true_fn(callable): A callable to be performed if ``pred`` is true.
#         false_fn(callable): A callable to be performed if ``pred`` is false.
#         get_args(callable): Get all arguments that needed in true_fn and false_fn.
#         set_args(callable): Update arguments that modified in trure_fn and false_fn.
#         return_name_ids(list[string], optional): the returned names.
#         push_pop_names(list[string], optional): the names on which called .append() or .pop().

#     Returns:
#         ``true_fn()`` if the predicate ``pred`` is true else ``false_fn()`` .

#     """
#     if isinstance(pred, Proxy):
#         out = _run_paddle_cond(
#             pred,
#             true_fn,
#             false_fn,
#             get_args,
#             set_args,
#             return_name_ids,
#             push_pop_names,
#         )
#     else:
#         out = _run_py_ifelse(
#             pred, true_fn, false_fn, get_args, set_args, return_name_ids
#         )

#     return out


# def _run_paddle_cond(
#     pred, true_fn, false_fn, get_args, set_args, return_name_ids, push_pop_names
# ):
#     """
#     Paddle cond API will evaluate both true_fn and false_fn codes.
#     """
#     helper = GetterSetterHelper(
#         get_args, set_args, return_name_ids, push_pop_names
#     )
#     _convert_tensor_arrray_if_necessary(helper, push_pop_names)
#     pred = cast_bool_if_necessary(pred)
#     init_args = helper.get(return_name_ids)
#     from control_flow_experimental.dygraph.dygraph_to_static.program_translator import ProgramTranslator

#     inplace_map = ProgramTranslator.get_instance()._inplace_map

#     def new_true_fn():
#         # init args may contain mutable python container like [var, 2], we copy then like in while_loop
#         inplace_map_checkpoint = inplace_map.save_checkpoint()
#         helper.set(
#             return_name_ids,
#             paddle.utils.copy_mutable_vars(init_args),
#         )
#         ret = true_fn()
#         # IfExpr will return a non-None return value, so we just return ret.
#         # We assume normal return has no return value.
#         if ret is None:
#             ret = helper.get(return_name_ids)
#         inplace_map.restore_checkpoint(inplace_map_checkpoint)
#         return ret

#     def new_false_fn():
#         # init args may contain mutable python container like [var, 2], we copy then like in while_loop
#         inplace_map_checkpoint = inplace_map.save_checkpoint()
#         helper.set(
#             return_name_ids,
#             paddle.utils.copy_mutable_vars(init_args),
#         )
#         ret = false_fn()
#         if ret is None:
#             ret = helper.get(return_name_ids)
#         inplace_map.restore_checkpoint(inplace_map_checkpoint)
#         return ret

#     try:
#         cond_outs = paddle.static.nn.cond(
#             pred, new_true_fn, new_false_fn, None, return_name_ids
#         )
#     except Exception as e:
#         if re.search(
#             "Unsupported return type of true_fn and false_fn in cond", str(e)
#         ):
#             raise Dygraph2StaticException(
#                 "Your if/else have different return type. TODO: add link to modifty. {}".format(
#                     str(e)
#                 )
#             )
#         if re.search("Incompatible return values of", str(e)):
#             raise Dygraph2StaticException(
#                 "Your if/else have different number of return value. TODO: add link to modifty. {}".format(
#                     str(e)
#                 )
#             )
#         raise e
#     get_args = lambda: helper.get(return_name_ids)
#     set_args = lambda vs: helper.set(return_name_ids, vs)
#     return _recover_args_state(cond_outs, get_args, set_args, return_name_ids)


def _run_py_ifelse(
    pred, true_fn, false_fn, get_args, set_args, return_name_ids
):
    """
    Evaluate python original branch function if-else.
    """
    py_outs = true_fn() if pred else false_fn()
    return py_outs


# def _remove_no_value_return_var(out):
#     if isinstance(out, tuple) and len(out) > 0:
#         processed_out = out
#         align_ret = out[0]
#         if isinstance(align_ret, tuple):
#             for index, item in enumerate(align_ret):
#                 if isinstance(item, Variable) and (
#                     RETURN_NO_VALUE_VAR_NAME in item.name
#                 ):
#                     # return None
#                     if index == 0:
#                         processed_out = (None,) + out[1:]
#                     elif index == 1:
#                         processed_out = align_ret[:1] + out[1:]
#                     else:
#                         processed_out = (align_ret[:index],) + out[1:]
#                     break

#         for index, item in enumerate(processed_out):
#             if isinstance(item, Variable) and (
#                 RETURN_NO_VALUE_VAR_NAME in item.name
#             ):
#                 processed_out = processed_out[:index]

#         if not processed_out:
#             return None
#         elif len(processed_out) == 1:
#             return processed_out[0]
#         else:
#             return processed_out

#     else:
#         return out


# def _check_no_undefined_var(outs, names, branch_name):
#     if names is None:
#         return
#     if not isinstance(outs, (list, tuple)):
#         outs = [outs]
#     for var, name in zip(list(outs), names):
#         if isinstance(var, UndefinedVar):
#             raise ValueError(
#                 "Required '{}' must be initialized both in if-else branch, but found it not initialized in '{}'.".format(
#                     name, branch_name
#                 )
#             )


# def _recover_args_state(outs, get_args, set_args, return_name_ids):
#     """
#     Currently we support variant length of early return statement by padding
#     _no_return_value.

#     # TODO(dev): We shall consider to evaluate whether should support this for Python if-else?
#     """
#     # IfExpr's return_name_ids maybe None
#     if return_name_ids is None:
#         return outs

#     init_args = get_args()
#     # recover args state
#     num_outs = len(return_name_ids)
#     num_args = len(init_args)
#     assert num_outs <= num_args

#     if num_args == 1:
#         final_outs = (
#             (outs,) if not isinstance(outs, (list, tuple)) else tuple(outs)
#         )
#     else:
#         outs = (outs,) if num_outs == 1 else tuple(outs)
#         final_outs = outs + init_args[num_outs:]

#     set_args(final_outs)
#     return final_outs


def convert_len(var):
    """
    Returns variable(length) from shape ops based on var.type

    Note: In addition to some ast transformations, some block-related
          operations are added in `len` transformation, such as appending
          `shape_op` in var.block.
    """
    if isinstance(var, Proxy):
        return var.tracer.create_proxy("call_function", len, (var,), {})
    else:
        if isinstance(var, VariableTuple):
            return var.__len__()
        return len(var)


def convert_zip(*args):
    for i, arg in enumerate(args):
        if isinstance(arg, Proxy) and arg.shape[0] == -1:
            raise RuntimeError(
                "Not support zip(tensor, ...) when tensor.shape[0] == -1, "
                "but found args[{}].shape[0] == -1 in 'zip'".format(str(i))
            )
    return zip(*args)


class VariableTuple:
    """
    this class will cause enumerate can't be wrapped by other iterator change function.
    this will be fixed when list<Variable> is producted.
    VariableTuple can only deal with variables which is fixed.
    """

    def __init__(self, var, start=0):
        self.var = var
        self.len = convert_len(var)
        if isinstance(self.len, Proxy):
            self.rag = ivy.arange(start, start + self.len, 1, ivy.int64)
        else:
            self.rag = range(start, start + self.len)

    def __getitem__(self, idx):
        return self.rag[idx], self.var[idx]

    def __len__(self):
        return self.len


def convert_enumerate(*args):
    has_variable = any(isinstance(x, Proxy) for x in args)
    if has_variable:
        return VariableTuple(*args)
    return enumerate(*args)


def convert_range(*args):
    has_variable = any(isinstance(x, Proxy) for x in args)
    if has_variable:
        if len(args) == 1:
            return ivy.arange(0, args[0], 1, ivy.int64)
        if len(args) == 2:
            return ivy.arange(args[0], args[1], 1, ivy.int64)
        if len(args) == 3:
            return ivy.arange(args[0], args[1], args[2], ivy.int64)
    return range(*args)


def convert_shape(x):
    """
    A function representation of the shape of variable.
    """

    def has_negative(list_shape):
        return any(x < 0 for x in list_shape)

    # When `x` is Variable:
    #  (1) if x.shape contains -1, such as [2, -1, 64], returns [2, var, 64],
    #      where var = paddle.shape(x)[1]

    #  (2) if x.shape does not contains -1, return lsit(x.shape) directly

    if isinstance(x, Proxy):
       x.tracer.create_proxy("call_function", getattr, (x,), {})
    else:
        return x.shape


# def convert_shape_compare(left, *args):
#     """
#     A function handles comparison difference between Paddle and Python.
#     For example, if x and y are Tensors, x.shape == y.shape will return single
#     boolean Value (True/False). However, paddle.shape(x) == paddle.shape(y) is
#     an element-wise comparison. The difference can cause dy2stat error. So we
#     create this function to handle the difference.

#     Args:
#         left: variable
#         *args: compare_op(str), variable, compare_op(str), variable, where
#             compare_op means "<", ">", "==", "!=", etc.
#     Returns:
#         If the variables to compare are NOT Paddle Variables, we will return as
#         Python like "a op1 b and b op2 c and ... ".
#         If the variables to compare are Paddle Variables, we will do elementwise
#         comparsion first and then reduce to a boolean whose numel is 1.

#     """
#     args_len = len(args)
#     assert (
#         args_len >= 2
#     ), "convert_shape_compare needs at least one right compare variable"
#     assert (
#         args_len % 2 == 0
#     ), "Illegal input for convert_shape_compare, *args should be op(str), var, op(str), var ..."
#     num_cmp = args_len // 2
#     if isinstance(left, Variable):

#         def reduce_compare(x, op_str, y):
#             element_wise_result = eval("x " + op_str + " y")
#             if op_str == "!=":
#                 return paddle.any(element_wise_result)
#             elif (
#                 op_str == "is"
#                 or op_str == "is not"
#                 or op_str == "in"
#                 or op_str == "not in"
#             ):
#                 return element_wise_result
#             else:
#                 return paddle.all(element_wise_result)

#         final_result = reduce_compare(left, args[0], args[1])
#         for i in range(1, num_cmp):
#             cmp_left = args[i * 2 - 1]
#             cmp_op = args[i * 2]
#             cmp_right = args[i * 2 + 1]
#             cur_result = reduce_compare(cmp_left, cmp_op, cmp_right)
#             final_result = convert_logical_and(
#                 lambda: final_result, lambda: cur_result
#             )
#         return final_result
#     else:
#         cmp_left = left
#         final_result = None
#         for i in range(num_cmp):
#             cmp_op = args[i * 2]
#             cmp_right = args[i * 2 + 1]
#             cur_result = eval("cmp_left " + cmp_op + " cmp_right")
#             if final_result is None:
#                 final_result = cur_result
#             else:
#                 final_result = final_result and cur_result

#             if final_result is False:
#                 return False
#             cmp_left = cmp_right
#         return final_result


# def cast_bool_if_necessary(var):
#     assert isinstance(var, Proxy)
#     if convert_dtype(var.dtype) not in ['bool']:
#         var = paddle.cast(var, dtype="bool")
#     return var


# def convert_var_dtype(var, dtype):
#     if isinstance(var, Variable):
#         src_dtype = convert_dtype(var.dtype)
#         assert src_dtype in [
#             'bool',
#             'float16',
#             'float32',
#             'float64',
#             'int32',
#             'int64',
#             'uint8',
#         ], "The dtype of var {} is {}, which is not supported in the cast op.".format(
#             var.name, src_dtype
#         )
#         assert dtype in [
#             'bool',
#             'int',
#             'float',
#         ], "The casted target dtype is {}, which is not supported in type casting.".format(
#             dtype
#         )
#         cast_map = {
#             'bool': 'bool',
#             'int': 'int32',
#             'float': 'float32',
#         }
#         return paddle.cast(var, dtype=cast_map[dtype])
#     else:
#         return eval(f'{dtype}(var)')


def convert_assert(cond, message=""):
    """
    A function representation of a Python ``assert`` statement.
    """
    if isinstance(cond, Proxy):
        cond = ivy.cast_bool(cond)
        # TODO: add an ivy.Assert function in the ivy API.

        #return ivy.Assert(cond)
    else:
        assert cond, message


def convert_print(*objects, sep=' ', end='\n', file=None, flush=False):
    """
    A function representing Python ``print`` function. It will print all arguments
    at compile time and only print the Tensor values at runtime.
    """
    for obj in objects:
        if isinstance(obj, Proxy):
            #TODO: add an ivy.Print function in the ivy API
            #ivy.Print(obj)
            pass
    print(*objects, sep=sep, end=end, file=file, flush=flush)


# def convert_pop(target, *args):
#     """
#     A function representation of a Python pop statement for a list or dict.

#     Args:
#         target(list|dict|Tensor): A variable to pop item from.
#         *args(tuple): index or default value to parse.

#     Returns:
#         A item poped from target.
#     """

#     is_variable = isinstance(target, Variable)
#     if is_variable:
#         is_tensor_array = target.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY

#     if is_variable and is_tensor_array:
#         return _run_paddle_pop(target, *args)
#     else:
#         return _run_python_pop(target, *args)


# def _run_paddle_pop(array, *args):
#     if len(args) == 0:
#         idx = -1
#     else:
#         idx = args[0]

#     assert isinstance(idx, int)

#     def cond(i, new_array):
#         return paddle.less_than(i, arr_len)

#     def body(i, new_array):
#         item = paddle.tensor.array_read(array=array, i=i)
#         paddle.tensor.array_write(
#             item, paddle.tensor.array_length(new_array), new_array
#         )

#         i = paddle.increment(i)
#         return i, new_array

#     arr_len = paddle.tensor.array_length(array)
#     if idx < 0:
#         idx = idx + arr_len
#     else:
#         from paddle.tensor import fill_constant

#         idx = fill_constant(shape=[1], dtype="int64", value=idx)

#     pop_item = paddle.tensor.array_read(array, idx)

#     tmp = paddle.assign(array)
#     new_array = _slice_tensor_array(tmp, 0, idx)
#     i = idx + 1
#     from paddle.static.nn import while_loop

#     _, new_array = while_loop(cond, body, [i, new_array])
#     paddle.assign(new_array, output=array)

#     return pop_item


# # TODO(liym27): A better way to slice tensor array.
# #  Maybe support start == end for slice op.
# def _slice_tensor_array(array, start, end):
#     def true_fn():
#         null_array = paddle.tensor.create_array("float32")
#         return null_array

#     def false_fn(array, start, end):
#         new_array = paddle.slice(array, starts=[start], ends=[end], axes=[0])
#         return new_array

#     new_array = paddle.static.nn.cond(
#         start == end, true_fn, lambda: false_fn(array, start, end)
#     )
#     return new_array


# def _run_python_pop(target, *args):
#     # 1. pop for a dict
#     if len(args) == 2:
#         idx, default = args
#         return target.pop(idx, default)

#     # 2. pop for a list or dict
#     else:
#         idx = args[0] if args else -1
#         return target.pop(idx)

# -------------------
# Logical operators
# -------------------


def bool_or(val1, val2):
    return val1 or val2


def bool_and(val1, val2):
    return val1 and val2


def unary_not(val):
    return not val

# -------------------
# Comparison operators
# -------------------


def cmp_eq(val1, val2):
    return val1 == val2


def cmp_ne(val1, val2):
    return val1 != val2


def cmp_gt(val1, val2):
    return val1 > val2


def cmp_ge(val1, val2):
    return val1 >= val2


def cmp_lt(val1, val2):
    return val1 < val2


def cmp_le(val1, val2):
    return val1 <= val2


def cmp_is(inp, val):
    return inp is val


def cmp_isnot(inp, val):
    return inp is not val


def cmp_in(inp, iterable):
    return inp in iterable


def cmp_notin(inp, iterable):
    return inp not in iterable

# --------------------
# Builtin type casts
# --------------------

def cast_to_bool(inp):
    return bool(inp)

def cast_to_int(inp):
    return int(inp)

def cast_to_float(inp):
    return float(inp)

def cast_to_dict(inp):
    return dict(inp)

def cast_to_tuple(inp):
    return tuple(inp)

def cast_to_set(inp):
    return set(inp)

def cast_to_list(inp):
    return list(inp)

def convert_to_f_string(func, *args):
    # map the operator to the corresponding symbol
    op_map = {
        "bool_or": "or",
        "bool_and": "and",
        "unary_not": "not",
        "cmp_eq": "==",
        "cmp_ne": "!=",
        "cmp_gt": ">",
        "cmp_ge": ">=",
        "cmp_lt": "<",
        "cmp_le": "<=",
        "cmp_is": "is",
        "cmp_isnot": "is not",
        "cmp_in": "in",
        "cmp_notin": "not in",
        "cast_to_bool": "bool",
        "cast_to_int": "int",
        "cast_to_float": "float",
        "cast_to_dict": "dict",
        "cast_to_list": "list",
        "cast_to_tuple": "tuple",
        "cast_to_set": "set",
    }
    op = op_map[func.__name__]

    # check if the function is a cast operation
    if func.__name__.startswith("cast_to_"):
        # construct the f-string for the cast operation
        f_string = f"{op}("
        for arg in args:
            # check if the argument is a list or tuple
            if isinstance(arg, (list, tuple)):
                # unpack the elements and join them with the operator
                f_string += f"{f_string}".join(str(e) for e in arg)
            else:
                # use the argument as it is
                f_string += str(arg)
        f_string += ")"
    else:
        # construct the f string for the boolean operators
        f_string = "("
        for arg in args:
            # check if the argument is a list or tuple
            if isinstance(arg, (list, tuple)):
                # unpack the elements and join them with the operator
                f_string += f" {op} ".join(str(e) for e in arg)
            else:
                # use the argument as it is
                f_string += str(arg)
            # add the operator between arguments
            f_string += f" {op} "
        # remove the extra operator at the end
        f_string = f_string[: -len(op) - 2]
        f_string += ")"
    return f_string

transform_funcs = {
    "bool_or": bool_or,
    "bool_and": bool_and,
    "unary_not": unary_not,
    "cmp_eq": cmp_eq,
    "cmp_ne": cmp_ne,
    "cmp_gt": cmp_gt,
    "cmp_ge": cmp_ge,
    "cmp_lt": cmp_lt,
    "cmp_le": cmp_le,
    "cmp_is": cmp_is,
    "cmp_isnot": cmp_isnot,
    "cmp_in": cmp_in,
    "cmp_notin": cmp_notin,
    "cast_to_bool": cast_to_bool,
    "cast_to_int": cast_to_int,
    "cast_to_float": cast_to_float,
    "cast_to_dict": cast_to_dict,
    "cast_to_list": cast_to_list,
    "cast_to_tuple": cast_to_tuple,
    "cast_to_set": cast_to_set
}
