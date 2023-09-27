def indexable(x, code=None):
    if hasattr(x, '__iter__'):
        return list(x)
    elif hasattr(x, '__len__') and hasattr(
        x, '__getitem__'
    ):  # used for customed type and non-iterable type.
        return x
    else:
        raise RuntimeError("X can't be convert into indexable.")


def unpack_by_structure(target, structure):
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


def convert_len(var):
    return len(var)


def convert_zip(*args):
    return zip(*args)


def convert_enumerate(*args):
    return enumerate(*args)


def convert_range(*args):
    return range(*args)


def convert_assert(cond, message=""):
    assert cond, message


def convert_print(*objects, sep=' ', end='\n', file=None, flush=False):
    """
    A function representing Python ``print`` function. It will print all arguments
    at compile time and only print the Tensor values at runtime.
    """
    print(*objects, sep=sep, end=end, file=file, flush=flush)

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
