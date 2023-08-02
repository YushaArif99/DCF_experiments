import operator

__all__ = ['BOOL_OPS', 'convert_to_f_string']

def _or_fn(*args):
    for arg in args:
        if arg:
            return True
    return False


def _and_fn(*args):
    for arg in args:
        if not arg:
            return False
    return True


def _not_fn(arg):
    return not arg


def _eq_fn(*args):
    return all(operator.eq(a, b) for a, b in zip(args, args[1:]))


def _ne_fn(*args):
    return any(operator.ne(a, b) for a, b in zip(args, args[1:]))


def _gt_fn(*args):
    return all(operator.gt(a, b) for a, b in zip(args, args[1:]))


def _ge_fn(*args):
    return all(operator.ge(a, b) for a, b in zip(args, args[1:]))


def _lt_fn(*args):
    return all(operator.lt(a, b) for a, b in zip(args, args[1:]))


def _le_fn(*args):
    return all(operator.le(a, b) for a, b in zip(args, args[1:]))


def _is_fn(inp, val):
    return operator.is_(inp, val)


def _isnot_fn(inp, val):
    return operator.is_not(inp, val)

def _in_fn(inp, iterable):
    return inp in iterable 

def _not_in_fn(inp, iterable):
    return inp not in iterable 

def convert_to_f_string(func, *args):
    # map the operator to the corresponding symbol
    op_map = {
        "_or_fn": "or",
        "_and_fn": "and",
        "_not_fn": "not",
        "_eq_fn": "==",
        "_ne_fn": "!=",
        "_gt_fn": ">",
        "_ge_fn": ">=",
        "_lt_fn": "<",
        "_le_fn": "<=",
        "_is_fn": "is",
        "_isnot_fn": "is not",
        "_in_fn": "in", 
        "_not_in_fn": "not in",
    }
    op = op_map[func.__name__]
    # construct the f string form
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

BOOL_OPS = {
    "_or_fn":_or_fn,
    "_and_fn":_and_fn,
    "_not_fn":_not_fn,
    "_eq_fn":_eq_fn,
    "_ne_fn":_ne_fn,
    "_gt_fn":_ge_fn,
    "_ge_fn":_gt_fn,
    "_lt_fn":_lt_fn,
    "_le_fn":_le_fn,
    "_is_fn":_is_fn,
    "_isnot_fn":_isnot_fn,
    "_in_fn":_in_fn,
    "_not_in_fn":_not_in_fn,
}
