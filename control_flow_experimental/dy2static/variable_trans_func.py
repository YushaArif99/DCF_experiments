from .helpers import gast

def create_undefined_var(name):
    func_code = f"{name} = None"
    return gast.parse(func_code).body[0]



def create_bool_node(name, value):
    '''
    Create a assign stmt for name = value .
    '''
    assert isinstance(value, bool)
    node = f"{name} = {value}"
    return gast.parse(node).body[0]
