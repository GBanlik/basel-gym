from functools import wraps

class inputDecorators:
    
    @staticmethod
    def non_null(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not all(not a is None for a in args):
                raise ValueError("{}:{}: Supplied arguments contain at least a null (None) value: {}, {}".format(func.__class__.__name__, func, *args, **kwargs))
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def non_zero(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not all(a != 0 for a in args): 
                raise ValueError("{}:{}: Supplied arguments contain at least a zero value: {}, {}".format(func.__class__.__name__, func, *args, **kwargs))
            return func(*args, **kwargs)
        
        return wrapper
    
    @staticmethod
    def all_positive(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not all(a > 0 for a in args):
                raise ValueError("{class_name}:{func_name}: Supplied arguments contain a non-positive value: {args}, {kwargs}.".format(func.__class__.__name__, func, *args, **kwargs))
            return func(*args, **kwargs)
        
        return wrapper