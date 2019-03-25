from functools import wraps

class inputDecorators:
    @staticmethod
    def non_null(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert all(not a is None for a in args), "Supplied arguments contain a null (None) value."
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def non_zero(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert all(a != 0 for a in args)
            return func(*args, **kwargs)
        
        return wrapper
    
    @staticmethod
    def all_positive(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert all(a > 0 for a in args)
            return func(*args, **kwargs)
        
        return wrapper