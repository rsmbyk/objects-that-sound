def display_params(func):
    def decorator(*args, **kwargs):
        print('args:', args)
        print('kwargs:', kwargs)
        func(*args, **kwargs)
    decorator.__name__ = func.__name__
    return decorator
