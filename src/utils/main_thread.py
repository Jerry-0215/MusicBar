import objc
from PyObjCTools import AppHelper

def call_on_main_thread(func):
    """Decorator to ensure a function is called on the main thread."""
    @objc.python_method
    def wrapper(*args, **kwargs):
        # The first argument is 'self'
        instance = args[0]
        # Create a function with arguments baked in
        partial_func = lambda: func(*args, **kwargs)
        AppHelper.callAfter(partial_func)
    return wrapper 