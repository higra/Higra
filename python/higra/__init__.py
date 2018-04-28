__all__ = []

import pkgutil
import inspect
import sys

globals()["__data_providers"] = {}
__all__.append("__data_providers")
globals()["__higra_global_cache"] = None
__all__.append("__higra_global_cache")


for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    if name not in sys.modules:
        module = loader.find_module(name).load_module(name)

        for name, value in inspect.getmembers(module):
            if name.startswith('__'):
                continue

            globals()[name] = value
            __all__.append(name)

_data_cache__init()
