############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

__all__ = []

import inspect
import pkgutil
import sys

globals()["__data_providers"] = {}
__all__.append("__data_providers")
globals()["__higra_global_cache"] = None
__all__.append("__higra_global_cache")


def __import_module(loader, name):
    if name not in sys.modules:
        module = loader.find_module(name).load_module(name)

        for name, value in inspect.getmembers(module):
            if name.startswith('__'):
                continue

            globals()[name] = value
            __all__.append(name)

modules = {}
for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    modules[name] = loader

# list of modules that have to be loaded before the others
# (required for correct file parsing, eg. defining top level decorators)
pre_load = ("higram", "data_cache", "concept", "hg_utils")

for name in pre_load:
    __import_module(modules[name], name)
    del modules[name]

_data_cache__init()

for name in modules:
    __import_module(modules[name], name)

def __logger_printer(m):
    print(m)
