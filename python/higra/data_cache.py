############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import weakref
import functools
import sys
import inspect
import higra as hg


def _data_cache__init():
    hg.__higra_global_cache = DataCache()


class DataCache:

    def __init__(self):
        self.__data = weakref.WeakKeyDictionary()

    def get_data(self, key):
        return self.__data.setdefault(key, {})

    def clear_data(self, key):
        if key in self.__data:
            del self.__data[key]

    def clear_all_data(self):
        self.__data.clear()


def list_data_providers():
    for p in hg.__data_providers:
        print(hg.__data_providers[p])


def list_attributes(key):
    return list(hg.__higra_global_cache.get_data(key).keys())


def get_attribute(key, attribute_name):
    return hg.__higra_global_cache.get_data(key).get(attribute_name, None)


def set_attribute(key, attribute_name, attribute):
    hg.__higra_global_cache.get_data(key)[attribute_name] = attribute


def clear_attributes(key, *attribute_name):
    if not attribute_name:
        hg.__higra_global_cache.clear_data(key)
    else:
        obj_cache = hg.__higra_global_cache.get_data(key)
        for k in obj_cache:
            del obj_cache[k]


def clear_all_attributes():
    hg.__higra_global_cache.clear_all_data()


class DataProvider:

    def __init__(self, name, fun, description):
        self.name = name
        self.fun = fun
        self.description = description

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

    def __str__(self):
        if self.description != "":
            return self.name + ": " + self.description
        return self.name


def data_provider(name, description=""):

    def decorator(fun):

        @functools.wraps(fun)
        def wrapper(obj, *args, **kwargs):
            data_name = kwargs.pop("attribute_name", name)
            force_recompute = kwargs.pop("force_recompute", False)
            data_cache = kwargs.pop("data_cache", hg.__higra_global_cache)
            obj_cache = data_cache.get_data(obj)

            if data_name not in obj_cache or force_recompute:
                obj_cache[data_name] = fun(obj, *args, **kwargs)

            return obj_cache[data_name]

        wrapper.name = name
        wrapper.original = fun

        if name in hg.__data_providers:
            print("Warning, a data provider with the same name was already defined: ", name, file=sys.stderr)

        hg.__data_providers[name] = DataProvider(name, wrapper, description)

        return wrapper

    return decorator


def __cache_lookup(obj, dep_path, data_cache):
    name, *tail = dep_path.split('.', maxsplit=1)
    obj_cache = data_cache.get_data(obj)
    name_data = None
    if name in obj_cache:
        name_data = obj_cache[name]
    elif name in hg.__data_providers:  # look in providers
        name_data = hg.__data_providers[name](obj)
    else:
        err = "Dynamic resolution of dependency '" + name + "' failed (no cache data or attribute provider with this name)."
        raise Exception(err)

    if len(tail) > 0:
        name_data = __cache_lookup(name_data, tail[0], data_cache)

    return name_data


def __resolve_dependency(obj, dep_name, dep_path, data_cache, kwargs):
    # if user has provided an explicit initialization for current dependency
    if dep_name in kwargs:
        provided_dep = kwargs[dep_name]
        # if user has provided a path
        if isinstance(provided_dep, str):
            # restart dependency resolution with new path
            del kwargs[dep_name]
            __resolve_dependency(obj, dep_name, provided_dep, data_cache, kwargs)
        # else use provided_dep as depName argument value
    else:
        kwargs[dep_name] = __cache_lookup(obj, dep_path, data_cache)


def __transfer_to_kw_arguments(signature, args, kwargs):
    nargs = list(args)
    for p in signature.parameters.values():
        if len(nargs) == 0:
            break
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            kwargs[p.name] = nargs[0]
            del nargs[0]
    return nargs


def data_consumer(*dependencies, **dependencies_path):
    def decorator(fun):

        signature = inspect.signature(fun)

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            obj = args[0]
            args = __transfer_to_kw_arguments(signature, args, kwargs)

            data_cache = kwargs.pop("data_cache", hg.__higra_global_cache)

            for dep_name in dependencies:
                __resolve_dependency(obj, dep_name, dep_name, data_cache, kwargs)

            for dep_name in dependencies_path:
                __resolve_dependency(obj, dep_name, dependencies_path[dep_name], data_cache, kwargs)

            return fun(*args, **kwargs)

        wrapper.original = fun

        return wrapper

    return decorator
