import weakref
import functools
import sys
import inspect

class DataCache:

    def __init__(self):
        self.__data = weakref.WeakKeyDictionary()

    def getData(self, key):
        return self.__data.setdefault(key, {})

    def clearData(self, key):
        if key in self.__data:
            del self.__data[key]

    def clearAllData(self):
        self.__data.clear()


__higra_global_cache = DataCache()

__data_providers = {}


def getAttribute(key, attributeName):
    return __higra_global_cache.getData(key).get(attributeName, None)


def setAttribute(key, attributeName, attribute):
    __higra_global_cache.getData(key)[attributeName] = attribute


def clearAttributes(key, *attributeName):
    if not attributeName:
        __higra_global_cache.clearData(key)
    else:
        objCache = __higra_global_cache.getData(key)
        for k in objCache:
            del objCache[k]


def clearAllAttributes():
    __higra_global_cache.clearAllData()


class DataProvider:

    def __init__(self, name, fun, description):
        self.name = name
        self.fun = fun
        self.description = description

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


def dataProvider(name, description=""):
    def decorator(fun):

        @functools.wraps(fun)
        def wrapper(obj, *args, **kwargs):
            dataName = kwargs.pop("attributeName", name)
            forceRecompute = kwargs.pop("forceRecompute", False)
            dataCache = kwargs.pop("dataCache", __higra_global_cache)
            objCache = dataCache.getData(obj)

            if dataName not in objCache or forceRecompute:
                objCache[dataName] = fun(obj, *args, **kwargs)

            return objCache[dataName]

        wrapper.name = name
        wrapper.original = fun

        if name in __data_providers:
            print("Warning, a data provider with the same name was already defined: ", name, file=sys.stderr)

        __data_providers[name] = DataProvider(name, wrapper, description)

        return wrapper

    return decorator


def __cacheLookUp(obj, depPath, dataCache):
    name, *tail = depPath.split('.', maxsplit=1)
    objCache = dataCache.getData(obj)
    nameData = None
    if name in objCache:
        nameData = objCache[name]
    elif name in __data_providers:  # look in providers
        nameData = __data_providers[name](obj)
    else:
        err = "Dynamic resolution of dependency '" + name + "' failed (no cache data or attribute provider with this name)."
        raise Exception(err)

    if len(tail) > 0:
        nameData = __cacheLookUp(nameData, tail[0], dataCache)

    return nameData


def __resolveDependency(obj, depName, depPath, dataCache, kwargs):
    # if user has provided an explicit initialization for current dependency
    if depName in kwargs:
        providedDep = kwargs[depName]
        # if user has provided a path
        if isinstance(providedDep, str):
            # restart dependency resolution with new path
            del kwargs[depName]
            __resolveDependency(obj, depName, providedDep, dataCache, kwargs)
        # else use providedDep as depName argument value
    else:
        kwargs[depName] = __cacheLookUp(obj, depPath, dataCache)


def __transferToKwArguments(signature, args, kwargs):
    nargs = list(args)
    for p in signature.parameters.values():
        if len(nargs) == 0:
            break;
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            kwargs[p.name] = nargs[0]
            del nargs[0]
    return nargs


def dataConsumer(*dependencies, **dependenciesPath):
    def decorator(fun):

        signature = inspect.signature(fun)

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            obj = args[0]
            args = __transferToKwArguments(signature, args, kwargs)

            dataCache = kwargs.pop("dataCache", __higra_global_cache)

            for depName in dependencies:
                __resolveDependency(obj, depName, depName, dataCache, kwargs)

            for depName in dependenciesPath:
                __resolveDependency(obj, depName, dependenciesPath[depName], dataCache, kwargs)

            return fun(*args, **kwargs)

        wrapper.original = fun

        return wrapper

    return decorator
