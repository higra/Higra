from higra.data_cache import _higra_global_cache
from functools import wraps
import sys

__attributes_providers = {}


def getAttribute(key, attributeName):
    return _higra_global_cache.getData(key)[attributeName]


def setAttribute(key, attributeName, attribute):
    _higra_global_cache.getData(key)[attributeName] = attribute


def HGAttribute(defaultName, dependencies=()):
    def decorator(fun):

        @wraps(fun)
        def wrapper(tree, *args, **kwargs):
            attributeName = kwargs.pop("attributeName", defaultName)
            forceRecompute = kwargs.pop("forceRecompute", False)
            dataCache = kwargs.pop("dataCache", _higra_global_cache)
            treeCache = dataCache.getData(tree)

            if attributeName not in treeCache or forceRecompute:

                for depName in dependencies:
                    # print("Resolving dependency ", depName)
                    if depName in kwargs:  # if user has provided an explicit initialization for current dependency
                        providedDep = kwargs[depName]
                        # print("Found in args ", providedDep)
                        if isinstance(providedDep, str):  # if it is an attribute name
                            if providedDep in treeCache:  # look in the cache
                                # print("Use cache ", treeCache[providedDep])
                                kwargs[depName] = treeCache[providedDep]
                            elif providedDep in __attributes_providers:  # look in providers
                                # print("Use provider ")
                                kwargs[depName] = __attributes_providers[providedDep](tree)
                            else:
                                err = "In computation of attribute " + defaultName + ". Dynamic resolution of dependency '" + depName + " with user specified value '" + providedDep + "' failed (no cache data or attribute provider with this name)."
                                raise Exception("Could not compute Attribute. " + err)
                    else:
                        if depName in treeCache:  # look in the cache
                            kwargs[depName] = treeCache[depName]
                            # print("Use cache ", treeCache[depName])
                        elif depName in __attributes_providers:  # look in providers
                            # print("Use provider ")
                            kwargs[depName] = __attributes_providers[depName](tree)
                        else:
                            kwargs[depName] = None
                            err = "In computation of attribute " + defaultName + ". Dynamic resolution of dependency '" + depName + "' failed (no cache data or attribute provider with this name)."
                            raise Exception("Could not compute Attribute. " + err)
                # print("Computing attr ")
                treeCache[attributeName] = fun(tree, *args, **kwargs)
            # print("Return cache ")
            return treeCache[attributeName]

        wrapper.attributeName = defaultName
        wrapper.original = fun

        if defaultName in __attributes_providers:
            print("Warning, an attribute with the same name was already defined: ", defaultName, file=sys.stderr)

        __attributes_providers[defaultName] = wrapper

        return wrapper

    return decorator
