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


###########################################################
#                                                         #
#                   DATA CACHE CLASS                      #
#                                                         #
###########################################################

class _WeakKeyDictionaryIterator:

    def __init__(self, wkd):
        self.wkd = wkd
        self.wkd_iter = iter(wkd._data)

    def __iter__(self):
        return self

    def __next__(self):
        k = self.wkd_iter.__next__()
        ref, value = self.wkd._data[k]
        return ref(), value


class WeakKeyDictionary:
    """
    Useful weak key dictionary (compared to the one provided by python...)

    Uses solely object id as key.
    """

    def __init__(self):
        self._data = {}

    def __get__(self, obj, owner):
        _, val = self._data[id(obj)]
        return val

    def __set__(self, obj, value):
        key = id(obj)
        try:
            ref, _ = self._data[key]
        except KeyError:
            def on_destroy(_):
                try:
                    del self._data[key]
                except:
                    pass

            ref = weakref.ref(obj, on_destroy)
        self._data[key] = ref, value

    def __delete__(self, obj):
        del self._data[id(obj)]

    def setdefault(self, obj, default=None):
        """
        If obj is in the dictionary, return its value.
        If not, insert key with a value of default and return default.

        :param default:
        :return:
        """
        key = id(obj)
        try:
            _, val = self._data[key]
            return val
        except KeyError:
            def on_destroy(_):
                try:
                    del self._data[key]
                except:
                    pass

            ref = weakref.ref(obj, on_destroy)
            self._data[key] = ref, default
            return default

    def clear(self):
        self._data.clear()

    def __iter__(self):
        return _WeakKeyDictionaryIterator(self)


def _data_cache__init():
    hg.__higra_global_cache = DataCache()


class DataCache:

    def __init__(self):
        self.__data = WeakKeyDictionary()

    def get_data(self, key):
        return self.__data.setdefault(key, {})

    def clear_data(self, key):
        if key in self.__data:
            del self.__data[key]

    def clear_all_data(self):
        self.__data.clear()

    def __iter__(self):
        return iter(self.__data)


def list_data_providers():
    for p in hg.__data_providers:
        print(hg.__data_providers[p])


def list_attributes(key):
    try:
        return list(hg.__higra_global_cache.get_data(key).keys())
    except TypeError:
        return ()


def get_attribute(key, attribute_name):
    try:
        return hg.__higra_global_cache.get_data(key).get(attribute_name, None)
    except TypeError:
        return None


def set_attribute(key, attribute_name, attribute):
    hg.__higra_global_cache.get_data(key)[attribute_name] = attribute


def get_tags(key):
    return hg.__higra_global_cache.get_data(key).setdefault("__tags__", set())


def add_tag(key, tag):
    tags = get_tags(key)
    tags.add(tag)


def remove_tag(key, tag):
    tags = get_tags(key)
    tags.remove(tag)


def has_tag(key, tag):
    tags = get_tags(key)
    return tag in tags


def clear_attributes(key, *attribute_name):
    if not attribute_name:
        hg.__higra_global_cache.clear_data(key)
    else:
        obj_cache = hg.__higra_global_cache.get_data(key)
        for k in obj_cache:
            del obj_cache[k]


def clear_all_attributes():
    hg.__higra_global_cache.clear_all_data()


###########################################################
#                                                         #
#                  UTILITY FOR SIGNATURES                 #
#                                                         #
###########################################################

def __check_valid_signature(signature):
    """
    Check if function signature contains only positional or keyword arguments

    :param signature:
    :return:
    """
    for p in signature.parameters.values():
        if p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise Exception("Can only handle simple functions, ie with only position and keyword parameters.")


def __transfer_to_kw_arguments(signature, args, kwargs):
    """
    Transfer positional arguments to keyword arguments.

    ``__check_valid_signature(signature)`` must be True

    :param signature:
    :param args:
    :param kwargs:
    :return:
    """
    nargs = list(args)
    for p in signature.parameters.values():
        if len(nargs) == 0:
            break
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            kwargs[p.name] = nargs[0]
            del nargs[0]
    return nargs


def __add_default_parameter(signature, args, kwargs):
    """
    Add default parameters present in the signature that are not already in :attr:`kwargs`

    :param signature:
    :param args:
    :param kwargs:
    :return:
    """
    for p in signature.parameters.values():
        if p.name not in kwargs and p.default is not p.empty:
            kwargs[p.name] = p.default


###########################################################
#                                                         #
#                  AUTO CACHE DECORATOR                   #
#                                                         #
###########################################################

def set_auto_cache_state(boolean):
    """
    Globally activates or deactivates the caching of :func:`~higra.auto_cache` decorated functions.

    If set to ``True`` (default), an auto cached function will save its results in the object cache of its first argument.
    Any other call to this function with the same arguments will return the result stored in the cache
    instead of recomputing a new result (except if this behaviour is locally overridden with the name arguments
    "force_recompute=True" or "no_cache=True").

    If set to ``False``, provider will behave as normal function: they won't try to cache results.

    :See:

    :func:`~higra.get_auto_cache_state`: get current state of the automatic caching.

    :param boolean: ``True`` to globally activate caching for provider, ``False`` to deactivate it
    :return: nothing
    """
    if not isinstance(boolean, type(True)):
        raise TypeError("Parameter must be a bool.")
    hg.__provider_caching = boolean


def get_auto_cache_state():
    """
    Returns the current state of the caching policy for of :func:`~higra.auto_cache` decorated functions.

    :See:

    :func:`~higra.set_auto_cache_state`: modify the state of the automatic caching.

    :return: True if caching of providers result is globally active, False otherwise
    """
    return hg.__provider_caching


# keyword used to store auto cached results in reference object data cache
_auto_cache_keyword = "data_auto_cache"


def clear_auto_cache(*, function_name=None, reference_object=None, data_cache=None):
    """
    Cleanup the result cache for the specified elements

    :param function_name: name of the :func:`~higra.auto_cache` decorated function to clear (if ``None`` all functions
        are cleared)
    :param reference_object: reference object whose cached data have to be cleared (if ``None`` cache data
        of all objects are cleared)
    :param data_cache: data cache to work on (will default to the global Higra cache)
    :return: nothing
    """
    if data_cache is None:
        data_cache = hg.__higra_global_cache

    if function_name is None and reference_object is None:
        for obj, cache in data_cache:
            if _auto_cache_keyword in cache:
                del cache[_auto_cache_keyword]

    elif reference_object is not None and function_name is None:
        cache = data_cache.get_data(reference_object)
        if _auto_cache_keyword in cache:
            del cache[_auto_cache_keyword]

    elif reference_object is None and function_name is not None:
        for obj, cache in data_cache:
            if _auto_cache_keyword in cache:
                cache = cache[_auto_cache_keyword]
                if function_name in cache:
                    del cache[function_name]

    else:
        cache = data_cache.get_data(reference_object)
        if _auto_cache_keyword in cache:
            cache = cache[_auto_cache_keyword]
            if function_name in cache:
                del cache[function_name]


def __hash_combine(h1, h2):
    """
    Combine two hash values to create a new hash value
    :param h1: int
    :param h2: int
    :return: int
    """
    h1 = h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h2 >> 2))
    return h1


def __has_method(o, m):
    """
    Test if a given object has a given method
    :param o: object
    :param m: method name
    :return: True or False
    """
    mm = getattr(o, m, None)
    return callable(mm)


def __make_key(o):
    """
    Computes a hash of the given object
    :param o:
    :return:
    """
    if isinstance(o, int):
        # because for an int x hash(x) == x which is not very usefull
        return hash(str(o))
    elif isinstance(o, (set, tuple, list)):
        if len(o) != 0:
            return functools.reduce(__hash_combine, [__make_key(e) for e in o])
        else:
            return 0x9e3775b2
    elif isinstance(o, dict):
        # not ideal but we use sum to be commutative, i.e. robust to arbitrary ordering of dictionary elements
        keys = sum([__make_key(k) for k in o.keys()])
        values = sum([__make_key(v) for v in o.values()])
        return __hash_combine(keys, values)
    elif __has_method(o, "__hash__"):
        try:
            return hash(o)
        except TypeError:
            pass

    return hash(id(o))


def __make_hash(*args, **kwargs):
    return __hash_combine(__make_key(args), __make_key(kwargs))


def auto_cache(fun):
    """
    Function decorator that provides automatic caching of function results.

    The basic idea is that two successive calls with the same arguments to an *auto_cache* decorated function
    will indeed produce a single function call. The second time, the result of the first call will be returned.

    Auto cache is useful for expensive functions that might be called in unrelated locations (where manual reuse
    of function result would be difficult).

    Auto cache can only decorates functions with only positional and named arguments. The first argument of the function
    must support weak references.

    :Special parameters:

    All functions decorated by ``auto-cache`` support the following named arguments:

        - :attr:`no_cache` (boolean value). If ``True``, auto caching is disabled for this function call: no cached data
          will be used or stored.
        - :attr:`force_recompute` (boolean value). If ``True``, no cache result will be used for this function call but
          the result of the function call will be cached for future use.

    :Caveats:

    A function call is identified by a hash of its arguments. This hash is computed in such a way that non hashable
    objects (except from dictionaries) are indeed identified by their unique object identifier. This means that calling the same auto-cached function with
    the same mutated arguments can potentially lead to unexpected results.

    >>> @hg.auto_cache
    >>> def cached_sum(a):
    >>>     return sum(a)
    >>>
    >>> a = numpy.zeros(10)
    >>> cached_sum(a)
    0
    >>> a[0] = 1 # mutate a
    >>> cached_sum(a) # auto cache cannot detect mutation of a and returns the cached result
    0

    This problem can be solved by using :attr:`no_cache`, :attr:`force_recompute` (see above) or by globally disabling
    function caching (see :func:`~set_auto_cache_state`).

    >>> cached_sum(a, no_cache=True) # or force_recompute=True
    1

    If you are really unlucky, it might also appends that two unrelated set of arguments get the same hash. In such case
    the auto cache decorator will consider that they are the same. This is however extremely unlikely to happen.

    :Life Time:

    The data stored in the cache are associated to the first argument of the function called the *reference object*.
    All data related to a reference object is cleared when it gets garbage collected.

    The cache data associated to a particular function or object can be manually cleared with
    :func:`~higra.clear_auto_cache`.

    :Global setting:

    Auto caching can be globally disabled, see:

        - :func:`~set_auto_cache_state`
        - :func:`~get_auto_cache_state`

    :return:
    """

    original_fun = fun
    while hasattr(original_fun, 'original'):
        original_fun = original_fun.original

    signature = inspect.signature(original_fun)
    __check_valid_signature(signature)

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        data_name = kwargs.pop("attribute_name", fun.__name__)
        force_recompute = kwargs.pop("force_recompute", False)
        data_cache = kwargs.pop("data_cache", hg.__higra_global_cache)
        no_cache = kwargs.pop("no_cache", False)

        if no_cache or not hg.__provider_caching:
            return fun(*args, **kwargs)

        try:
            obj = None
            if len(args) > 0:
                obj = args[0]
            elif len(kwargs) > 0:
                first_param_name = signature.parameters.values().__iter__().__next__().name
                if first_param_name in kwargs:
                    obj = kwargs[first_param_name]

            if obj is None:
                raise TypeError("cannot find first parameter")

            cache = data_cache.get_data(obj)
            cache = cache.setdefault(_auto_cache_keyword, {})
            cache = cache.setdefault(data_name, {})

            args = __transfer_to_kw_arguments(signature, args, kwargs)
            __add_default_parameter(signature, args, kwargs)
            if len(args) > 0:
                import warnings
                warnings.warn('Auto cache: all positional parameters could not be transformed into '
                              'named parameters.')

            h = __make_hash(*args, **kwargs)

            if force_recompute or h not in cache:
                cache[h] = fun(*args, **kwargs)

            return cache[h]
        except TypeError as e:
            # cannot cache obj...
            return fun(*args, **kwargs)

    wrapper.original = fun
    if wrapper.__doc__ is not None:
        wrapper.__doc__ = wrapper.__doc__ + \
                          "\n\n    **Auto-cache**: This function is decorated with the :func:`~higra.auto_cache` decorator."

    return wrapper


###########################################################
#                                                         #
#               DATA PROVIDER DECORATOR                   #
#                                                         #
###########################################################

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
    """
    Function decorator that associates the given function to the global data provider
    registry.

    :param name:
    :param description:
    :return:
    """

    def decorator(fun):
        if name in hg.__data_providers:
            print("Warning, a data provider with the same name was already defined: ", name, file=sys.stderr)

        hg.__data_providers[name] = DataProvider(name, fun, description)

        return fun

    return decorator


###########################################################
#                                                         #
#              ARGUMENT HELPER DECORATOR                  #
#                                                         #
###########################################################


class __CacheLookupException(Exception):
    pass


def __cache_lookup(obj, dep_path, data_cache):
    """
    Tries to find the dependency name in the object data_cache or use a registered data provider

    :param obj: reference object
    :param dep_path: path to the related data
    :param data_cache:
    :return:
    """
    if obj is None:
        raise __CacheLookupException("Cannot lookup into None-type object cache")

    try:
        obj_cache = data_cache.get_data(obj)
    except TypeError:
        obj_cache = None

    if obj_cache is not None and dep_path in obj_cache:
        name_data = obj_cache[dep_path]
    elif dep_path in hg.__data_providers:  # look in providers
        name_data = hg.__data_providers[dep_path](obj)
    else:
        raise __CacheLookupException("Lookup of "
                                     + dep_path
                                     + " in data cache of "
                                     + str(obj)
                                     + " and in data provider list failed.")

    return name_data


def __resolve_dependency(obj, dep_name, dep_path, data_cache, kwargs):
    """
    If :attr:`dep_name` is not already present in the dictionary :attr:`kwargs`, then tries to find it
    in the data_cache or with a data_provider

    :param obj: reference object
    :param dep_name:
    :param dep_path:
    :param data_cache:
    :param kwargs:
    :return:
    """
    # if user has provided an explicit initialization for current dependency
    if dep_name in kwargs and kwargs[dep_name] is not None:
        pass
        # old path rewriting functionality
        # provided_dep = kwargs[dep_name]
        # # if user has provided a path
        # if isinstance(provided_dep, str):
        #     raise Exception("patatra")
        #     # restart dependency resolution with new path
        #     del kwargs[dep_name]
        #     __resolve_dependency(obj, dep_name, provided_dep, data_cache, kwargs)
        # # else use provided_dep as depName argument value
    else:
        try:
            kwargs[dep_name] = __cache_lookup(obj, dep_path, data_cache)
        except __CacheLookupException as e:
            err = "Lookup for the following argument failed: '" \
                  + dep_name \
                  + "' with associated data path '" \
                  + dep_path \
                  + "' as:\n" \
                    "\t-the caller function did not provide an explicit value for this argument, and\n" \
                    "\t-the reference object '" \
                  + str(obj) \
                  + "' does not contain any attribute at this path, and\n" \
                    "\t-there is no automatic data provider for this path.\n\n" \
                    "Did you forget to specify an argument in the function call?"
            raise __CacheLookupException(err) from e


def __resolve_concept(arg_name, concept, all_parameters_name, all_data_found, kwargs, data_cache):
    """
    Tries to expand the elements contained in the concept ``concept`` for the data associated to the name
    ``arg_name``.

    :param arg_name: name or the data element
    :param concept: concept type or concept object
    :param all_parameters_name: name of all known parameters of the function
    :param all_data_found: dictionary of all found data in the concept resolutions so far
    :param kwargs: dictionary of all known name arguments so far
    :param data_cache:
    :return:
    """
    if not type(concept) is type:
        # if concept is a concept object, get the associated type and potential name mapping
        concept_type = type(concept)
        concept_name_to_arg_name_map = concept.name_mapping
    else:
        concept_type = concept
        concept_name_to_arg_name_map = {}

    if not issubclass(concept_type, hg.Concept):
        raise Exception(str(concept_type) + " is not a subclass of the abstract Concept class.")

    # if arg_name is not associated to any found data, then we can not do anything
    if arg_name in all_data_found:
        arg_value = all_data_found[arg_name]
        concept_elements = concept_type.construct(arg_value, strict=False)

        for data_element_name, data_element in concept_elements.items():
            # tries to map the given element name to a new name or itself if no such mapping exists
            argument_name = concept_name_to_arg_name_map.get(data_element_name, data_element_name)
            # if element name is requested by function and not already in kwargs
            if argument_name not in kwargs and argument_name in all_parameters_name:
                kwargs[argument_name] = data_element

        # add all found elements to found data
        all_data_found.update(concept_elements)


def argument_helper(*concepts):
    def decorator(fun):

        original_fun = fun
        while hasattr(original_fun, 'original'):
            original_fun = original_fun.original

        signature = inspect.signature(original_fun)
        __check_valid_signature(signature)

        all_parameters_name = set()
        for p in signature.parameters.values():
            all_parameters_name.add(p.name)

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            args = __transfer_to_kw_arguments(signature, args, kwargs)
            data_debug = kwargs.pop("data_debug", False)
            data_cache = kwargs.pop("data_cache", hg.__higra_global_cache)
            all_data_found = dict(kwargs)
            for concept_elem in concepts:

                try:
                    arg_name, concept = concept_elem
                except:  # failed to unpack, use first parameter name
                    concept = concept_elem
                    arg_name = signature.parameters.values().__iter__().__next__().name

                if type(concept) is type or issubclass(type(concept), hg.Concept):
                    __resolve_concept(arg_name, concept, all_parameters_name, all_data_found, kwargs, data_cache)
                else:
                    try:
                        if arg_name in all_data_found:
                            arg_value = all_data_found[arg_name]
                        else:
                            arg_value = None
                        __resolve_dependency(arg_value, concept, concept, data_cache, kwargs)

                    except __CacheLookupException as e:
                        if signature.parameters[arg_name].default is signature.parameters[arg_name].empty:
                            if data_debug:
                                err = "Error during the resolution of the arguments of the function '" \
                                      + fun.__name__
                                raise Exception(err) from e
                            else:  # swallow exception chain for readability
                                err = "Error during the resolution of the arguments of the function '" + fun.__name__ + "'.\n" \
                                      + str(e) \
                                      + "\nYou can call your function with the extra parameter 'data_debug=True' to " \
                                        "get more information about this error."
                                raise Exception(err) from None

            if len(args) > 0:
                import warnings
                warnings.warn('Argument helper: all positional parameters could not be transformed into '
                              'named parameters.')
            return fun(*args, **kwargs)

        wrapper.original = fun

        return wrapper

    return decorator
