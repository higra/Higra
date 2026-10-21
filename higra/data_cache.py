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
import warnings as _warnings
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


def __support_dynamic_attributes(obj):
    return hasattr(obj, '__dict__') and isinstance(obj.__dict__, dict)


def __get_attributes(key):
    """
    Get a dictionary associated to the object :attr:`key` to store Higra related attributes.

    If the object support dynamic attributes, this dictionary is stored as a (new) object attribute called
    "__higra_attributes__". Otherwise, the dictionary is stored in a global data store with weak references.
    This last case means that such attributes are not part of the given object and won't be pickled by Python if the
    object is stored.

    :param key: an object
    :return: a dictionary of attributes associated to the given key
    """
    if __support_dynamic_attributes(key):
        data = getattr(key, "__higra_attributes__", None)
        if data is None:
            data = {}
            key.__higra_attributes__ = data
        return data
    else:
        return hg.__higra_global_cache.get_data(key)


def list_attributes(key):
    """
    List all Higra attributes of the object :attr:`key`.

    See functions :func:`~higra.get_attribute` and :func:`~higra.set_attribute`.

    :param key: an object
    :return: a list of strings
    """
    try:
        data = __get_attributes(key)
        return list(data.keys())
    except TypeError:
        return ()


def get_attribute(key, attribute_name):
    """
    Get the Higra attributes named :attr:`attribute_name` of the object :attr:`key`.

    See functions :func:`~higra.list_attributes` and :func:`~higra.set_attribute`.

    :param key: an object
    :param attribute_name: a string
    :return: the attribute value associated to the given name (``None`` if no such attribute exists)
    """
    try:
        return __get_attributes(key).get(attribute_name, None)
    except TypeError:
        return None


def set_attribute(key, attribute_name, attribute, insert_dynamic=True):
    """
    Set the Higra attributes named :attr:`attribute_name` of the object :attr:`key` to the value :attr:`attribute`.

    See functions :func:`~higra.list_attributes` and :func:`~higra.get_attribute`.

    :param key: an object
    :param attribute_name: a string
    :param attribute: a value
    :param insert_dynamic: if ``True`` and if :attr:`key` supports dynamic attributes, a new object attribute with the
           given name and values will be added to the given object
    :return: None
    """
    __get_attributes(key)[attribute_name] = attribute
    if insert_dynamic and __support_dynamic_attributes(key):
        setattr(key, attribute_name, attribute)


def get_tags(key):
    if __support_dynamic_attributes(key):
        data = getattr(key, "__higra_tags__", None)
        if data is None:
            data = set()
            key.__higra_tags__ = data
        return data
    else:
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
    """Deprecated no-op; automatic result caching is disabled.

    Emits ``FutureWarning`` and will be removed in the next release.
    :param boolean: ignored bool; other types still raise ``TypeError``
    """
    _warnings.warn("Automatic caching is deprecated and disabled. set_auto_cache_state is now a no-op and will be removed in the next release",
                  FutureWarning, stacklevel=2)
    if not isinstance(boolean, bool):
        raise TypeError("Parameter must be a bool.")


def get_auto_cache_state():
    """Emit ``FutureWarning`` and return ``False``; removed in the next release."""
    _warnings.warn("Automatic caching is deprecated and disabled. get_auto_cache_state is now a no-op and will be removed in the next release",
                  FutureWarning, stacklevel=2)
    return False


def clear_auto_cache(*, function=None, reference_object=None, data_cache=None):
    """Deprecated no-op; emit ``FutureWarning`` without modifying any data.

    All parameters are ignored. Will be removed in the next release.
    Attributes, tags, concept links and explicit preprocessing are preserved.
    """
    _warnings.warn("Automatic caching is deprecated and disabled.clear_auto_cache is now a no-op and will be removed in the next release",
                  FutureWarning, stacklevel=2)


def _legacy_cache_keywords(fun):
    """Accept obsolete controls without caching; keep this outside argument_helper."""
    parameters = inspect.signature(fun).parameters
    controls = {"no_cache", "force_recompute", "attribute_name", "data_cache"} - parameters.keys()

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        supplied = controls.intersection(kwargs)
        if supplied:
            _warnings.warn("Automatic caching is deprecated and disabled. The automatic caching controls " + ", ".join(sorted(supplied)) +
                          " are ignored and will be removed in the next release",
                          FutureWarning, stacklevel=2)
            for name in supplied:
                del kwargs[name]
        return fun(*args, **kwargs)

    wrapper.original = fun
    wrapper._legacy_cache_keywords = True
    return wrapper


def auto_cache(fun):
    """Automatic caching is deprecated and disabled. Deprecated no-op decorator.

    Emits ``FutureWarning`` at decoration, with no warning on ordinary calls.
    The decorator will be removed in the next release. Implicit ``no_cache``,
    ``force_recompute``, ``attribute_name`` and ``data_cache`` keywords are
    accepted but ignored with ``FutureWarning`` when supplied.
    """
    _warnings.warn("Automatic caching is deprecated and disabled. auto_cache is now a no-op and will be removed in the next release",
                  FutureWarning, stacklevel=2)
    return _legacy_cache_keywords(fun)

###########################################################
#                                                         #
#              ARGUMENT HELPER DECORATOR                  #
#                                                         #
###########################################################


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
            if argument_name in all_parameters_name and (argument_name not in kwargs or kwargs[argument_name] is None):
                kwargs[argument_name] = data_element

        # add all found elements to found data
        all_data_found.update(concept_elements)


def argument_helper(*concepts):
    """
    Argument helper decorator helps to unpack :class:`~higra.Concept` data elements.

    The decorator expresses that a function parameter :attr:`p` is designed as *possibly* satisfying the concept `c`.
    Then, whenever the function is called, the decorator will check if :attr:`p` satisfies `c`, it will extract all the
    data elements associated to :attr:`p` for `c`. Let `e` be one of these data element associated to the name `n`, if
    the function has a parameter called *n* that is either undefined or ``None`` in the current call, then the decorator
    will inject ``n=e`` as a new keyword parameter in the function call.

    :param concepts:
    :return:
    """
    for concept_elem in concepts:
        try:
            _, concept = concept_elem
        except (ValueError, TypeError):  # failed to unpack, use first parameter name
            concept = concept_elem

        if not (type(concept) is type or issubclass(type(concept), hg.Concept)):
            raise TypeError("Argument helper can only proceed Concept types or "
                            "object instances deriving of Concept. Invalid element is: " + concept)

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
            data_cache = (kwargs.get("data_cache") if "data_cache" in all_parameters_name
                          else kwargs.pop("data_cache", hg.__higra_global_cache))
            all_data_found = dict(kwargs)
            for concept_elem in concepts:

                try:
                    arg_name, concept = concept_elem
                except (ValueError, TypeError):  # failed to unpack, use first parameter name
                    concept = concept_elem
                    arg_name = signature.parameters.values().__iter__().__next__().name

                __resolve_concept(arg_name, concept, all_parameters_name, all_data_found, kwargs, data_cache)

            if len(args) > 0:
                _warnings.warn('Argument helper: all positional parameters could not be transformed into '
                              'named parameters.')
            return fun(*args, **kwargs)

        wrapper.original = fun

        if getattr(fun, "_legacy_cache_keywords", False):
            return _legacy_cache_keywords(wrapper)
        return wrapper

    return decorator
