Data cache
==========

Higra attributes and concepts store metadata associated with graphs, trees and
arrays.

Automatic caching compatibility
-------------------------------

``auto_cache``, ``set_auto_cache_state``, ``get_auto_cache_state`` and
``clear_auto_cache`` are deprecated no-ops and will be removed in the next
release. They emit ``FutureWarning``; the decorator warns at decoration, while
the state and clear functions warn when called. Automatic caching stays disabled
and ``get_auto_cache_state()`` returns ``False``. The setter still requires a
bool. Clearing the retired result cache leaves metadata and explicit LCA
preprocessing intact.
Related ``no_cache``, ``force_recompute``, ``attribute_name`` and ``data_cache``
keywords are ignored and also emit ``FutureWarning`` whenever
supplied, even with false or default values.

.. currentmodule:: higra

.. autosummary::

    list_attributes
    get_attribute
    set_attribute

    auto_cache
    set_auto_cache_state
    get_auto_cache_state
    clear_auto_cache

.. autofunction:: list_attributes

.. autofunction:: get_attribute

.. autofunction:: set_attribute

.. autodecorator:: higra.auto_cache

.. autofunction:: higra.set_auto_cache_state

.. autofunction:: higra.get_auto_cache_state

.. autofunction:: higra.clear_auto_cache
