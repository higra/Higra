.. _hg_utils:

Utility functions
=================

.. currentmodule:: higra

.. autosummary::

    arg_sort
    sort
    set_num_threads
    is_iterable
    extend_class
    normalize_shape
    linearize_vertex_weights
    delinearize_vertex_weights
    has_method
    is_in_bijection
    mean_angle_mod_pi
    dtype_info
    common_type
    cast_to_common_type
    cast_to_dtype
    get_include
    get_lib_include
    get_lib_cmake

.. autofunction:: higra.arg_sort

.. autofunction:: higra.sort

.. autofunction:: set_num_threads

.. autofunction:: higra.is_iterable

.. autofunction:: higra.extend_class

.. autofunction:: higra.normalize_shape

Vertex weights and image shapes
-------------------------------

Use :func:`linearize_vertex_weights` to collapse an image's spatial axes into
one vertex axis, and :func:`delinearize_vertex_weights` to restore them.
You can supply ``shape`` directly or let a grid graph provide it. If you supply
both a graph and a shape, their vertex counts must agree. Trailing component
axes and the array dtype are preserved::

    import higra as hg
    import numpy as np

    image = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    linear = hg.linearize_vertex_weights(image, shape=(2, 3))
    assert linear.shape == (6, 4)
    restored = hg.delinearize_vertex_weights(linear, shape=(2, 3))
    np.testing.assert_array_equal(restored, image)

Both helpers check for a matching spatial prefix before checking the first
axis against the number of vertices. For example, shape ``(4, 1)`` and weights
of shape ``(4, 1, 3)`` linearize to ``(4, 3)``; delinearizing the original
weights returns them unchanged. Without an explicit or graph-derived shape,
the input is returned unchanged.

.. autofunction:: higra.linearize_vertex_weights

.. autofunction:: higra.delinearize_vertex_weights

.. autofunction:: higra.has_method

.. autofunction:: higra.is_in_bijection

.. autofunction:: higra.mean_angle_mod_pi

.. autofunction:: higra.dtype_info

.. autofunction:: higra.common_type

.. autofunction:: higra.cast_to_common_type

.. autofunction:: higra.cast_to_dtype

.. autofunction:: higra.get_include

.. autofunction:: higra.get_lib_include

.. autofunction:: higra.get_lib_cmake