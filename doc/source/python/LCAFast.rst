.. _LCAFast:

LCAFast
=======

``LCAFast`` is a utility class to perform fast computation of lowest common ancestors in a tree.
In exchange of a :math:`n\log(n)` pre-processing it offers a constant time query for any pairs of vertices.

.. currentmodule:: higra

.. autosummary::

    make_lca_fast
    LCAFast

.. autofunction:: higra.make_lca_fast

.. autoclass:: higra.LCAFast
    :special-members:
    :members: