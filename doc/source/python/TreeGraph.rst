.. _TreeGraph:

Tree Graph
==========

Tree graphs represent trees as undirected rooted graphs.

Python pickles preserve the tree's parents, category, and Python instance state,
including user attributes, tags, and concept links whose associated objects are
picklable. Linked leaf graphs, grid shapes, and stored maps are restored by value.
Cached objects must themselves support pickling; pickling a tree does not add
serialization support to its caches.

Older pickles whose constructor arguments contain only the parent array remain
readable and use the historical ``TreeCategory.PartitionTree`` default. Their
original category was not stored and cannot be recovered, even if the original
was a component tree.
The constructor's default category remains ``TreeCategory.PartitionTree``.

.. currentmodule:: higra

.. autosummary::

    TreeCategory
    Tree

.. autoclass:: higra.TreeCategory
    :members:
    :undoc-members:

.. autoclass:: higra.Tree
    :special-members:
    :members:
