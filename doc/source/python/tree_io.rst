.. _tree_io:

Tree IO
=======

Tree IO allows de/serialization of a tree and associated attributes in a custom simple format.

For Python pickle serialization, including category preservation and the
compatibility rules for older category-less pickles, see :ref:`TreeGraph`.

.. currentmodule:: higra

.. autosummary::

    print_partition_tree
    read_tree
    save_tree

.. autofunction:: higra.print_partition_tree

.. autofunction:: higra.read_tree

.. autofunction:: higra.save_tree
