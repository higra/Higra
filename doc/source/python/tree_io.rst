.. _tree_io:

Tree IO
=======

Tree IO saves trees and scalar node attributes. 
Reading returns ``(tree, attribute_map)`` and registers each attribute on
the returned tree. Attributes are stored and read as one-dimensional arrays of
double precision values. 

For Python pickle serialization, see :ref:`TreeGraph`.

.. currentmodule:: higra

.. autosummary::

    print_partition_tree
    read_tree
    save_tree

.. autofunction:: higra.print_partition_tree

.. autofunction:: higra.read_tree

.. autofunction:: higra.save_tree
