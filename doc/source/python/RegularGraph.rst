.. _RegularGraph:

Regular graph
=============

Regular graphs are translation invariant graphs in the d-dimensional integer grid.
Regular graphs do not explicitly store edges: they are thus not suited to represent edge weighted graphs.

For a given dimension :math:`N\in\{1,2,3,4,5\}`, there is a specific regular graph class called ``RegularGraphNd``.
All the specific regular graph classes implement the same interface.

.. autoclass:: higra.RegularGraph2d
    :members: