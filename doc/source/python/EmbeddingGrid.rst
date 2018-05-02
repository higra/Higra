.. _EmbeddingGrid:

Embedding Grid
==============

Grid embeddings are utility classes to ease the manipulation of point coordinates in the d-dimensional integer grid.
An embedding has a shape (height and width in 2 dimensions).


For a given dimension :math:`N\in\{1,2,3,4,5\}`, there is a specific grid embedding class called ``EmbeddingGridNd``.
All the specific grid embedding classes implement the same interface.

.. autoclass:: higra.EmbeddingGrid2d
    :members: