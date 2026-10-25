.. _hierarchy_core:

Core hierarchy algorithm
========================

.. currentmodule:: higra

.. autosummary::

    bpt_canonical
    bpt_canonical_boruvka
    saliency
    quasi_flat_zone_hierarchy
    simplify_tree
    canonize_hierarchy
    tree_2_binary_tree

.. autofunction:: higra.bpt_canonical

.. autofunction:: higra.canonize_hierarchy

.. autofunction:: higra.quasi_flat_zone_hierarchy

.. autofunction:: higra.simplify_tree

.. autofunction:: higra.tree_2_binary_tree

.. autofunction:: higra.saliency

.. autofunction:: higra.bpt_canonical_boruvka

For example, construct a binary partition tree of a weighted image grid:

.. code-block:: python

    import higra as hg
    import numpy as np

    graph = hg.get_4_adjacency_graph((2, 3))
    weights = np.array([1., 0., 2., 1., 1., 1., 2.])
    tree, altitudes = hg.bpt_canonical_boruvka(graph, weights)

The default reproduces :func:`bpt_canonical` for the same edge order.
The experimental ``stable=False`` option preserves the merges and selected
edges while changing internal node numbers. Use the default when consumers
require node indices to follow global merge order. Parallel execution does
not guarantee a speedup.

For input MSTs linked to a base graph, keep ``compute_mst=True``; deferred
MST retrieval for these inputs may return incorrect edges.

For these inputs, the returned tree's ``mst_edge_map`` indexes the supplied
MST. To obtain base-graph edge IDs, use
``graph.mst_edge_map[tree.mst_edge_map]``, where ``graph`` is the input MST.
The edge map attached to the eager output MST also indexes the supplied MST;
use the composed map when base-graph edge IDs are needed.
