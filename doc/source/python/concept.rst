.. _concept:

Concepts
========

Concepts associate a canonical object with related data. Built-in ``link``
methods attach the association to a dynamic object. The fields are ordinary
instance fields: assigning a replacement changes subsequent getters and argument
deduction. Deleting a required field or assigning ``None`` makes that data
absent. ``construct(strict=True)`` raises for missing data;
``construct(strict=False)`` omits it. Explicit non-``None`` arguments take
precedence over deduced values.

For example::

    import higra as hg
    import numpy as np

    image = np.arange(6).reshape(2, 3)
    graph = hg.get_4_adjacency_graph((2, 3))
    tree, altitudes = hg.component_tree_max_tree(graph, image)
    assert hg.CptHierarchy.get_leaf_graph(tree) is graph
    assert graph.shape == (2, 3)

Custom Concepts can declare fields and use ``Concept.tag`` in their link method::

    class Labelled(hg.Concept):
        _canonical_data_element = "owner"
        _data_elements = {"label": ("display label", "label")}

        @classmethod
        def link(cls, owner, label):
            cls.tag(owner)
            owner.label = label

    class Item:
        pass

    item = Item()
    Labelled.link(item, "sample")
    assert Labelled.validate(item)
    assert Labelled.construct(item)["label"] == "sample"


.. currentmodule:: higra

.. autosummary::

    Concept
    CptGridGraph
    CptRegionAdjacencyGraph
    CptHierarchy
    CptBinaryHierarchy
    CptMinimumSpanningTree

.. autoclass:: higra.Concept
    :members:

.. autoclass:: higra.CptGridGraph
    :members:

.. autoclass:: higra.CptRegionAdjacencyGraph
    :members:

.. autoclass:: higra.CptHierarchy
    :members:

.. autoclass:: higra.CptBinaryHierarchy
    :members:

.. autoclass:: higra.CptMinimumSpanningTree
    :members:
