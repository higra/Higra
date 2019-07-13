Changelog
=========

0.3.6
-----

- Add ``plot_graph`` and ``plot_partition_tree``
  `#104 <https://github.com/PerretB/Higra/pull/104>`_
- Add ``make_graph_from_points``
  `#104 <https://github.com/PerretB/Higra/pull/104>`_
- Add ``print_partition_tree``
  `#103 <https://github.com/PerretB/Higra/pull/103>`_
- Add ``tree_2_binary_tree``
  `#101 <https://github.com/PerretB/Higra/pull/101>`_
- Add ``tree.num_children()`` overload that returns the number of children of every non leaf nodes
  `#101 <https://github.com/PerretB/Higra/pull/101>`_


0.3.5
-----

Breaking change
***************

- Rename ``quasi_flat_zones_hierarchy`` to ``quasi_flat_zone_hierarchy``
  `<https://github.com/PerretB/Higra/commit/8aa95694fc7b8b59fd61ffe264943586e935a686>`_

Other changes
*************

- Add ``exponential-linkage`` for agglomerative clustering
  `<https://github.com/PerretB/Higra/commit/a523d8cc484576907e356113dde23adf832eb13b>`_
- Add ``canonize_hierarchy``
  `<https://github.com/PerretB/Higra/commit/9a2c8d9e103fc3444f733e0c5a83b2bd775fdea8>`_

0.3.4
-----

- Add ``filter_non_relevant_node_from_tree``, ``filter_small_nodes_from_tree``, and ``filter_weak_frontier_nodes_from_tree``
  `<https://github.com/PerretB/Higra/commit/521f2416b9b649ace76168728c6d5c06edfde8c6>`_
- Add ``labelisation_horizontal_cut_from_num_regions``
  `<https://github.com/PerretB/Higra/commit/cb9cc0d6ebeaa97f76c60ae1b879f2bfb777c01b>`_
- Add ``at_least`` and ``at_most`` parameters for ``horizontal_cut_from_num_regions``
  `<https://github.com/PerretB/Higra/commit/7b5d00422562840de93df9fcef247b27a2d7365d>`_
- Optimize Horizontal cut explorer construction
  `<https://github.com/PerretB/Higra/commit/68128b9f0201360888d7409dad397ceba23b100d>`_
- Add ``tree.child(i)`` overload that returns the i-th child of every non leaf nodes
  `<https://github.com/PerretB/Higra/commit/6d47a21e942debfdebb633d6e7b7de88238c30ba>`_

0.3.3
-----

- Add ``accumulate_at``
  `<https://github.com/PerretB/Higra/commit/4dadfad522aa6f8d59fa185507a0941c6fc0d0b0>`_
- Add ``altitude_correction`` parameter to Ward linkage
  `<https://github.com/PerretB/Higra/commit/196386fe7e96aa9c8d97dd269b40ca022bb5dfbb>`_
- Make ``edge_weights`` parameter of ``undirected_graph_2_adjacency_matrix`` optional
  `<https://github.com/PerretB/Higra/commit/ca195a9d26ef7eaeb24afc7df5db9b90ba8e5ee7>`_

0.3.2
-----

- Add ``dendrogram_purity``
  `<https://github.com/PerretB/Higra/commit/fb84d6fbc908d2bc1971cf6fc840f3da8c23c5bb>`_
- Add ``random_binary_partition_tree``
  `<https://github.com/PerretB/Higra/commit/46ff1e54d65b658c8d90682761fd77606b764e3c>`_
- Fix altitudes increasingness in Ward linkage
  `<https://github.com/PerretB/Higra/commit/82ba29f940a85c328df76bf9642cfc85f0b94dc7>`_

0.3.1
-----

- Code cleanup
  `#95 <https://github.com/PerretB/Higra/pull/95>`_
- Add Ward linkage
  `#94 <https://github.com/PerretB/Higra/pull/94>`_
- Add ``make_lca_fast`` for fast lca result caching
  `#93 <https://github.com/PerretB/Higra/pull/93>`_

0.3.0
-----

Breaking change
***************

- Refactor Python concepts
  `#88 <https://github.com/PerretB/Higra/pull/88>`_


Other changes
*************

- Fix bug with ``saliency`` working on rags
  `#92 <https://github.com/PerretB/Higra/pull/92>`_
- Fix bug in wheels generation (test result were ignored)
  `#90 <https://github.com/PerretB/Higra/pull/90>`_
- Fix bug in ``linearize_vertex_weights``
  `#89 <https://github.com/PerretB/Higra/pull/89>`_
- Update ``xtensor``
  `#86 <https://github.com/PerretB/Higra/pull/86>`_
- Add ``tree.lowest_common_ancestor``
  `#85 <https://github.com/PerretB/Higra/pull/85>`_
- Add ``attribute_perimeter_component_tree``
  `#84 <https://github.com/PerretB/Higra/pull/84>`_
- Add Tree of shapes
  `#82 <https://github.com/PerretB/Higra/pull/82>`_




