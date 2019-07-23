Changelog
=========

0.3.8
-----

- Add attributes: :func:`~higra.attribute_height`, :func:`~higra.attribute_extrema`,
  :func:`~higra.attribute_extinction_value`, and :func:`~higra.attribute_dynamics`
  `#110 <https://github.com/PerretB/Higra/pull/110>`_
- Fix tree category propagation
  `#109 <https://github.com/PerretB/Higra/pull/109>`_

0.3.7
-----

- Hardening: add range checks in various Python bindings
  `#107 <https://github.com/PerretB/Higra/pull/107>`_
- Bundle ``Higra`` and third party libraries into pip wheel for easy C++ extension development:
  :func:`~higra.get_include`, :func:`~higra.get_lib_include`, :func:`~higra.get_lib_cmake`
  `#106 <https://github.com/PerretB/Higra/pull/106>`_
- Make ``deleted_nodes`` parameter of :func:`~higra.reconstruct_leaf_data` optional
  `#105 <https://github.com/PerretB/Higra/pull/105>`_


0.3.6
-----

- Add ``plot_graph`` and :func:`~higra.plot_partition_tree`
  `#104 <https://github.com/PerretB/Higra/pull/104>`_
- Add :func:`~higra.make_graph_from_points`
  `#104 <https://github.com/PerretB/Higra/pull/104>`_
- Add :func:`~higra.print_partition_tree`
  `#103 <https://github.com/PerretB/Higra/pull/103>`_
- Add :func:`~higra.tree_2_binary_tree`
  `#101 <https://github.com/PerretB/Higra/pull/101>`_
- Add :func:`~higra.Tree.num_children` overload that returns the number of children of every non leaf nodes
  `#101 <https://github.com/PerretB/Higra/pull/101>`_


0.3.5
-----

Breaking change
***************

- Rename ``quasi_flat_zones_hierarchy`` to :func:`~higra.quasi_flat_zone_hierarchy`
  `<https://github.com/PerretB/Higra/commit/8aa95694fc7b8b59fd61ffe264943586e935a686>`_

Other changes
*************

- Add ``exponential-linkage`` for agglomerative clustering :func:`~higra.binary_partition_tree_exponential_linkage`
  `<https://github.com/PerretB/Higra/commit/a523d8cc484576907e356113dde23adf832eb13b>`_
- Add :func:`~higra.canonize_hierarchy`
  `<https://github.com/PerretB/Higra/commit/9a2c8d9e103fc3444f733e0c5a83b2bd775fdea8>`_

0.3.4
-----

- Add :func:`~higra.filter_non_relevant_node_from_tree`, :func:`~higra.filter_small_nodes_from_tree`,
  and :func:`~higra.filter_weak_frontier_nodes_from_tree`
  `<https://github.com/PerretB/Higra/commit/521f2416b9b649ace76168728c6d5c06edfde8c6>`_
- Add :func:`~higra.labelisation_horizontal_cut_from_num_regions`
  `<https://github.com/PerretB/Higra/commit/cb9cc0d6ebeaa97f76c60ae1b879f2bfb777c01b>`_
- Add ``at_least`` and ``at_most`` parameters for :func:`~higra.HorizontalCutExplorer.horizontal_cut_from_num_regions`
  `<https://github.com/PerretB/Higra/commit/7b5d00422562840de93df9fcef247b27a2d7365d>`_
- Optimize Horizontal cut explorer construction
  `<https://github.com/PerretB/Higra/commit/68128b9f0201360888d7409dad397ceba23b100d>`_
- Add :func:`~higra.Tree.child` overload that returns the i-th child of every non leaf nodes
  `<https://github.com/PerretB/Higra/commit/6d47a21e942debfdebb633d6e7b7de88238c30ba>`_

0.3.3
-----

- Add :func:`~higra.accumulate_at`
  `<https://github.com/PerretB/Higra/commit/4dadfad522aa6f8d59fa185507a0941c6fc0d0b0>`_
- Add ``altitude_correction`` parameter to Ward linkage :func:`~higra.binary_partition_tree_ward_linkage`
  `<https://github.com/PerretB/Higra/commit/196386fe7e96aa9c8d97dd269b40ca022bb5dfbb>`_
- Make ``edge_weights`` parameter of :func:`~higra.undirected_graph_2_adjacency_matrix` optional
  `<https://github.com/PerretB/Higra/commit/ca195a9d26ef7eaeb24afc7df5db9b90ba8e5ee7>`_

0.3.2
-----

- Add :func:`~higra.dendrogram_purity`
  `<https://github.com/PerretB/Higra/commit/fb84d6fbc908d2bc1971cf6fc840f3da8c23c5bb>`_
- Add :func:`~higra.random_binary_partition_tree`
  `<https://github.com/PerretB/Higra/commit/46ff1e54d65b658c8d90682761fd77606b764e3c>`_
- Fix altitudes increasingness in Ward linkage :func:`~higra.binary_partition_tree_ward_linkage`
  `<https://github.com/PerretB/Higra/commit/82ba29f940a85c328df76bf9642cfc85f0b94dc7>`_

0.3.1
-----

- Code cleanup
  `#95 <https://github.com/PerretB/Higra/pull/95>`_
- Add Ward linkage :func:`~higra.binary_partition_tree_ward_linkage`
  `#94 <https://github.com/PerretB/Higra/pull/94>`_
- Add :func:`~higra.make_lca_fast` for fast lca result caching
  `#93 <https://github.com/PerretB/Higra/pull/93>`_

0.3.0
-----

Breaking change
***************

- Refactor Python concepts
  `#88 <https://github.com/PerretB/Higra/pull/88>`_


Other changes
*************

- Fix bug with :func:`~higra.saliency` working on rags
  `#92 <https://github.com/PerretB/Higra/pull/92>`_
- Fix bug in wheels generation (test result were ignored)
  `#90 <https://github.com/PerretB/Higra/pull/90>`_
- Fix bug in :func:`~higra.linearize_vertex_weights`
  `#89 <https://github.com/PerretB/Higra/pull/89>`_
- Update ``xtensor``
  `#86 <https://github.com/PerretB/Higra/pull/86>`_
- Add :func:`~higra.Tree.lowest_common_ancestor`
  `#85 <https://github.com/PerretB/Higra/pull/85>`_
- Add :func:`~higra.attribute_perimeter_length_component_tree`
  `#84 <https://github.com/PerretB/Higra/pull/84>`_
- Add Tree of shapes :func:`~higra.component_tree_tree_of_shapes_image2d`
  `#82 <https://github.com/PerretB/Higra/pull/82>`_




