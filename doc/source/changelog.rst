Changelog
=========

0.6.13
------

- Add support for Python 3.14 and remove Python 3.9 `#292 <https://github.com/higra/Higra/pull/292>`_
- Add wheels for Mac ARM  `#292 <https://github.com/higra/Higra/pull/292>`_
- Add option to change the reference node in  :func:`~higra.attribute_height`  `#289 <https://github.com/higra/Higra/pull/289>`_
- Add new example in the doc:  `*Attribute Maps - Component Tree Attribute as Image* <https://github.com/higra/Higra-Notebooks/blob/master/Component%20Tree%20Attribute%20as%20Image.ipynb>`_

0.6.12
------

- Add support for Python 3.13 and remove Python 3.8 `#283 <https://github.com/higra/Higra/pull/283>`_
- Add functions  :func:`~higra.component_tree_tree_of_shapes_image` and :func:`~higra.component_tree_tree_of_shapes_image3d` for 3D tree of shapes `#280 <https://github.com/higra/Higra/pull/280>`_
- Add utility functions  :func:`~higra.get_6_adjacency_graph` and :func:`~higra.get_6_adjacency_implicit_graph` for 3D regular graphs `#280 <https://github.com/higra/Higra/pull/280>`_

0.6.10
------

- Add support for Numpy 2.0 `#278 <https://github.com/higra/Higra/pull/278>`_
- Improve performance of function :func:`~higra.attribute_extrema` `#273 <https://github.com/higra/Higra/pull/273>`_

0.6.8
-----

- Add support for Python 3.12 and remove Python 3.7 `#272 <https://github.com/higra/Higra/pull/272>`_
- Fix bug with empty graphs in function :func:`~higra.bipartite_graph_matching` for minimum cost matching in bipartite graphs `#270 <https://github.com/higra/Higra/pull/270>`_
- Fix bug in function :func:`~higra.ultrametric_open` `#268 <https://github.com/higra/Higra/pull/268>`_

0.6.7
-----

- Add function :func:`~higra.is_bipartite_graph` `#265 <https://github.com/higra/Higra/pull/265>`_
- Add function :func:`~higra.bipartite_graph_matching` for minimum cost matching in bipartite graphs `#265 <https://github.com/higra/Higra/pull/265>`_
- Add function :func:`~higra.match_pixels_image_2d` to compute the optimal matching between the pixels/contours of 2 images `#265 <https://github.com/higra/Higra/pull/265>`_
- Add "similarity" mode in Dasgupta quality measure  :func:`~higra.dasgupta_cost` `#264 <https://github.com/higra/Higra/pull/264>`_
- Add :func:`~higra.Tree.to_undirected_graph` to the tree class `#263 <https://github.com/higra/Higra/pull/263>`_
- Add option to use a preexisting graph in :func:`~higra.graph_4_adjacency_2_khalimsky` `#262 <https://github.com/higra/Higra/pull/262>`_


0.6.6
-----

- Add support for Python 3.11  `#261 <https://github.com/higra/Higra/pull/261>`_
- Improve `notebooks section <https://higra.readthedocs.io/en/stable/notebooks.html>`_ of the doc and add new notebooks on 3D mesh processing:
  `*Watershed on a mesh with IGL* <https://github.com/higra/Higra-Notebooks/blob/master/Watershed_on_a_Mesh_IGL.ipynb>`_
  and
  `*Watershed on a mesh with Trimesh* <https://github.com/higra/Higra-Notebooks/blob/master/Watershed_on_a_Mesh_Trimesh.ipynb>`_
  `#258 <https://github.com/higra/Higra/pull/258>`_
- Fix unwanted behavior in :func:`~higra.reconstruct_leaf_data` `#254 <https://github.com/higra/Higra/pull/254>`_
- Fix crash when creating an empty tree in Python `#252 <https://github.com/higra/Higra/pull/252>`_

0.6.5
-----

- Add support for Python 3.10, remove support of Python 3.6  `#243 <https://github.com/higra/Higra/pull/243>`_
- Add an optional condition to the tree propagate and accumulate function  see documentation
  :ref:`tree conditional propagate and accumulate` :func:`~higra.propagate_sequential_and_accumulate`
  `#241 <https://github.com/higra/Higra/pull/241>`_

0.6.4
-----

- Add function to compute the line graph of a graph :func:`~higra.line_graph`
  `#239 <https://github.com/higra/Higra/pull/239>`_
- Add function to extract a subtree from a tree :func:`~higra.Tree.sub_tree`
  `#238 <https://github.com/higra/Higra/pull/238>`_
- Add new fast lowest common ancestor algorithm  :func:`~higra.Tree.lowest_common_ancestor_preprocess` with :math:`O(n)`
  time preprocessing and average :math:`O(1)` time query (based on sparse table with blocks). The expected speedup
  is comprised between 1.2 and 1.8 compared to the previous method  (parse table without block which is still available).
  `#237 <https://github.com/higra/Higra/pull/237>`_
- Add options to set a global thread number limit :func:`~higra.set_num_threads`
  `#234 <https://github.com/higra/Higra/pull/234>`_

0.6.2
-----

- Add support for Python 3.9, remove support of Python 3.5  `#233 <https://github.com/higra/Higra/pull/233>`_

0.6.1
-----

Breaking changes
****************

- C++ only: the children relation of trees is not computed automatically anymore, a call to the member function
  ``compute_children`` is required before accessing to any children information of a tree,
  see documentation :ref:`tree children`  `#228 <https://github.com/higra/Higra/pull/228>`_


Other changes
*************
- Function :func:`~higra.bpt_canonical` can now process a graph given as a edge list (arrays of source and target vertices)
  `#230 <https://github.com/higra/Higra/pull/230>`_
- Add functions :func:`~higra.UndirectedGraph.sources` and :func:`~higra.UndirectedGraph.targets` to
  the classes :class:`~higra.UndirectedGraph` and :class:`~higra.Tree` which will returns views whenever possible
  `#230 <https://github.com/higra/Higra/pull/230>`_
- Add options for memory pre-allocation in :class:`~higra.UndirectedGraph` constructor
  `#232 <https://github.com/higra/Higra/pull/232>`_
- Improve performance of regular grid graphs `#231 <https://github.com/higra/Higra/pull/231>`_
- Improve memory usage of marginal accumulators `#228 <https://github.com/higra/Higra/pull/228>`_
- Remove the need of the  children relation anymore of trees in several functions
  `#228 <https://github.com/higra/Higra/pull/228>`_
- Bugfix: Regular grid graph will now always fulfils the :class:`~higra.CptGridGraph` concept
  `#229 <https://github.com/higra/Higra/pull/229>`_

0.6.0
-----

Breaking changes
****************

- The functions :func:`~higra.filter_non_relevant_node_from_tree`, :func:`~higra.filter_small_nodes_from_tree`, and
  :func:`~higra.filter_weak_frontier_nodes_from_tree` now return canonized tree by default (old behaviour is obtained with
  the argument ``canonize_tree=False``) `#221 <https://github.com/higra/Higra/pull/221>`_
- C++ only: the function ``bpt_canonical`` does not compute an explicit representation of the minimum spanning tree.
  The mst can still be reconstructed with the field ``mst_edge_map`` in the result using the new function ``subgraph_spanning``
  `f50ebc8 <https://github.com/higra/Higra/commit/f50ebc86b516ef00d23472cafb201f9bba72f58b>`_
- C++ only: fields of the class ``regular_graph`` are now private with public const accessors
  `#211 <https://github.com/higra/Higra/pull/211>`_

Other changes
*************

- Major Python classes (Trees, graphs, ...) are now pickable `#212 <https://github.com/higra/Higra/pull/212>`_ and
  `#214 <https://github.com/higra/Higra/pull/214>`_
- Python classes now support dynamic attributes and higra attributes are now stored directly in the objects dictionaries
  with a direct access as class members. `#209 <https://github.com/higra/Higra/pull/209>`_ and `#210 <https://github.com/higra/Higra/pull/210>`_
- Function :func:`~higra.bpt_canonical` now supports: construction of the tree based on an arbitrary given ordering,
  avoid the explicit computation of the minimum spanning tree, multidimentional edge weights (with lexicographic sorting).
  The documentation has also been improved. `#222 <https://github.com/higra/Higra/pull/222>`_
- Fast lowest ancestor computation is now better integrated to the :class:`~higra.Tree` class.
  Calling :func:`~higra.Tree.lowest_common_ancestor_preprocess()` will now make any call to :func:`~higra.Tree.lowest_common_ancestor`
  to use the pre-processing `#216 <https://github.com/higra/Higra/pull/216>`_
- Add parallel sorting functions :func:`~higra.sort` and :func:`~higra.arg_sort` (also support lexicographic ordering).
  `#219 <https://github.com/higra/Higra/pull/219>`_
- Add function :func:`~higra.subgraph` to extract the subgraph induced by a set of edges from an undirected graph
  `4cfa9ac <https://github.com/higra/Higra/commit/4cfa9ac5f04859f8f0322d881addf07292179720>`_
- Functions for watershed hierarchies in Python can now return the non canonized tree (option ``canonize_tree=False``)
  `#220 <https://github.com/higra/Higra/pull/220>`_
- Function :func:`~higra.canonize_hierarchy` can now return the ``node_map`` which associates any node of the canonized tree to
  a node of the original tree.   `5701d29 <https://github.com/higra/Higra/commit/5701d29e60934aef72a2cf15532b2b6d72c4b52e>`_
- Fix bug in :func:`~higra.filter_small_nodes_from_tree` when the base graph is a region adjacency graph
  `#215 <https://github.com/higra/Higra/pull/215>`_

0.5.3
-----

- Fix bug in :func:`~higra.watershed_hierarchy_by_attribute`: on some conditions a large minima could be split in two
  or more regions.
  `#205 <https://github.com/higra/Higra/pull/205>`_

0.5.2
-----

- Add function :func:`~higra.tree_monotonic_regression`: perform regression on a tree with an increasingness constraint
  `#198 <https://github.com/higra/Higra/pull/198>`_
- Add attribute :func:`~higra.attribute_moment_of_inertia`: first Hu moment on hierarchies constructed on 2d pixel graphs.
  `#197 <https://github.com/higra/Higra/pull/197>`_
- Add attribute :func:`~higra.attribute_topological_height`: number of edges on the longest path from a node to the leaf.
  `#194 <https://github.com/higra/Higra/pull/194>`_
- Improve support for regular graphs: add functions :func:`~higra.RegularGraph2d.as_explicit_graph`
  (convert an implicit graph to an explicit graph), :func:`~higra.mask_2_neighbours` (create an neighbour list from
  an adjacency mask), :func:`~higra.get_nd_regular_graph` and :func:`~higra.get_nd_regular_implicit_graph` (create
  a regular implicit or explicit regular graph)
  `#204 <https://github.com/higra/Higra/pull/204>`_
- Improve conversions functions between adjacency matrices and undirected graphs: improve functions
  :func:`~higra.adjacency_matrix_2_undirected_graph` and :func:`~higra.undirected_graph_2_adjacency_matrix`
  (support for *Scipy* sparse matrix), and :func:`~higra.make_graph_from_points` (add symmetrization strategies).
  `#201 <https://github.com/higra/Higra/pull/201>`_
- Improve documentation of :func:`~higra.binary_partition_tree`, fix typos in :ref:`tree`, add section :ref:`troubleshooting`.
  `#199 <https://github.com/higra/Higra/pull/199>`_ `#196 <https://github.com/higra/Higra/pull/196>`_
- Add altitudes increasingness assertions in several functions
  `#193 <https://github.com/higra/Higra/pull/193>`_
- Fix incorrect overload resolution in :func:`~higra.RegularGraph2d.as_explicit_graph` when seeds are not of
  type ``np.int64``
  `#203 <https://github.com/higra/Higra/pull/203>`_
- Fix incorrect number of regions computation in fragmentation curves when ground-truth labels are not contiguous
  :ref:`fragmentation_curve`
  `#200 <https://github.com/higra/Higra/pull/200>`_
- Fix :func:`~higra.delinearize_vertex_weights` not supporting `Numpy` arrays as shapes.
  `#188 <https://github.com/higra/Higra/pull/188>`_
- Fix :func:`~higra.save_tree` incorrectly failing with no tree attributes.
  `#181 <https://github.com/higra/Higra/pull/181>`_



0.5.1
-----

- Decrease ABI compatibility of linux wheels to 8 (G++ 4.9)
  `#177 <https://github.com/higra/Higra/pull/177>`_

0.5.0
-----

Breaking change
***************

- Removed overload of function :func:`~higra.weight_graph` taking a custom weighting function.
  An equivalent, and much efficient, behavior can be achieved be applying a vectorized
  function on the edge list (see :func:`~higra.UndirectedGraph.edge_list`)
  `5914574 <https://github.com/higra/Higra/commit/5914574e825258a3d0bb7fddd108ec59e6a65919>`_
- Removed support for Python 3.4
  `#174 <https://github.com/higra/Higra/pull/174>`_

Other changes
*************

- Add support for Python 3.8
  `#174 <https://github.com/higra/Higra/pull/174>`_
- Fix and add more efficient implementation of seeded watershed labelisation :func:`~higra.labelisation_seeded_watershed`
  `#173 <https://github.com/higra/Higra/pull/173>`_
- Parallelize several algorithms with Intel TBB (parallel sort, hierarchy construction, fast LCA, graph weighting)
  `#168 <https://github.com/higra/Higra/pull/168>`_ `#169 <https://github.com/higra/Higra/pull/169>`_
- Add support for Intel Threading Building Blocks (TBB), see usage in :ref:`installation_instruction`
  `#168 <https://github.com/higra/Higra/pull/168>`_ `#175 <https://github.com/higra/Higra/pull/175>`_
- Update third party libs
  `#170 <https://github.com/higra/Higra/pull/170>`_
- Fix agglomerative clustering when the input graph has duplicated edges :ref:`binary_partition_tree`
  `#167 <https://github.com/higra/Higra/pull/167>`_
- Fix missing overloads for unsigned types in :func:`~higra.weight_graph`
  `#166 <https://github.com/higra/Higra/pull/166>`_
- Fix a bug in hierarchical watershed when leaves had non zero values :ref:`watershed_hierarchy`
  `#165 <https://github.com/higra/Higra/pull/165>`_

0.4.5
-----

- Add new notebook: `*Visualizing hierarchical image segmentations* <https://github.com/higra/Higra-Notebooks/blob/master/Visualizing%20hierarchical%20image%20segmentations.ipynb>`_
  `#159 <https://github.com/higra/Higra/pull/159>`_
- Add hierarchical cost function :func:`~higra.tree_sampling_divergence`
  `#158 <https://github.com/higra/Higra/pull/158>`_
- Add attribute :func:`~higra.attribute_tree_sampling_probability`
  `9faf740 <https://github.com/higra/Higra/commit/9faf7408b878962c5146df7f19533cd2c843702a>`_
- Add attribute :func:`~higra.attribute_children_pair_sum_product`
  `0c6c958 <https://github.com/higra/Higra/commit/0c6c95860293d65776058a9f449d819e725d0fee>`_
- Improvements in documentation
  `#157 <https://github.com/higra/Higra/pull/157>`_
- Add hierarchy algorithm :func:`~higra.component_tree_multivariate_tree_of_shapes_image2d`
  `#156 <https://github.com/higra/Higra/pull/156>`_
- Fix return policy in :func:`~higra.Tree.parents()`, now returns a non writable reference
  `e3eb5aa <https://github.com/higra/Higra/commit/e3eb5aa902e81e2d6ce38b54d2e41171256035d6>`_
- Add option to deactivate immersion in tree of shapes
  `9efb6b6 <https://github.com/higra/Higra/commit/9efb6b670beb7f42a28f05bdd3c9ead1062180b9>`_
- Add algorithm :func:`~higra.tree_fusion_depth_map`
  `11e4f53 <https://github.com/higra/Higra/commit/11e4f530f07778247f04833b0e90d607aef228ac>`_

0.4.4
-----

- Fix *codecov* incorrectly including third party libs
  `#152 <https://github.com/higra/Higra/pull/152>`_
- Add hierarchical cost :func:`~higra.dasgupta_cost`
  `#151 <https://github.com/higra/Higra/pull/151>`_
- Add new attribute :func:`~higra.attribute_child_number`
  `#149 <https://github.com/higra/Higra/pull/149>`_
- Fix bug in :func:`~higra.simplify_tree`
  `#148 <https://github.com/higra/Higra/pull/148>`_ and `#150 <https://github.com/higra/Higra/pull/150>`_
- Add *argmin* and *argmax* accumulators
  `#146 <https://github.com/higra/Higra/pull/146>`_
- Add new notebooks: *PRL article illustrations* and *Astromical object detection with the Max-Tree*
  `#145 <https://github.com/higra/Higra/pull/145>`_ and `#155 <https://github.com/higra/Higra/pull/155>`_
- Documentation improvements
  `#143 <https://github.com/higra/Higra/pull/143>`_, `#153 <https://github.com/higra/Higra/pull/153>`_,
  `#154 <https://github.com/higra/Higra/pull/154>`_
- Update third party libs
  `#141 <https://github.com/higra/Higra/pull/141>`_


0.4.2
-----

Breaking change
***************

- Rename function `attribute_mean_weights` into :func:`~higra.attribute_mean_vertex_weights`
  `#136 <https://github.com/higra/Higra/pull/136>`_


Other changes
*************

- Add SoftwareX illustrations notebook
  `#140 <https://github.com/higra/Higra/pull/140>`_
- Replace specialized C++ bindings for hierarchical watershed by a generic calls to :func:`~higra.watershed_hierarchy_by_attribute`
  `#139 <https://github.com/higra/Higra/pull/139>`_
- Fix inconsistency between Python and C++ definitions of :func:`~higra.attribute_volume`
  `#138 <https://github.com/higra/Higra/pull/138>`_
- Separate code and documentation on graph and tree attributes
  `#137 <https://github.com/higra/Higra/pull/137>`_
- Fix bug in  :func:`~higra.attribute_mean_vertex_weights`
  `#136 <https://github.com/higra/Higra/pull/136>`_

0.4.1
-----

- Add function :func:`~higra.accumulate_on_contours`.
  `#134 <https://github.com/higra/Higra/pull/134>`_
- Better handling of null perimeter in :func:`~higra.attribute_contour_strength`.
  `#133 <https://github.com/higra/Higra/pull/133>`_
- Add links to :ref:`notebooks` in the documentation.
  `#132 <https://github.com/higra/Higra/pull/132>`_
- Fix bug in :func:`~higra.common_type` support for `bool` type was missing.
  `#131 <https://github.com/higra/Higra/pull/131>`_
- Fix bug in :func:`~higra.attribute_contour_length` with tree of shapes when interpolated are removed.
  `#129 <https://github.com/higra/Higra/pull/129>`_


0.4.0
-----

Breaking change
***************

- Refactor attributes related to perimeter: there is now a single homogeneous function
  :func:`~higra.attribute_contour_length` that replaces `attribute_perimeter_length`,
  `attribute_perimeter_length_component_tree`, and `attribute_perimeter_length_partition_tree`
  `#121 <https://github.com/higra/Higra/pull/121>`_ and `#124 <https://github.com/higra/Higra/pull/124>`_
- Add decorator :func:`~higra.auto_cache` for auto-caching of function results which replaces the
  decorator `data_provider`.
  `#122 <https://github.com/higra/Higra/pull/122>`_ and `#127 <https://github.com/higra/Higra/pull/127>`_

Other changes
*************

- Add a Cookiecutter project for c++ higra extension development `Higra-cppextension-cookiecutter <https://github.com/higra/Higra-cppextension-cookiecutter>`_
- Add more documentation for installation and compiling
  `#123 <https://github.com/higra/Higra/pull/123>`_
- Fix bug with integer data in  :func:`~higra.attribute_gaussian_region_weights_model`
  `#126 <https://github.com/higra/Higra/pull/126>`_
- Fix bug in graph associated to the :func:`~higra.component_tree_tree_of_shapes_image2d`
  `#120 <https://github.com/higra/Higra/pull/120>`_
- Improve algorithm for :func:`~higra.attribute_extrema`
  `#119 <https://github.com/higra/Higra/pull/119>`_
- Moved repository to `higra` Github organization
  `#118 <https://github.com/higra/Higra/pull/118>`_



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




