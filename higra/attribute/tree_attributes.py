############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################


import numpy as np
import higra as hg


@hg.argument_helper(hg.CptHierarchy)
@hg.auto_cache
def attribute_area(tree, vertex_area=None, leaf_graph=None):
    """
    Area of each node the given tree.
    The area of a node is equal to the sum of the area of the leaves of the subtree rooted in the node.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param vertex_area: area of the vertices of the leaf graph of the tree (provided by :func:`~higra.attribute_vertex_area` on `leaf_graph` )
    :param leaf_graph: (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array
    """
    if vertex_area is None:
        if leaf_graph is not None:
            vertex_area = hg.attribute_vertex_area(leaf_graph)
        else:
            vertex_area = np.ones((tree.num_leaves(),), dtype=np.float64)

    if leaf_graph is not None:
        vertex_area = hg.linearize_vertex_weights(vertex_area, leaf_graph)
    return hg.accumulate_sequential(tree, vertex_area, hg.Accumulators.sum)


@hg.auto_cache
def attribute_volume(tree, altitudes, area=None):
    """
    Volume of each node the given tree.
    The volume :math:`V(n)` of a node :math:`n` is defined recursively as:

    .. math::

        V(n) = area(n) * | altitude(n) - altitude(parent(n)) | +  \sum_{c \in children(n)} V(c)

    :param tree: input tree
    :param altitudes: node altitudes of the input tree
    :param area: area of the nodes of the input hierarchy (provided by :func:`~higra.attribute_area` on `tree`)
    :return: a 1d array
    """
    if area is None:
        area = hg.attribute_area(tree)

    height = np.abs(altitudes[tree.parents()] - altitudes)
    height = height * area
    volume_leaves = np.zeros(tree.num_leaves(), dtype=np.float64)
    return hg.accumulate_and_add_sequential(tree, height, volume_leaves, hg.Accumulators.sum)


@hg.argument_helper(hg.CptHierarchy)
@hg.auto_cache
def attribute_lca_map(tree, leaf_graph):
    """
    Lowest common ancestor of `i` and `j` for each edge :math:`(i, j)` of the leaf graph of the given tree.

    Complexity: :math:`\mathcal{O}(n\log(n)) + \mathcal{O}(m)` where :math:`n` is the number of nodes in `tree` and
    :math:`m` is the number of edges in :attr:`leaf_graph`.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param leaf_graph: graph on the leaves of the input tree (deduced from :class:`~higra.CptHierarchy` on `tree`)
    :return: a 1d array
    """
    lca = tree.lowest_common_ancestor_preprocess()
    res = lca.lca(leaf_graph)
    return res


@hg.argument_helper(hg.CptHierarchy)
@hg.auto_cache
def attribute_frontier_length(tree, edge_length=None, leaf_graph=None):
    """
    Length of the frontier represented by each node the given partition tree.

    In a partition tree, each node represent the merging of 2 or more regions.
    The frontier of a node is then defined as the common contour between the merged regions.
    This function compute the length of these common contours as the sum of the length of edges going from one of the
    merged region to the other one.

    The result has the same dtype as the edge_length array.

    :param tree: input tree
    :param edge_length: length of the edges of the leaf graph (provided by :func:`~higra.attribute_edge_length` on `leaf_graph`)
    :param leaf_graph: graph on the leaves of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array
    """

    if edge_length is None:
        edge_length = hg.attribute_edge_length(leaf_graph)

    lca_map = attribute_lca_map(tree, leaf_graph)

    frontier_length = np.zeros((tree.num_vertices(),), dtype=edge_length.dtype)
    np.add.at(frontier_length, lca_map, edge_length)
    return frontier_length


@hg.argument_helper(hg.CptHierarchy)
@hg.auto_cache
def attribute_frontier_strength(tree, edge_weights, leaf_graph):
    """
    Mean edge weight along the frontier represented by each node the given partition tree.

    In a partition tree, each node represent the merging of 2 or more regions.
    The frontier of a node is then defined as the common contour between the merged regions.
    This function compute the strength of a common contour as the sum of the weights of edges going from one of the
    merged region to the other one divided by the length of the contour.

    The result has the same dtype as the edge_weights array.

    :param tree: input tree
    :param edge_weights: weight of the edges of the leaf graph (if leaf_graph is a region adjacency graph, edge_weights might be weights on the edges of the pre-graph of the rag).
    :param leaf_graph: graph on the leaves of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array
    """
    # this is a rag like graph
    if hg.CptRegionAdjacencyGraph.validate(leaf_graph) and edge_weights.shape[0] != leaf_graph.num_edges():
        edge_weights = hg.rag_accumulate_on_edges(leaf_graph, hg.Accumulators.sum, edge_weights=edge_weights)

    frontier_length = hg.attribute_frontier_length(tree, leaf_graph=leaf_graph)
    frontier_strength = hg.attribute_frontier_length(tree, edge_weights, leaf_graph)
    frontier_strength[tree.num_leaves():] = frontier_strength[tree.num_leaves():] / frontier_length[tree.num_leaves():]
    return frontier_strength


@hg.argument_helper(hg.CptHierarchy)
@hg.auto_cache
def attribute_contour_length(tree, vertex_perimeter=None, edge_length=None, leaf_graph=None):
    """
    Length of the contour (perimeter) of each node of the given tree.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param vertex_perimeter: perimeter of each vertex of the leaf graph (provided by :func:`~higra.attribute_vertex_perimeter` on `leaf_graph`)
    :param edge_length: length of each edge of the leaf graph (provided by :func:`~higra.attribute_edge_length` on `leaf_graph`)
    :param leaf_graph: (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array
    """

    if vertex_perimeter is None:
        vertex_perimeter = hg.attribute_vertex_perimeter(leaf_graph)

    if edge_length is None:
        edge_length = hg.attribute_edge_length(leaf_graph)

    if leaf_graph is not None:
        vertex_perimeter = hg.linearize_vertex_weights(vertex_perimeter, leaf_graph)

    frontier_length = hg.attribute_frontier_length(tree, edge_length, leaf_graph)
    perimeter = hg.accumulate_and_add_sequential(tree, -2 * frontier_length, vertex_perimeter,
                                                 hg.Accumulators.sum)

    # hg.cpp._attribute_contour_length_component_tree is more efficient than the partition tree
    # algorithm but it does not work for tree of shapes left in original space (the problem is that
    # two children of a node may become adjacent when the interpolated pixels are removed).

    # if tree.category() == hg.TreeCategory.PartitionTree:
    #     frontier_length = hg.attribute_frontier_length(tree, edge_length, leaf_graph)
    #     perimeter = hg.accumulate_and_add_sequential(tree, -2 * frontier_length, vertex_perimeter,
    #                                                  hg.Accumulators.sum)
    # elif tree.category() == hg.TreeCategory.ComponentTree:
    #     perimeter = hg.cpp._attribute_contour_length_component_tree(tree, leaf_graph, vertex_perimeter,
    #                                                                 edge_length)

    return perimeter


@hg.argument_helper(hg.CptHierarchy)
@hg.auto_cache
def attribute_contour_strength(tree, edge_weights, vertex_perimeter=None, edge_length=None, leaf_graph=None):
    """
    Strength of the contour of each node of the given tree. The strength of the contour of a node is defined as the
    mean edge weights on the contour.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param edge_weights: edge_weights of the leaf graph
    :param vertex_perimeter: perimeter of each vertex of the leaf graph (provided by :func:`~higra.attribute_vertex_perimeter` on `leaf_graph`)
    :param edge_length: length of each edge of the leaf graph (provided by :func:`~higra.attribute_edge_length` on `leaf_graph`)
    :param leaf_graph: (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array
    """

    if vertex_perimeter is None:
        vertex_perimeter = hg.attribute_vertex_perimeter(leaf_graph)

    if edge_length is None:
        edge_length = hg.attribute_edge_length(leaf_graph)

    perimeter = attribute_contour_length(tree, vertex_perimeter, edge_length, leaf_graph)

    # perimeter of the root may be null
    if np.isclose(perimeter[-1], 0):
        perimeter[-1] = 1

    if hg.CptRegionAdjacencyGraph.validate(leaf_graph):
        edge_weights = hg.rag_accumulate_on_edges(leaf_graph, hg.Accumulators.sum, edge_weights)

    vertex_weights_sum = hg.accumulate_graph_edges(leaf_graph, edge_weights, hg.Accumulators.sum)
    edge_weights_sum = attribute_contour_length(tree, vertex_weights_sum, edge_weights, leaf_graph)

    return edge_weights_sum / perimeter


@hg.argument_helper(hg.CptHierarchy)
@hg.auto_cache
def attribute_compactness(tree, area=None, contour_length=None, normalize=True, leaf_graph=None):
    """
    The compactness of a node is defined as its area divided by the square of its perimeter length.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param area: node area of the input tree (provided by :func:`~higra.attribute_area` on `tree`)
    :param contour_length: node contour length of the input tree (provided by :func:`~higra.attribute_perimeter_length` on `tree`)
    :param normalize: if True the result is divided by the maximal compactness value in the tree
    :param leaf_graph: (deduced from :class:`~higra.CptHierarchy`)
    :return: a 1d array
    """
    if area is None:
        area = hg.attribute_area(tree)

    if contour_length is None:
        contour_length = hg.attribute_contour_length(tree, leaf_graph=leaf_graph)

    compactness = area / (contour_length * contour_length)
    if normalize:
        max_compactness = np.nanmax(compactness)
        compactness = compactness / max_compactness

    return compactness


@hg.argument_helper(hg.CptHierarchy)
@hg.auto_cache
def attribute_mean_vertex_weights(tree, vertex_weights, area=None, leaf_graph=None):
    """
    Mean vertex weights of the leaf graph vertices inside each node of the given tree.

    For any node :math:`n`, the mean vertex weights :math:`a(n)` of :math:`n` is

    .. math::

        a(n) = \\frac{\sum_{x\in n} vertex\_weights(x)}{area(n)}

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param vertex_weights: vertex weights of the leaf graph of the input tree
    :param area: area of the tree nodes  (provided by :func:`~higra.attribute_area`)
    :param leaf_graph: leaf graph of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: a nd array
    """
    if area is None:
        area = hg.attribute_area(tree)

    if leaf_graph is not None:
        vertex_weights = hg.linearize_vertex_weights(vertex_weights, leaf_graph)

    attribute = hg.accumulate_sequential(
        tree,
        vertex_weights.astype(np.float64),
        hg.Accumulators.sum) / area.reshape([-1] + [1] * (vertex_weights.ndim - 1))
    return attribute


@hg.auto_cache
def attribute_sibling(tree, skip=1):
    """
    Sibling index of each node of the given tree.

    For each node :math:`n` which is the :math:`k`-th child of its parent node :math:`p` among :math:`N` children,
    the attribute sibling of :math:`n` is the index of the :math:`(k + skip) % N`-th child of :math:`p`.

    The sibling of the root node is itself.

    The sibling attribute enables to easily emulates a (doubly) linked list among brothers.

    In a binary tree, the sibling attribute of a node is effectively its only brother (with `skip` equals to 1).

    :param tree: Input tree
    :param skip: Number of skipped element in the children list (including yourself)
    :return: a nd array
    """
    attribute = hg.cpp._attribute_sibling(tree, skip)
    return attribute


@hg.auto_cache
def attribute_depth(tree):
    """
    The depth of a node :math:`n` of the tree :math:`T` is equal to the number of ancestors of :math:`n` in :math:`T`.

    The depth of the root node is equal to 0.

    :param tree: Input tree
    :return: a nd array
    """
    attribute = hg.cpp._attribute_depth(tree)
    return attribute


@hg.auto_cache
def attribute_regular_altitudes(tree, depth=None):
    """
    Regular altitudes is comprised between 0 and 1 and is inversely proportional to the depth of a node

    :param tree: input tree
    :param depth: depth of the tree node (provided by :func:`~higra.attribute_depth`)
    :return: a nd array
    """

    if depth is None:
        depth = hg.attribute_depth(tree)

    altitudes = 1 - depth / np.max(depth)
    altitudes[:tree.num_leaves()] = 0
    return altitudes


@hg.auto_cache
def attribute_vertex_list(tree):
    """
    List of leaf nodes inside the sub-tree rooted in a node.

    **WARNING**: This function is slow and will use O(n²) space, with n the number of leaf nodes !

    **SHOULD ONLY BE USED FOR DEBUGGING AND TESTING**

    :param tree: input tree
    :return: a list of lists
    """
    result = [[i] for i in tree.leaves()]

    for i in tree.leaves_to_root_iterator(include_leaves=False):
        tmp = []
        for c in tree.children(i):
            tmp.extend(result[c])
        result.append(tmp)

    return result


@hg.argument_helper(hg.CptHierarchy)
@hg.auto_cache
def attribute_gaussian_region_weights_model(tree, vertex_weights, leaf_graph=None):
    """
    Estimates a gaussian model (mean, (co-)variance) for leaf weights inside a node.

    The result is composed of two arrays:

        - the first one contains the mean value inside each node, scalar if vertex weights are scalar and vectorial otherwise,
        - the second one contains the variance of the values inside each node, scalar if vertex weights are scalar and a (biased) covariance matrix otherwise.

    Vertex weights must be scalar or 1 dimensional.

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param vertex_weights: vertex weights of the leaf graph of the input tree
    :param leaf_graph: leaf graph of the input tree (deduced from :class:`~higra.CptHierarchy`)
    :return: two arrays mean and variance
    """
    if leaf_graph is not None:
        vertex_weights = hg.linearize_vertex_weights(vertex_weights, leaf_graph)

    if vertex_weights.ndim > 2:
        raise ValueError("Vertex weight can either be scalar or 1 dimensional.")

    if vertex_weights.dtype not in (np.float32, np.float64):
        vertex_weights = vertex_weights.astype(np.float64)

    area = hg.attribute_area(tree, leaf_graph=leaf_graph)
    mean = hg.accumulate_sequential(tree, vertex_weights, hg.Accumulators.sum, leaf_graph)

    if vertex_weights.ndim == 1:
        # general case below would work but this is simpler
        mean /= area
        mean2 = hg.accumulate_sequential(tree, vertex_weights * vertex_weights, hg.Accumulators.sum, leaf_graph)
        mean2 /= area
        variance = mean2 - mean * mean
    else:
        mean /= area[:, None]
        tmp = vertex_weights[:, :, None] * vertex_weights[:, None, :]
        mean2 = hg.accumulate_sequential(tree, tmp, hg.Accumulators.sum, leaf_graph)
        mean2 /= area[:, None, None]

        variance = mean2 - mean[:, :, None] * mean[:, None, :]

    return mean, variance


@hg.auto_cache
def attribute_extrema(tree, altitudes):
    """
    Identify nodes in a hierarchy that represent extrema (minima or maxima).

    An extremum (minimum or maximum) of the hierarchy :math:`T` with altitudes :math:`alt` is a node :math:`n` of :math:`T` such that the
    altitude of any non leaf node included in :math:`n` is equal to the altitude of :math:`n` and the altitude of
    the parent of :math:`n` is different from the altitude of :math:`n`.

    The result is a boolean array such that :math:`result(n)` is ``True`` if the node :math:`n` is an extremum and ``False``
    otherwise.

    :param tree: Input tree
    :param altitudes: Tree node altitudes
    :return: a 1d boolean array
    """

    res = hg.cpp._attribute_extrema(tree, altitudes)

    return res


def __process_param_increasing_altitudes(tree, altitudes, increasing_altitudes):
    """
    Assuming that altitudes are monotone for the input tree, test if they are increasing or decreasing.

    :param tree:
    :param altitudes:
    :return:
    """
    if isinstance(increasing_altitudes, bool):
        return increasing_altitudes

    if increasing_altitudes == "auto":
        alt_root = altitudes[tree.root()]
        alt_min = np.min(altitudes[tree.num_leaves():])
        return bool(alt_root > alt_min)
    elif increasing_altitudes == "increasing":
        return True
    elif increasing_altitudes == "decreasing":
        return False
    else:
        raise ValueError("Unknown mode '" + str(increasing_altitudes) + "' valid values are 'auto', True, False, "
                                                                        "'increasing', and 'decreasing'.")


def attribute_extinction_value(tree, altitudes, attribute, increasing_altitudes="auto"):
    """
    The extinction value of a node :math:`n` of the input tree :math:`T` with increasing altitudes :math:`alt`
    for the increasing attribute :math:`att` is the equal to the threshold :math:`k` such that the node :math:`n`
    is still in an minima of :math:`t` when all nodes having an attribute value smaller than :math:`k` are removed.

    Formally, let :math:`\{M_i\}` be the set of minima of the hierarchy :math:`T` with altitudes :math:`alt`.
    Let :math:`prec` be a total ordering of :math:`\{M_i\}` such that :math:`M_i \prec M_j \Rightarrow alt(M_i) \leq alt(M_j)`.
    Let :math:`r(M_i)` be the smallest node of :math:`t` containing :math:`M_i` and another minima :math:`M_j` such
    that :math:`M_j \prec M_i`. The extinction value of :math:`M_i` is then defined as :math:`alt(r(M_i)) - alt(M_i)`.

    Extinction values of minima are then extended to other nodes in the tree with the following rules:

        - the extinction value of a non-leaf node :math:`n` which is not a minimum is defined as the largest
          extinction values among all the minima contained in :math:`n`
          (and 0 if :math:`n` does not contain any minima); and
        - the extinction value of a leaf node :math:`n` belonging to a minima :math:`M_i` is equal to the extinction
          value of :math:`M_i`. I :math:`n` does not belong to any minima its extinction value is 0.

    The function can also handle decreasing altitudes, in which case *minima* should be replaced by *maxima*
    in the description above. Possible values of :attr:`increasing_altitude` are:

        - ``'auto'``: the function will automatically determine if :attr:`altitudes` are increasing or decreasing (this has
          small computational cost but does not impact the runtime complexity).
        - ``True`` or ``'increasing'``: this means that altitudes are increasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitudes(n) \leq altitudes(parent(n))`.
        - ``False`` or ``'decreasing'``: this means that altitudes are decreasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitude(n) \geq altitude(parent(n))`.


    :param tree: Input tree
    :param altitudes: Tree node altitudes
    :param attribute: Tree node attribute
    :param increasing_altitudes: possible values 'auto', True, False, 'increasing', and 'decreasing'
    :return: a 1d array like :attr:`attribute`
    """
    inc = __process_param_increasing_altitudes(tree, altitudes, increasing_altitudes)

    altitudes, attribute = hg.cast_to_common_type(altitudes, attribute)

    res = hg.cpp._attribute_extinction_value(tree, altitudes, attribute, inc)

    return res


@hg.auto_cache
def attribute_height(tree, altitudes, increasing_altitudes="auto"):
    """
    In a tree :math:`T`, given that the altitudes of the nodes vary monotically from the leaves to the root,
    the height of a node :math:`n` of :math:`T` is equal to the difference between the altitude of the parent
    of :math:`n` and the altitude of the deepest non-leaf node in the subtree of :math:`T` rooted in :math:`n`.

    Possible values of :attr:`increasing_altitude` are:

        - ``'auto'``: the function will automatically determine if :attr:`altitudes` are increasing or decreasing (this has
          small computational cost but does not impact the runtime complexity).
        - ``True`` or ``'increasing'``: this means that altitudes are increasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitudes(n) \leq altitudes(parent(n))`.
        - ``False`` or ``'decreasing'``: this means that altitudes are decreasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitude(n) \geq altitude(parent(n))`.

    :param tree: Input tree
    :param altitudes: Tree node altitudes
    :param increasing_altitudes: possible values 'auto', True, False, 'increasing', and 'decreasing'
    :return: a 1d array like :attr:`altitudes`
    """
    inc = __process_param_increasing_altitudes(tree, altitudes, increasing_altitudes)

    res = hg.cpp._attribute_height(tree, altitudes, inc)

    return res


@hg.auto_cache
def attribute_dynamics(tree, altitudes, increasing_altitudes="auto"):
    """
    Given a node :math:`n` of the tree :math:`T`, the dynamics of :math:`n` is the difference between
    the altitude of the deepest minima of the subtree rooted in :math:`n` and the altitude
    of the closest ancestor of :math:`n` that has a deeper minima in its subtree. If no such
    ancestor exists then, the dynamics of :math:`n` is equal to the difference between the
    altitude of the highest node of the tree (the root) and the depth of the deepest minima.

    The dynamics is the *extinction values* (:func:`~higra.attribute_extinction_value`) for the attribute *height*
    (:func:`~higra.attribute_height`).

    Possible values of :attr:`increasing_altitude` are:

        - ``'auto'``: the function will automatically determine if :attr:`altitudes` are increasing or decreasing (this has
          small computational cost but does not impact the runtime complexity).
        - ``True`` or ``'increasing'``: this means that altitudes are increasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitudes(n) \leq altitudes(parent(n))`.
        - ``False`` or ``'decreasing'``: this means that altitudes are decreasing from the leaves to the root
          (ie. for any node :math:`n`, :math:`altitude(n) \geq altitude(parent(n))`.


    :param tree: Input tree
    :param altitudes: Tree node altitudes
    :param increasing_altitudes: possible values 'auto', True, False, 'increasing', and 'decreasing'
    :return: a 1d array like :attr:`altitudes`
    """

    inc = __process_param_increasing_altitudes(tree, altitudes, increasing_altitudes)

    height = hg.attribute_height(tree, altitudes, inc)

    return hg.attribute_extinction_value(tree, altitudes, height, inc)


@hg.auto_cache
def attribute_child_number(tree):
    """
    Given a node :math:`n` whose parent is :math:`p`, the attribute value of :math:`n` is the rank of :math:`n`
    in the list of children of :math:`p`. In other :math:`attribute(n)=i` means that :math:`n` is the :math:`i`-th
    child of :math:`p`.

    The root of the tree, who has no parent, take the value -1.

    :param tree: Input tree
    :return: a 1d array
    """

    res = hg.cpp._attribute_child_number(tree)

    return res


def attribute_children_pair_sum_product(tree, node_weights):
    """
    Given a tree :math:`T` with node weights :math:`w`: the children pair sum product for a node :math:`n` sums for
    every pairs :math:`(c_i, c_j)` of children of :math:`n`, the product of the node weights of :math:`c_i` and
    :math:`c_j`. Formally:

    .. math::

        res(n) = \sum_{i=0}^{i<numc(n)} \sum_{j=0}^{j<i} w(child(i, n)) * w(child(j, n))

    where :math:`numc(n)` is the number of children of :math:`n` and :math:`child(i, n)` is the :math:`i`-th child
    of the node :math:`n`.

    The result is thus an array with the same shape as :attr:`node_weights`

    :param tree: Input tree
    :param node_weights: node weights of the input tree
    :return: an array with the same shape as :attr:`node_weights`
    """

    res = hg.cpp._attribute_children_pair_sum_product(tree, node_weights)

    return res


def attribute_tree_sampling_probability(tree, leaf_graph, leaf_graph_edge_weights, model='edge'):
    """
    Given a tree :math:`T`, estimate the probability that a node :math:`n` of the tree represents the smallest cluster
    containing a pair of vertices :math:`\{a, b\}` of the graph :math:`G=(V, E)`
    with edge weights :math:`w`.

    This method is defined in [1]_.

    We define the probability :math:`P(\{a,b\})` of a pair of vertices :math:`\{a,b\}` as :math:`w(\{a,b\}) / Z`
    with :math:`Z=\sum_{e\in E}w(E)` if :math:`\{a,b\}` is an edge of :math:`G` and 0 otherwise.
    Then the probability :math:`P(a)` of a vertex :math:`b` is defined as :math:`\sum_{b\in V}P(\{a, b\})`

    Two sampling strategies are proposed for sampling pairs of vertices to compute the probability of a node of the tree:

    - *edge*: the probability of sampling the pair :math:`\{a, b\}` is given by :math:`P(\{a, b\})`; and
    - *null*: the probability of sampling the pair :math:`\{a, b\}` is given by the product of the probabilities
      of :math:`a` and :math:`b`: :math:`P(a)*P(b)`.

    Assuming that the edge weights on the leaf graph of a hierarchy represents similarities:

    .. epigraph::

        *We expect these distributions to differ significantly if the tree indeed represents the hierarchical structure of the graph.
        Specifically, we expect [the edge distribution] to be mostly concentrated on deep nodes of the tree
        (far from the root), as two nodes* :math:`u`, :math:`v` *connected with high weight* :math:`w(\{u, v\})` *in the graph
        typically  belong to a small cluster, representative of the clustering structure of the graph; on the contrary,
        we expect [the null distribution] to be concentrated over shallow nodes (close to the root) as two nodes*
        :math:`w(\{u, v\})` *sampled independently at random typically belong to large clusters, less representative of the
        clustering structure of the graph*. [1]_


    .. [1] Charpentier, B. & Bonald, T. (2019).  `"Tree Sampling Divergence: An Information-Theoretic Metric for \
           Hierarchical Graph Clustering." <https://hal.telecom-paristech.fr/hal-02144394/document>`_ Proceedings of IJCAI.

    :Complexity:

    The tree sampling divergence runtime complexity depends of the sampling model:

     - *edge*: :math:`\mathcal{O}(N\log(N) + M)` with :math:`N` the number of  nodes in the tree and :math:`M` the number of edges in the leaf graph.
     - *null*: :math:`\mathcal{O}(N\\times C^2)` with :math:`N` the number of nodes in the tree  and :math:`C` the maximal number of children of a node in the tree.

    :see:

    The :func:`~higra.tree_sampling_divergence` is a non supervised hierarchical cost function defined as the
    Kullback-Leibler divergence between the edge sampling model and the independent (null) sampling model.

    :param tree: Input tree
    :param leaf_graph: Graph defined on the leaves of the input tree
    :param leaf_graph_edge_weights: Edge weights of the leaf graphs (similarities)
    :param model: defines the edge sampling strategy, either "edge" or "null"
    :return: a 1d array
    """
    if model not in ("edge", "null"):
        raise ValueError("Parameter 'model' must be either 'edge' or 'null'.")

    if model == 'edge':
        lca_map = hg.attribute_lca_map(tree, leaf_graph=leaf_graph)
        leaf_graph_edge_weights = leaf_graph_edge_weights / np.sum(leaf_graph_edge_weights)
        return hg.accumulate_at(lca_map, leaf_graph_edge_weights, hg.Accumulators.sum)
    else:  # model = 'null'
        leaf_graph_vertex_weights = hg.accumulate_graph_edges(leaf_graph, leaf_graph_edge_weights, hg.Accumulators.sum)
        leaf_graph_vertex_weights = leaf_graph_vertex_weights / np.sum(leaf_graph_edge_weights)
        tree_node_weights = hg.accumulate_sequential(tree, leaf_graph_vertex_weights, hg.Accumulators.sum)
        return hg.attribute_children_pair_sum_product(tree, tree_node_weights)


@hg.auto_cache
def attribute_topological_height(tree):
    """
    Given a node :math:`n` of :attr:`tree`, the topological height of :math:`n` is the number of edges on the longest
    path from the node :math:`n` to a leaf of :attr:`tree`.

    The topological height of the leaves is equal to 0.

    :param tree: Input tree
    :return: a 1d array
    """

    res = hg.accumulate_and_add_sequential(tree,
                                           np.ones(tree.num_vertices(), dtype=np.int64),
                                           np.zeros(tree.num_leaves(), dtype=np.int64),
                                           hg.Accumulators.max)

    return res


@hg.argument_helper(hg.CptHierarchy)
@hg.auto_cache
def attribute_moment_of_inertia(tree, leaf_graph):
    """
    Moment of inertia (first Hu moment) of each node of the given tree.
    This function works only if :attr:`leaf_graph` is a 2D grid graph.
    The moment of inertia is a translation, scale and rotation invariant characterization of the shape of the nodes.

    Given a node :math:`X` of :attr:`tree`, the raw moments :math:`M_{ij}` are defined as:

    .. math::

        M_{ij} = \sum_{x}\sum_{y} x^i y^j

    where :math:`(x,y)` are the coordinates of every vertex in :math:`X`.
    Then, the centroid :math:`\{\overline{x},\overline{y}\}` of :math:`X` is given by

    .. math::

         \overline{x} = \\frac{M_{10}}{M_{00}} \\textrm{ and  } \overline{y} = \\frac{M_{01}}{M_{00}}

    Some central moments of :math:`X` are then:

    - :math:`\mu_{00} = M_{00}`
    - :math:`\mu_{20} = M_{20} - \overline{x} \\times M_{10}`
    - :math:`\mu_{02} = M_{02} - \overline{y} \\times M_{01}`

    The moment of inertia :math:`I_1` of :math:`X` if finally defined as

    .. math::

        I_1 = \eta_{20} + \eta_{02}
        
    where :math:`\eta_{ij}` are given by:
    
        :math:`\eta_{ij} = \\frac{\mu_{ij}}{\mu_{00}^{1+\\frac{i+j}{2}}}`

    :param tree: input tree (Concept :class:`~higra.CptHierarchy`)
    :param leaf_graph: graph on the leaves of the input tree (deduced from :class:`~higra.CptHierarchy` on `tree`)
    :return: a 1d array
    """

    if (not hg.CptGridGraph.validate(leaf_graph)) or (len(hg.CptGridGraph.get_shape(leaf_graph)) != 2):
        raise ValueError("Parameter 'leaf_graph' must be a 2D grid graph.")

    coordinates = hg.attribute_vertex_coordinates(leaf_graph)
    coordinates = np.reshape(coordinates, (coordinates.shape[0] * coordinates.shape[1], coordinates.shape[2]))

    M_00_leaves = np.ones((tree.num_leaves())).astype(dtype=np.float64)
    x_leaves = coordinates[:, 0].astype(dtype=np.float64)
    y_leaves = coordinates[:, 1].astype(dtype=np.float64)

    M_10_leaves = x_leaves * M_00_leaves
    M_01_leaves = y_leaves * M_00_leaves
    M_20_leaves = (x_leaves ** 2) * M_00_leaves
    M_02_leaves = (y_leaves ** 2) * M_00_leaves

    M_00 = hg.accumulate_sequential(tree, M_00_leaves, hg.Accumulators.sum)
    M_01 = hg.accumulate_sequential(tree, M_01_leaves, hg.Accumulators.sum)
    M_10 = hg.accumulate_sequential(tree, M_10_leaves, hg.Accumulators.sum)
    M_02 = hg.accumulate_sequential(tree, M_02_leaves, hg.Accumulators.sum)
    M_20 = hg.accumulate_sequential(tree, M_20_leaves, hg.Accumulators.sum)

    _x = M_10 / M_00
    _y = M_01 / M_00
    miu_20 = M_20 - _x * M_10
    miu_02 = M_02 - _y * M_01
    I_1 = (miu_20 + miu_02) / (M_00 ** 2)

    return I_1
