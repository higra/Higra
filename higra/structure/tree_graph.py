############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg
import numpy as np


@hg.extend_class(hg.Tree, method_name="find_region")
def __find_region(self, vertex, level, altitudes):
    """
    Searches for the largest node of altitude lower than the given level and containing the given vertex.
    If no such node exists the given vertex is returned.

    :param vertex: a vertex or a 1d array of vertices
    :param level: a level or a 1d array of levels (should have the same dtype as altitudes)
    :param altitudes: altitudes of the nodes of the tree
    :return: a vertex or a 1d array of vertices
    """

    if isinstance(vertex, np.ndarray):
        if not isinstance(level, np.ndarray):
            level = np.full_like(vertex, level, dtype=altitudes.dtype)
        else:
            level = hg.cast_to_dtype(level, altitudes.dtype)
    else:
        if np.issubdtype(altitudes.dtype, np.integer):
            level = int(level)
        else:
            level = float(level)

    result = self._find_region(vertex, level, altitudes)

    return result


@hg.extend_class(hg.Tree, method_name="child")
def __child(self, index, vertex=None):
    """
    Get the :attr:`index`-th (starting at 0) child of the given vertex/array of vertices.

    If :attr:`vertex` is ``None``, the function will return the :attr:`index`-th child of every non leaf
    node of the tree.

    :param index: positive integer
    :param vertex: a vertex index or a 1d array of vertex indices
        (default to ``np.arange(self.num_leaves(), self.num_vertices()``)
    :return: a vertex index or a 1d array of vertex indices
    """

    index = int(index)

    if vertex is None:
        vertex = np.arange(self.num_leaves(), self.num_vertices())

    result = self._child(index, vertex)

    return result


@hg.extend_class(hg.Tree, method_name="num_children")
def __num_children(self, vertex=None):
    """
    Get the the number of children of the given vertices.

    If :attr:`vertex` is ``None``, the function will return the number of children of every non leaf
    node of the tree.

    :param vertex: a vertex index or a 1d array of vertex indices
        (default to ``np.arange(self.num_leaves(), self.num_vertices()``)
    :return: an integer or a 1d array of integers
    """

    if vertex is None:
        vertex = np.arange(self.num_leaves(), self.num_vertices())

    return self._num_children(vertex)


@hg.extend_class(hg.Tree, method_name="__reduce__")
def ____reduce__(self):
    return self.__class__, (self.parents(),), self.__dict__


@hg.extend_class(hg.Tree, method_name="lowest_common_ancestor_preprocess")
def __lowest_common_ancestor_preprocess(self, algorithm="sparse_table_block", block_size=1024, force_recompute=False):
    """
    Preprocess the tree to obtain a fast constant time :math:`\\mathcal{O}(1)` lowest common ancestor query.
    Once this function has been called on a given tree instance, every following calls to the function
    :func:`~higra.Tree.lowest_common_ancestor` will use this preprocessing. Calling twice this function does nothing
    except if :attr:`force_recompute` is ``True``.

    Two algorithms are available:

    - ``sparse_table`` has a preprocessing time and space complexity in :math:`\\mathcal{O}(n\log(n))` with :math:`n`
      the number of vertices in the tree and performs every query in constant time :math:`\\mathcal{O}(1)`.
    - ``sparse_table_block`` (default) has a linear preprocessing time and space complexity in :math:`\\mathcal{O}(n)`
      and performs queries in average-case constant time :math:`\\mathcal{O}(1)`. With this algorithm the user can specify
      the block size to be used, the general rule of thumb being that larger block size will decrease the pre-processing
      time but increase the query time.

    :param algorithm: specify the algorithm to be used, can be either ``sparse_table`` or ``sparse_table_block``.
    :param block_size: if :attr:`algorithm` is ``sparse_table_block``, specify the block size to be used (default 1024)
    :param force_recompute: if ``False`` (default) calling this function twice won't re-preprocess the tree, even if the
           specified algorithm or algorithm parameter have changed.
    :return: An object of type :class:`~higra.hg.LCA_rmq_sparse_table_block` or :class:`~higra.hg.LCA_rmq_sparse_table`
    """
    lca_fast = hg.get_attribute(self, "lca_fast")
    if lca_fast is None or force_recompute:
        if algorithm == "sparse_table":
            lca_fast = hg.LCA_rmq_sparse_table(self)
        elif algorithm == "sparse_table_block":
            block_size = int(block_size)
            if block_size <= 0:
                raise ValueError("Invalid block size: " + str(block_size))

            lca_fast = hg.LCA_rmq_sparse_table_block(self, block_size)
        else:
            raise ValueError("Unknown LCA algorithm: " + str(algorithm))
        hg.set_attribute(self, "lca_fast", lca_fast)
    return lca_fast


@hg.extend_class(hg.Tree, method_name="lowest_common_ancestor")
def __lowest_common_ancestor(self, vertices1, vertices2):
    """
    Compute the lowest common ancestor between pairs of vertices defined by :attr:`vertices1` and :attr:`vertices2`.

    :attr:`vertices1` and :attr:`vertices2` must be either:

    - two positive integers strictly smaller than the number of vertices in the tree;
    - two 1d arrays of positive integers strictly smaller than the number of vertices in the tree and of the same size.

    :Complexity:

    The worst case time complexity is :math:`\mathcal{O}(qn)` with :math:`q` the number of lowest ancestors to compute and :math:`n`
    the number of vertices in the tree.

    If many lowest ancestors are needed, this time complexity can be reduced to :math:`\mathcal{O}(q)` at the cost of a linearithmic
    time :math:`\\mathcal{O}(n\log(n))` preprocessing by calling the function :func:`~higra.Tree.lowest_common_ancestor_preprocess`.

    :param vertices1: a vertex index or an array of vertex indices
    :param vertices2: a vertex index or an array of vertex indices
    :return: the lowest common ancestor(s) of every pair of input vertices (a single index or an array of indices)
    """

    lca_fast = hg.get_attribute(self, "lca_fast")
    if lca_fast is None:
        return self._lowest_common_ancestor(vertices1, vertices2)
    else:
        return lca_fast.lca(vertices1, vertices2)


@hg.extend_class(hg.Tree, method_name="sources")
def __sources(self):
    """
    Source vertex of every edge of the graph.

    :Example:

    >>> t = Tree((5, 5, 6, 6, 6, 7, 7, 7))
    >>> t.sources()
    array([0, 1, 2, 3, 4, 5, 6])

    :return: a 1d array of size ``self.num_edges()``
    """
    return np.arange(self.num_vertices() - 1)


@hg.extend_class(hg.Tree, method_name="targets")
def __targets(self):
    """
    Target vertex of every edge of the graph.

    :Example:

    >>> t = Tree((5, 5, 6, 6, 6, 7, 7, 7))
    >>> t.targets()
    array([5, 5, 6, 6, 6, 7, 7])

    :return: a 1d array of size ``self.num_edges()``
    """
    return self.parents()[:-1]


@hg.extend_class(hg.Tree, method_name="edge_list")
def __edge_list(self):
    """
    Returns a tuple of two arrays (sources, targets) defining all the edges of the graph.

    :Example:

    >>> t = Tree((5, 5, 6, 6, 6, 7, 7, 7))
    >>> t.edge_list()
    (array([0, 1, 2, 3, 4, 5, 6]), array([5, 5, 6, 6, 6, 7, 7]))


    :return: pair of two 1d arrays
    """
    return self.sources(), self.targets()


@hg.extend_class(hg.Tree, method_name="sub_tree")
def __sub_tree(self, root_node):
    """
    Extract the sub tree rooted in the given node of the current tree.

    The result is a new tree :math:`st` and a node map :math:`nm` such that:

    - the node map associates each node of the sub tree :math:`st` to its corresponding node in the original tree
    - the order of the nodes of the original tree is preserved in the sub tree:
      for any vertices :math:`x` and :math:`y` of :math:`st` such that  :math:`x < y` then :math:`nm[x] < nm[y]`

    :Complexity:

    The tree is constructed in linearithmic time :math:`\mathcal{O}(n\log(n))` with :math:`n` the number of vertices
    in the sub tree.

    :Example:

    >>> t = Tree((5, 5, 6, 6, 6, 7, 7, 7))
    >>> sub_tree, node_map = t.sub_tree(6)
    >>> sub_tree.parents()
    array([3, 3, 3, 3])
    >>> node_map
    array([2, 3, 4, 6])

    :param root_node: a vertex of the current tree
    :return: the sub tree rooted in :attr:`root` and the node map
    """
    return hg.cpp._sub_tree(self, root_node)