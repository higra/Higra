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
from ..data_cache import _get_attribute, _legacy_cache_keywords


@_legacy_cache_keywords
def attribute_vertex_area(graph):
    """
    Compute the area of each vertex of the given graph.

    The area :math:`a(v)` of a vertex :math:`v` is determined as follows:

    - if :attr:`graph` is a region adjacency graph, each vertex represents a region of the underlying graph of the
      region adjacency graph: its area is the sum of the areas of the vertices of this region, these areas being
      computed recursively with :func:`~higra.attribute_vertex_area` on the underlying graph (see
      :func:`~higra.rag_accumulate_on_vertices`). For a region adjacency graph built on a pixel graph, the area of a
      region is thus its number of pixels;
    - otherwise, the area of each vertex is 1.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(n)` time, where :math:`n` is the number of vertices of the graph. For a
    region adjacency graph, add the cost of the computation on its underlying graph.

    :Example:
    =======

    On a grid graph, the result follows the shape of the grid:

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((2, 3))
        hg.attribute_vertex_area(graph)
        # array([[1., 1., 1.],
        #        [1., 1., 1.]])

    Consider now the region adjacency graph defined by the following labelisation of a 3 x 3 image:

    .. code-block:: text

        0  0  1
        0  0  1
        2  2  2

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        labels = np.asarray(((0, 0, 1), (0, 0, 1), (2, 2, 2)))
        rag = hg.make_region_adjacency_graph_from_labelisation(graph, labels)
        hg.attribute_vertex_area(rag)
        # array([4., 2., 3.])

    The region 0 contains 4 pixels, the region 1 contains 2 pixels and the region 2 contains 3 pixels.

    :param graph: input graph
    :return: an array of floating point values representing the area of each vertex of the graph, with one element per vertex or, if the graph is a grid graph, of the shape of the grid
    """
    if hg.CptRegionAdjacencyGraph.validate(graph):  # this is a rag like graph
        pre_graph = hg.CptRegionAdjacencyGraph.get_pre_graph(graph)
        pre_graph_vertex_area = attribute_vertex_area(pre_graph)
        return hg.rag_accumulate_on_vertices(graph, hg.Accumulators.sum, vertex_weights=pre_graph_vertex_area)
    res = np.ones((graph.num_vertices(),), dtype=np.float64)
    res = hg.delinearize_vertex_weights(res, graph)
    return res


@_legacy_cache_keywords
def attribute_edge_length(graph):
    """
    Compute the length of each edge of the given graph.

    The length :math:`l(e)` of an edge :math:`e` is the length of the border shared by the two vertices it connects. It
    is determined as follows:

    - if :attr:`graph` is a region adjacency graph, each edge represents the set of the edges of the underlying graph
      linking two adjacent regions: its length is the sum of the lengths of these edges, these lengths being computed
      recursively with :func:`~higra.attribute_edge_length` on the underlying graph (see
      :func:`~higra.rag_accumulate_on_edges`). For a region adjacency graph built on a 4-adjacency pixel graph, the
      length of an edge is thus the number of pixel sides shared by the two regions;
    - otherwise, the length of each edge is 1.

    This function requires an explicit graph: it does not accept implicit grid graphs such as the ones created by
    :func:`~higra.get_4_adjacency_implicit_graph`.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(m)` time, where :math:`m` is the number of edges of the graph. For a
    region adjacency graph, add the cost of the computation on its underlying graph.

    :Example:
    =======

    Consider the region adjacency graph defined by the following labelisation of a 3 x 3 image:

    .. code-block:: text

        0  0  1
        0  0  1
        2  2  2

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        labels = np.asarray(((0, 0, 1), (0, 0, 1), (2, 2, 2)))
        rag = hg.make_region_adjacency_graph_from_labelisation(graph, labels)
        hg.attribute_edge_length(graph)
        # array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

        rag.edge_list()
        # (array([0, 0, 1]), array([1, 2, 2]))
        hg.attribute_edge_length(rag)
        # array([2., 2., 1.])

    The regions 0 and 1 share two pixel sides, as do the regions 0 and 2, while the regions 1 and 2 share a single
    pixel side.

    :param graph: input graph
    :return: a 1d array of floating point values representing the length of each edge of the graph (Numpy array of shape (graph.num_edges(),))
    """
    if hg.CptRegionAdjacencyGraph.validate(graph):  # this is a rag like graph
        pre_graph = hg.CptRegionAdjacencyGraph.get_pre_graph(graph)
        pre_graph_edge_length = attribute_edge_length(pre_graph)
        return hg.rag_accumulate_on_edges(graph, hg.Accumulators.sum, edge_weights=pre_graph_edge_length)
    res = np.ones((graph.num_edges(),), dtype=np.float64)
    return res


@_legacy_cache_keywords
def attribute_vertex_perimeter(graph, edge_length=None):
    """
    Compute the perimeter of each vertex of the given graph.

    The perimeter :math:`p(v)` of a vertex :math:`v` is determined as follows:

    - if :attr:`graph` is a grid graph created by a function such as :func:`~higra.get_4_adjacency_graph`,
      :func:`~higra.get_8_adjacency_graph`, :func:`~higra.get_6_adjacency_graph` or
      :func:`~higra.get_nd_regular_graph` (or their implicit versions), the perimeter of every vertex is the number
      :math:`k` of neighbours of a vertex located inside the grid (for example 4 for a 4-adjacency graph and 8 for an
      8-adjacency graph). The vertices on the border of the grid thus have the same perimeter as the other ones: the
      border of the image is considered as part of their contour. In this case, :attr:`edge_length` is not used;
    - otherwise, the perimeter of a vertex is the sum of the lengths of the edges incident to this vertex:

      .. math::

          p(v) = \\sum_{e \\in E,\\ v \\in e} l(e)

      where :math:`E` is the edge set of the graph and :math:`l(e)` is the length of the edge :math:`e`, given by
      :attr:`edge_length` if provided, and otherwise by :func:`~higra.attribute_edge_length`. In particular, for a
      region adjacency graph, the perimeter of a region is the total length of its borders with the other regions:
      the border of the domain (for example the border of the image) is not counted.

    When :attr:`edge_length` is not provided, the graph must be an explicit graph (see
    :func:`~higra.attribute_edge_length`).

    :Complexity:

    The computation takes :math:`\\mathcal{O}(n + m)` time, where :math:`n` is the number of vertices and :math:`m`
    the number of edges of the graph. If :attr:`edge_length` is omitted, add the cost of
    :func:`~higra.attribute_edge_length`.

    :Example:
    =======

    On a 4-adjacency grid graph, every pixel has a perimeter of 4, including the pixels on the border of the image:

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((2, 3))
        hg.attribute_vertex_perimeter(graph)
        # array([[4., 4., 4.],
        #        [4., 4., 4.]])

    On a general graph, the perimeter is the sum of the lengths of the incident edges:

    .. code-block:: python

        graph = hg.UndirectedGraph(3)
        graph.add_edges((0, 1), (1, 2))
        hg.attribute_vertex_perimeter(graph)
        # array([1., 2., 1.])

        hg.attribute_vertex_perimeter(graph, edge_length=np.asarray((2., 3.)))
        # array([2., 5., 3.])

    Consider now the region adjacency graph defined by the following labelisation of a 3 x 3 image:

    .. code-block:: text

        0  0  1
        0  0  1
        2  2  2

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        labels = np.asarray(((0, 0, 1), (0, 0, 1), (2, 2, 2)))
        rag = hg.make_region_adjacency_graph_from_labelisation(graph, labels)
        hg.attribute_vertex_perimeter(rag)
        # array([4., 3., 3.])

    The region 0 shares 2 pixel sides with the region 1 and 2 pixel sides with the region 2: its perimeter is 4, the
    border of the image being not counted.

    :param graph: input graph
    :param edge_length: length of the edges of the input graph (provided by :func:`~higra.attribute_edge_length` on `graph`)
    :return: an array of floating point values representing the perimeter of each vertex of the graph, with one element per vertex or, if the graph is a grid graph, of the shape of the grid
    """
    if edge_length is None:
        edge_length = hg.attribute_edge_length(graph)

    special_case_border_graph = _get_attribute(graph, "no_border_vertex_out_degree")

    if special_case_border_graph is not None:
        res = np.full((graph.num_vertices(),), special_case_border_graph, dtype=np.float64)
        res = hg.delinearize_vertex_weights(res, graph)
        return res

    res = hg.accumulate_graph_edges(graph, edge_length, hg.Accumulators.sum)
    res = hg.delinearize_vertex_weights(res, graph)
    return res


@_legacy_cache_keywords
@hg.argument_helper(hg.CptGridGraph)
def attribute_vertex_coordinates(graph, shape):
    """
    Compute the coordinates of each vertex of the given 2d grid graph.

    The vertex of the grid graph located at row :math:`i` and column :math:`j` has the coordinates :math:`(i, j)`.
    The result follows the shape of the grid, with an additional last dimension of size 2 holding the coordinates:
    for a grid of shape :math:`(h, w)`, the result has shape :math:`(h, w, 2)` and :math:`result[i, j] = (i, j)`.

    The graph must be a 2d grid graph (Concept :class:`~higra.CptGridGraph`), for example created by
    :func:`~higra.get_4_adjacency_graph` or :func:`~higra.get_8_adjacency_graph`.

    :Complexity:

    The computation takes :math:`\\mathcal{O}(n)` time, where :math:`n` is the number of vertices of the graph.

    :Example:
    =======

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((2, 3))
        hg.attribute_vertex_coordinates(graph)
        # array([[[0, 0],
        #         [0, 1],
        #         [0, 2]],
        #
        #        [[1, 0],
        #         [1, 1],
        #         [1, 2]]])

    The vertex 4, located in the second row and the second column of the grid, has the coordinates :math:`(1, 1)`.

    :param graph: input 2d grid graph (Concept :class:`~higra.CptGridGraph`)
    :param shape: shape of the grid (deduced from :class:`~higra.CptGridGraph`)
    :return: an array of integers of shape (shape[0], shape[1], 2) representing the coordinates of each vertex of the graph
    """
    coords = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = [c.reshape((-1,)) for c in coords]
    attribute = np.stack(list(reversed(coords)), axis=1)
    attribute = hg.delinearize_vertex_weights(attribute, graph)
    return attribute
