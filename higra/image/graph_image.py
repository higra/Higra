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


@hg.argument_helper(hg.CptGridGraph)
def graph_4_adjacency_2_khalimsky(graph, edge_weights, shape, add_extra_border=False):
    """
    Create a contour image in the Khalimsky grid from a 4 adjacency edge-weighted graph.

    :param graph: must be a 4 adjacency 2d graph (Concept :class:`~higra.CptGridGraph`)
    :param edge_weights: edge weights of the graph
    :param shape: shape of the graph (deduced from :class:`~higra.CptGridGraph`)
    :param add_extra_border: if False result size is 2 * shape - 1 and 2 * shape + 1 otherwise
    :return: a 2d array
    """
    shape = hg.normalize_shape(shape)
    return hg.cpp._graph_4_adjacency_2_khalimsky(graph, shape, edge_weights, add_extra_border)


def khalimsky_2_graph_4_adjacency(khalimsky, extra_border=False, graph=None):
    """
    Create a 4 adjacency edge-weighted graph from a contour image in the Khalimsky grid.

    :param khalimsky: a 2d array
    :param extra_border: if False the shape of the Khalimsky image  is 2 * shape - 1 and 2 * shape + 1 otherwise, where shape is the shape of the resulting grid graph
    :param graph: a 4-adjacency graph (Concept :class:`~higra.CptGridGraph`) of the correct shape (optional). If not given, a new graph is created.
    :return: a graph (Concept :class:`~higra.CptGridGraph`) and its edge weights
    """

    border = 0 if extra_border else 1
    res_shape = khalimsky.shape[0] // 2 + border, khalimsky.shape[1] // 2 + border

    if graph is not None:
        graph_shape = hg.CptGridGraph.get_shape(graph)
        if graph_shape is None:
            raise ValueError("The given graph must be a grid graph.")
        if len(graph_shape) != 2:
            raise ValueError("The given graph must be a 2d grid graph.")
        if hg.get_attribute(graph, "no_border_vertex_out_degree") != 4:
            raise ValueError("The given graph must be a 4 adjacency graph.")
        if graph_shape != res_shape:
            raise ValueError("The given graph shape is " + str(graph_shape) + " but the expected shape is " + str(res_shape) + ".")
    else:
        graph = hg.get_4_adjacency_graph(res_shape)

    edge_weights = hg.cpp._khalimsky_2_graph_4_adjacency(khalimsky, graph, res_shape, extra_border)

    return graph, edge_weights


def mask_2_neighbours(mask, center=None):
    """
    Converts as a neighbouring mask as a neighbour list. A neighbouring :attr:`mask` is a :math:`n`-d matrix where
    each strictly positive value represent a neighbour of the :attr:`center` point. The neighbour list is obtained by
    offsetting the coordinates of those positive values by the coordinates of the center.

    The default center is the center of the :attr:`mask` matrix: i.e. ``mask.shape // 2``.

    :Example:

    >>> mask = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> hg.mask_2_neighbours(mask)
    array([[-1, 0], [0, -1], [0, 1], [1, 0]])
    
    >>> mask = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> center = [2, 1]
    >>> hg.mask_2_neighbours(mask)
    array([[-2, 0], [-1, -1], [-1, 1], [0, 0]])
    
    >>> mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    >>>         [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    >>>         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    >>> hg.mask_2_neighbours(mask)
    array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
       
    :param mask:  a :math:`n`-d matrix
    :param center: a 1d array of size :math:`n` (optional)
    :return: a list of point coordinates
    """
    mask = np.asarray(mask)

    if center is None:
        center = np.floor_divide(np.asarray(mask.shape, dtype=np.int64), 2)
    else:
        center = np.asarray(center)
        if len(center) != mask.ndim:
            raise ValueError("'center' size does not match 'mask' dimension.")

    neighbours = np.argwhere(mask > 0)
    neighbours -= center

    return neighbours


def get_4_adjacency_graph(shape):
    """
    Create an explicit undirected 4 adjacency graph of the given shape.

    :param shape: a pair (height, width)
    :return: a graph (Concept :class:`~higra.CptGridGraph`)
    """
    graph_implicit = get_4_adjacency_implicit_graph(shape)
    graph = graph_implicit.as_explicit_graph()

    hg.CptGridGraph.link(graph, hg.CptGridGraph.get_shape(graph_implicit))
    hg.set_attribute(graph, "no_border_vertex_out_degree",
                     hg.get_attribute(graph_implicit, "no_border_vertex_out_degree"))

    return graph


def get_8_adjacency_graph(shape):
    """
    Create an explicit undirected 8 adjacency graph of the given shape.

    :param shape: a pair (height, width)
    :return: a graph (Concept :class:`~higra.CptGridGraph`)
    """
    graph_implicit = get_8_adjacency_implicit_graph(shape)
    graph = graph_implicit.as_explicit_graph()

    hg.CptGridGraph.link(graph, hg.CptGridGraph.get_shape(graph_implicit))
    hg.set_attribute(graph, "no_border_vertex_out_degree",
                     hg.get_attribute(graph_implicit, "no_border_vertex_out_degree"))

    return graph


def get_4_adjacency_implicit_graph(shape):
    """
    Create an implicit undirected 4 adjacency graph of the given shape (edges are not stored).

    :param shape: a pair (height, width)
    :return: a graph (Concept :class:`~higra.CptGridGraph`)
    """
    shape = hg.normalize_shape(shape)
    if len(shape) != 2:
        raise ValueError("Shape must be a 1d array of size 2.")

    neighbours = np.array(((-1, 0), (0, -1), (0, 1), (1, 0)), dtype=np.int64)
    graph = hg.RegularGraph2d(shape, neighbours)

    hg.CptGridGraph.link(graph, shape)
    hg.set_attribute(graph, "no_border_vertex_out_degree", 4)

    return graph


def get_8_adjacency_implicit_graph(shape):
    """
    Create an implicit undirected 8 adjacency graph of the given shape (edges are not stored).

    :param shape: a pair (height, width)
    :return: a graph (Concept :class:`~higra.CptGridGraph`)
    """
    shape = hg.normalize_shape(shape)
    if len(shape) != 2:
        raise ValueError("Shape must be a 1d array of size 2.")

    neighbours = np.array(((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)), dtype=np.int64)
    graph = hg.RegularGraph2d(shape, neighbours)

    hg.CptGridGraph.link(graph, shape)
    hg.set_attribute(graph, "no_border_vertex_out_degree", 8)

    return graph


def get_nd_regular_implicit_graph(shape, neighbour_list):
    """
    Creates an implicit regular graph of the given :attr:`shape` with the adjacency given as a
    :attr:`neighbour_list`.

    See the helper function :func:`~higra.mask_2_neighbours` to create a suitable :attr:`neighbour_list`.

    :Example:

    Create a 2d 4-adjacency implicit graph of size ``(13, 24)``:

    >>> graph = get_nd_regular_implicit_graph((13, 24), ((-1, 0), (0, -1), (0, 1), (1, 0)))

    Create a 3d 6-adjacency implicit graph of size ``(10, 13, 24)``:

    >>> mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    >>>         [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    >>>         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    >>> neighbours = mask_2_neighbours(mask)
    >>> graph = get_nd_regular_implicit_graph((10, 13, 24), neighbours)

    :param shape: a tuple of :math:`n` elements representing the dimension of the graph vertices.
    :param neighbour_list: a 2d array of :math:`k` :math:`n`-d integer vectors
    :return: an implicit regular graph
    """

    neighbour_list = np.asarray(neighbour_list)

    if not np.issubdtype(neighbour_list.dtype, np.integer):
        raise ValueError("'neighbour_list' must be of integral type.")

    if neighbour_list.ndim != 2:
        raise ValueError("'neighbour_list' must be a 2d array.")

    shape = hg.normalize_shape(shape)

    if len(shape) != neighbour_list.shape[1]:
        raise ValueError("Shape size does not match provided adjacency dimension.")

    if len(shape) > 5 or len(shape) == 0:
        raise ValueError("Shape size must between 1 and 5 (included).")

    if len(shape) == 1:
        graph = hg.RegularGraph1d(shape, neighbour_list)
    elif len(shape) == 2:
        graph = hg.RegularGraph2d(shape, neighbour_list)
    elif len(shape) == 3:
        graph = hg.RegularGraph3d(shape, neighbour_list)
    elif len(shape) == 4:
        graph = hg.RegularGraph4d(shape, neighbour_list)
    elif len(shape) == 5:
        graph = hg.RegularGraph5d(shape, neighbour_list)

    hg.CptGridGraph.link(graph, shape)
    hg.set_attribute(graph, "no_border_vertex_out_degree", neighbour_list.shape[0])

    return graph


def get_nd_regular_graph(shape, neighbour_list):
    """
    Creates a regular graph of the given :attr:`shape` with the adjacency given as a
    :attr:`neighbour_list`.

    See the helper function :func:`~higra.mask_2_neighbours` to create a suitable :attr:`neighbour_list`.

    :Example:

    Create a 2d 4-adjacency implicit graph of size ``(13, 24)``:

    >>> graph = get_nd_regular_graph((13, 24), ((-1, 0), (0, -1), (0, 1), (1, 0)))

    Create a 3d 6-adjacency implicit graph of size ``(10, 13, 24)``:

    >>> mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    >>>         [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    >>>         [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    >>> neighbours = mask_2_neighbours(mask)
    >>> graph = get_nd_regular_graph((10, 13, 24), neighbours)

    :param shape: a tuple of :math:`n` elements representing the dimension of the graph vertices.
    :param neighbour_list: a 2d array of :math:`k` :math:`n`-d integer vectors
    :return: a regular graph
    """

    graph_implicit = get_nd_regular_implicit_graph(shape, neighbour_list)
    graph = graph_implicit.as_explicit_graph()

    hg.CptGridGraph.link(graph, hg.CptGridGraph.get_shape(graph_implicit))
    hg.set_attribute(graph, "no_border_vertex_out_degree",
                     hg.get_attribute(graph_implicit, "no_border_vertex_out_degree"))

    return graph


def match_pixels_image_2d(image1, image2, max_distance=0.0075, mode="relative"):
    """
    Computes a matching between the pixels of two images.

    The matching is computed by solving a minimum cost bipartite graph matching problem.
    Any pixel whose value if less than or equal to 0 is considered as a missing pixel and is not matched.
    The cost of a match between two pixels is the Euclidean distance between the two pixels.

    Two pixels are matchable if their distance is less than:

        - :attr:`max_distance` * diagonal size if :attr:`mode` is "relative"
        - :attr:`max_distance` if :attr:`mode` is "absolute"

    The algorithm computes a matching of maximum size and minimum cost between the matchable pixels of both images.
    The algorithm returns a pair of arrays ``(sources, targets)`` representing the edges linking matched pixels:

        - ``sources[i]`` is the linear index of the source pixel in :attr:`image_1` of the edge ``i``
        - ``targets[i]`` is the linear index of the target pixel in :attr:`image_2` of the edge ``i``

    The following figure illustrates the matching of pixels between two images. The blue squares represent the pixels
    of the first image, the green squares represent the pixels of the second images, and the red lines represent the
    matching between blue and green pixels.

    .. image:: images/demo_match_pixels_image_2d.svg
      :width: 400
      :alt: Demonstration of the matching of pixels between two images.



    This function is a reimplementation of the function ``correspondPixels`` used to compute boundary based assessment
    measures of segmentations in the `BSDS500 dataset <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/>`_.

    :Example:

    Let's match two images, with a maximum distance of :math:`1.3\\approx \sqrt(2)`:

    >>> im1 = np.asarray([[1, 0, 0, 1, 1],
    >>>                   [0, 0, 0, 1, 0],
    >>>                   [0, 0, 0, 1, 0],
    >>>                   [0, 0, 1, 1, 1]])
    >>> im2 = np.asarray([[0, 0, 1, 1, 0],
    >>>                   [0, 0, 1, 0, 0],
    >>>                   [0, 0, 1, 0, 1],
    >>>                   [1, 1, 1, 1, 0]])
    >>> hg.match_pixels_image_2d(im1, im2, 1.3, "absolute")
    (array([ 3, 4, 8, 13, 17, 18, 19]), array([ 2, 3, 7, 12, 17, 18, 14]))

    Hence, pixel 3 of ``im1`` is matched with pixel 2 of ``im2``, pixel 4 of ``im1`` is matched with pixel 3 of ``im2``,
    and so on.

    :param image1: a 2d array representing the first image
    :param image2: a 2d array representing the second image
    :param max_distance: the maximum distance between two pixels to be matched
    :param mode: "relative" or "absolute"
    :return: a pair of arrays ``(sources, targets)`` representing the edges linking matched pixels
    """
    if max_distance < 0:
        raise ValueError("max_distance must be positive.")

    if mode == "relative":
        height, width = image1.shape
        diagonal = np.sqrt(height * height + width * width)
        max_distance = max_distance * diagonal
    elif mode == "absolute":
        pass
    else:
        raise ValueError("mode must be 'relative' or 'absolute'.")

    sources, targets, weights, node_map, num_nodes1, num_nodes2 = \
        hg.cpp._get_bipartite_matching_graph_contour_image_2d(image1 > 0, image2 > 0, max_distance)

    matched_edges = hg.bipartite_graph_matching((sources, targets, num_nodes1 + num_nodes2), (weights * 1000).astype(np.int64))

    return node_map[sources[matched_edges]], node_map[targets[matched_edges]]