############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) :                                                         #
#   - Benjamin Perret                                                      #
#   - Giovanni Chierchia                                                   #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg
import numpy as np


def labelisation_2_graph_cut(graph, vertex_labels):
    """
    Determines the graph cut that corresponds to a given labeling
    of the graph vertices.

    The result is a weighting of the graph edges where edges with
    a non zero weight are part of the cut.

    :param graph: input graph
    :param vertex_labels: Weights on the vertices of the graph
    :return: graph edge-weights representing the equivalent cut
    """
    vertex_labels = hg.linearize_vertex_weights(vertex_labels, graph)
    graph_cut = hg.cpp._labelisation_2_graph_cut(graph, vertex_labels)

    return graph_cut


def graph_cut_2_labelisation(graph, edge_weights):
    """
    Labelises graph vertices according to the given graph cut.

    Each edge having a non zero value in the given edge_weights
    are assumed to be part of the cut.

    :param graph: Input graph
    :param edge_weights: Weights on the edges of the graph
    :return: A labelisation of the graph vertices
    """
    vertex_labels = hg.cpp._graph_cut_2_labelisation(graph, edge_weights)

    vertex_labels = hg.delinearize_vertex_weights(vertex_labels, graph)

    return vertex_labels


def undirected_graph_2_adjacency_matrix(graph, edge_weights=None, non_edge_value=0, sparse=True):
    """
    Adjacency matrix corresponding to an undirected edge-weighted graph (the result is thus symmetric).

    As the given graph is not necessarily complete, non-existing edges will receive the value :attr:`non_edge_value` in
    the adjacency matrix.

    :param graph: Input graph
    :param edge_weights: Graph edge weights (default to ``np.ones(graph.num_edges())`` if ``None``)
    :param non_edge_value: Value used to represent edges that are not in the input graph (must be 0 if :attr:`sparse`
           is ``True``)
    :param sparse: if ``True`` the result will be a sparse matrix in the csr format (requires Scipy to be installed)
    :return: A 2d symmetric square matrix
    """
    if edge_weights is None:
        edge_weights = np.ones((graph.num_edges(),), np.float64)

    num_v = graph.num_vertices()
    sources, targets = graph.edge_list()

    if sparse:
        try:
            from scipy.sparse import csr_matrix
        except:
            raise RuntimeError("scipy required to create a sparse matrix.")

        if non_edge_value != 0:
            raise ValueError("'non_edge_value' must be equal to 0 is 'sparse' is True: Scipy sparse matrix dor not "
                             "support custom default value.")

        A = csr_matrix((edge_weights, (sources, targets)), shape=(num_v, num_v), dtype=edge_weights.dtype)
        A += A.T

    else:
        A = np.empty((num_v, num_v), dtype=edge_weights.dtype)
        A.fill(non_edge_value)
        A[sources, targets] = edge_weights
        A[targets, sources] = edge_weights

    return A


def adjacency_matrix_2_undirected_graph(adjacency_matrix, non_edge_value=0):
    """
    Undirected edge-weighted graph corresponding to an adjacency matrix.

    Adjacency matrix entries which are equal to :attr:`non_edge_value` are not considered to be part of the graph.

    :param adjacency_matrix: Input adjacency matrix (A 2d symmetric square matrix)
    :param non_edge_value: Value used to represent non existing edges in the adjacency matrix
    :return: a pair (UndirectedGraph, ndarray) representing the graph and its edge_weights (Concept :class:`~higra.CptEdgeWeightedGraph`)
    """
    if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("'adjacency_matrix' must be a 2d square matrix.")

    try:
        import scipy.sparse as sp
        scipy_available = True
    except:
        scipy_available = False

    if scipy_available and sp.issparse(adjacency_matrix):
        if non_edge_value != 0:
            raise ValueError("'non_edge_value' must be equal to 0 is 'adjacency_matrix' is a Scipy sparse matrix.")
        adjacency_matrix = sp.triu(adjacency_matrix)
        sources, targets, edge_weights = sp.find(adjacency_matrix)
    else:
        adjacency_matrix = adjacency_matrix.copy()
        adjacency_matrix[np.tri(*adjacency_matrix.shape, k=-1, dtype=np.bool)] = non_edge_value
        if non_edge_value != 0:
            mask = adjacency_matrix != non_edge_value
        else:
            mask = adjacency_matrix
        sources, targets = np.nonzero(mask)
        edge_weights = adjacency_matrix[sources, targets]

    graph = hg.UndirectedGraph(adjacency_matrix.shape[0])
    graph.add_edges(sources, targets)

    return graph, edge_weights


def ultrametric_open(graph, edge_weights):
    """
    Subdominant ultrametric of the given edge weighted graph.

    The subdominant ultrametric relative to a given dissimilarity measure (here the graph edge weights)
    is defined as the largest ultrametric smaller than the dissimilarity measure.

    In the case of an edge weighted undirected graph, the value of the subdominant ultrametric on the
    edge :math:`e_{xy}` is given by the min-max distance between :math:`x` and :math:`y`.

    Complexity: :math:`\mathcal{O}(n*log(n))` with :math:`n` the number of edges in the graph

    :param graph: Input graph
    :param edge_weights: Graph edge weights
    :return: edge weights corresponding to the subdominant ultrametric
    """
    tree, altitudes = hg.bpt_canonical(edge_weights, graph)
    return hg.saliency(altitudes)


def minimum_spanning_tree(graph, edge_weights):
    """
    Computes the minimum spanning tree of the given edge weighted graph with Kruskal's algorithm.

    If the input graph is not connected, the result is indeed a minimum spanning forest.

    Complexity: :math:`\mathcal{O}(n*log(n))` with :math:`n` the number of edges in the graph

    :param graph: Input graph
    :param edge_weights: Graph edge weights
    :return: a minimum spanning tree of the input edge weighted graph (Concept :class:`~higra.CptMinimumSpanningTree`)
    """
    mst, edge_map = hg.cpp._minimum_spanning_tree(graph, edge_weights)
    hg.CptMinimumSpanningTree.link(mst, mst_edge_map=edge_map, base_graph=graph)
    return mst


def make_graph_from_points(X, graph_type="knn+mst", symmetrization="max", **kwargs):
    """
    Creates a graph from vertex coordinates.

    The argument :attr:`graph_type` selects the graph creation methods. Possible values are:

        - ``"complete"``: creates the complete graph
        - ``"knn"``: creates a :math:`k`-nearest neighbor graph, the parameter :math:`k` can be controlled
          with the extra parameter 'n_neighbors' (default value 5).
          The resulting graph may have several connected components.
        - ``"knn+mst"`` (default): creates a :math:`k`-nearest neighbor graph and add the edges of an mst of the complete graph.
          This method ensures that the resulting graph is connected.
          The parameter :math:`k` can be controlled with the extra parameter 'n_neighbors' (default value 5).
        - ``"delaunay"``: creates a graph corresponding to the Delaunay triangulation of the points
          (only works in low dimensions).

    The weight of an edge :math:`\{x,y\}` is equal to the Euclidean distance between
    :math:`x` and :math:`y`: :math:`w(\{x,y\})=\|X[x, :] - X[y, :]\|`.

    :math:`K`-nearest neighbor based graphs are naturally directed, the argument :attr:`symmetrization` enables to chose a
    symmetrization strategy. Possible values are:

        - ``"min"``: an edge :math:`\{x,y\}` is created if there both arcs :math:`(x,y)` and :math:`(y,x)` exist.
          Its weight is given by the minimum weight of the two arcs.
        - ``"max"``: an edge :math:`\{x,y\}` is created if there is any of the two arcs :math:`(x,y)` and :math:`(y,x)` exists.
          Its weight is given by the weight of the existing arcs (if both arcs exists they necessarily have the same weight).

    This method is not suited for large set of points.

    :param X: A 2d array of vertex coordinates
    :param graph_type: ``"complete"``, ``"knn"``, ``"knn+mst"`` (default), or ``"delaunay"``
    :param symmetrization: `"min"`` or ``"max"``
    :param kwargs: extra args depends of chosen graph type
    :return: a graph and its edge weights
    """
    try:
        from scipy.spatial.distance import pdist, squareform, euclidean
        from sklearn.neighbors import kneighbors_graph
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial import Delaunay
    except:
        raise RuntimeError("scipy and sklearn required.")

    n_neighbors = kwargs.get('n_neighbors', 5)
    mode = kwargs.get('mode', 'distance')

    def symmetrization_fun(A):
        if symmetrization == "min":
            return A.minimum(A.T)
        elif symmetrization == "max":
            return A.maximum(A.T)
        else:
            raise ValueError("Unknown symmetrization: " + str(symmetrization))

    if graph_type == "complete":
        d = pdist(X)
        A = squareform(d)
        g, edge_weights = hg.adjacency_matrix_2_undirected_graph(A)
    elif graph_type == "knn":
        A = kneighbors_graph(X, n_neighbors=n_neighbors, mode=mode)
        A = symmetrization_fun(A)
        g, edge_weights = hg.adjacency_matrix_2_undirected_graph(A)
    elif graph_type == "knn+mst":
        A = kneighbors_graph(X, n_neighbors=n_neighbors, mode=mode)
        A = symmetrization_fun(A)
        D = squareform(pdist(X))
        MST = minimum_spanning_tree(D)
        MST = MST + MST.T
        A = A.maximum(MST)
        g, edge_weights = hg.adjacency_matrix_2_undirected_graph(A)
    elif graph_type == "delaunay":
        g = hg.UndirectedGraph(X.shape[0])
        edge_weights = []

        # add QJ to ensure that coplanar point are not discarded
        tmp = Delaunay(X)
        nbp = X.shape[0]
        if tmp.coplanar.size != 0:
            print("Warning coplanar points detected!")
        indices, indptr = tmp.vertex_neighbor_vertices

        for k in range(nbp):
            neighbours = indptr[indices[k]:indices[k+1]]
            for n in neighbours:
                if n > k:
                    d = euclidean(X[k, :], X[n, :])
                    g.add_edge(k, n)
                    edge_weights.append(d)

        edge_weights = np.asarray(edge_weights, dtype=np.float64)
    elif graph_type == "mst":
        D = squareform(pdist(X))
        MST = minimum_spanning_tree(D).toarray()
        MST = MST + MST.T
        g, edge_weights = hg.adjacency_matrix_2_undirected_graph(MST)
    else:
        raise ValueError("Unknown graph_type: " + str(graph_type))

    return g, edge_weights


def subgraph(graph, edge_indices, spanning=True, return_vertex_map=False):
    """
    Extract a subgraph of the input graph. Let :math:`G=(V,E)` be the graph :attr:`graph` and let :math:`E^*`
    be a subset of :math:`E`. The subgraph of :math:`G` induced by :math:`E^*` is equal to:

    - :math:`(V, E^*)` is :attr:`spanning` is ``True``; and
    - :math:`(\\bigcup E^*, E^*)` otherwise (the set of vertices of the subgraph is equal to the set of vertices present at
      an extremity of an edge in :math:`E^*`).

    The array :attr:`edge_indices` contains the indices of the edges in the set :math:`E^*`. The edges in the subgraph
    are in the same order as the edges in the array :attr:`edge_indices`.

    If :attr:`spanning` is ``False``, the subgraph may contain less vertices than the input graph. In such case, the
    optional array result :math:`vertex\_map` (returned if :attr:`return_vertex_map` is ``True``) indicates for each
    vertex :math:`i` of the subgraph, its corresponding index in the input graph.

    :Example:

        >>> # linear graph with 6 vertices
        >>> graph = hg.UndirectedGraph(6)
        >>> graph.add_edges(np.arange(5), np.arange(1, 6))
        >>>
        >>> # select edges (4, 5), (0, 1), and (3, 4), note that vertex 2 is not in any edge
        >>> edge_indices = np.asarray((4, 0, 3))
        >>> subgraph, vertex_map = hg.subgraph(graph, edge_indices, spanning=False, return_vertex_map=True)
        >>>
        >>> subgraph.num_vertices()
        5
        >>> vertex_map
        [0 1 3 4 5]
        >>> subgraph.edge_list()
        ([3 0 2], [4 1 3])
        >>> vertex_map
        [0 1 3 4 5]

    :param graph: input graph.
    :param edge_indices: an array of edge indices of the input graph.
    :param spanning: if ``True``, the subgraph has the same vertex set as the input graph.
    :param return_vertex_map: if ``True``, also returns an array mapping each vertex of the current to its corresponding
           vertex in the input graph.
    :return: a subgraph and, if :attr:`return_vertex_map` is ``True``, a vertex map
    """
    if spanning:
        subgraph = hg.UndirectedGraph(graph.num_vertices())
        sources, targets = graph.edge_list()
        subgraph.add_edges(sources[edge_indices], targets[edge_indices])

        if return_vertex_map:
            vertex_map = np.arange(graph.num_vertices())
    else:
        sources, targets = graph.edge_list()
        sources = sources[edge_indices]
        targets = targets[edge_indices]
        all_vertices = np.concatenate((sources, targets))
        vertex_map, inverse = np.unique(all_vertices, return_inverse=True)

        sources = inverse[:edge_indices.size]
        targets = inverse[edge_indices.size:]

        subgraph = hg.UndirectedGraph(vertex_map.size)
        subgraph.add_edges(sources, targets)

    if return_vertex_map:
        return subgraph, vertex_map
    else:
        return subgraph