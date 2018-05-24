import higra as hg


@hg.data_consumer(vertex_labels="vertex_labels")
def make_region_adjacency_graph(graph, vertex_labels):
    """
    Create a region adjacency graph (rag) of a vertex labelled graph.
    Each maximal connected set of vertices having the same label is a region.
    Each region is represented by a vertex in the rag.
    There is an edge between two regions of labels l1 and l2 in the rag iff there exists an edge linking 2
    vertices of labels l1 and l2 int he original graph.

    The returned rag is equipped with three attributes:

    * vertex_map: an array of size graph.num_vertices() which indicates for each vertex v of the graph
    the index of the vertex of the rag that represents the region containing v;
    * edge_map: an array of size graph.num_edges() which indicates for each edge e of the graph
    the index of the edge of the rag that links the two regions containing the extremities of e. If no such edge exists
    (if both extremities of e are in the same region), the value -1 is used.
    * original_graph: the graph from which this rag has been computed

    :param graph:
    :param vertex_labels:
    :return:
    """
    rag, vertex_map, edge_map = hg._make_region_adjacency_graph(graph, vertex_labels)

    hg.set_attribute(rag, "vertex_map", vertex_map)
    hg.set_attribute(rag, "edge_map", edge_map)
    hg.set_attribute(rag, "original_graph", graph)

    return rag


def rag_back_project_vertex_weights(rag, rag_vertex_weights):
    """
    Projects rag vertex weights onto original graph vertices.
    The result is an array weighting the vertices of the original graph of the rag such that:
    for any vertex v of the original graph, its weight is equal to the weight of the vertex of the rag that represents
    the region that contains v.

    For any vertex index i,
    result[i] = rag_vertex_weight[rag_vertex_map[i]]

    :param rag:
    :param rag_vertex_weights:
    :return:
    """
    return hg._rag_back_project_weights(hg.get_attribute(rag, "vertex_map"), rag_vertex_weights)


def rag_back_project_edge_weights(rag, rag_vertex_weights):
    """
    Projects rag edge weights onto original graph edges.
    The result is an array weighting the edges of the original graph of the rag such that:
    for any edge e of the original graph, its weight is equal to the weight of the edge of the rag that represents
    that links the two regions containing the extremities of e. If no such edge exists (if the extremities of e are
    in the same region), its value is 0.

    For any edge index ei,
    result[ei] = rag_edge_weight[rag_edge_map[ei]] if rag_edge_map[ei] != -1 and 0 otherwise

    :param rag:
    :param rag_vertex_weights:
    :return:
    """
    return hg._rag_back_project_weights(hg.get_attribute(rag, "edge_map"), rag_vertex_weights)


def rag_accumulate_on_vertices(rag, vertex_weights, accumulator):
    """
    Computes rag vertex weights by accumulating values from the vertex weights of the original graph.

    For any vertex index i of the rag,
    result[i] = accumulate({vertex_weights[j] | rag_vertex_map[j] == i})

    :param rag:
    :param vertex_weights:
    :param accumulator:
    :return:
    """
    return hg._rag_accumulate(hg.get_attribute(rag, "vertex_map"), vertex_weights, accumulator)


def rag_accumulate_on_edges(rag, edge_weights, accumulator):
    """
    Computes rag edge weights by accumulating values from the edge weights of the original graph.

    For any edge index i of the rag,
    result[i] = accumulate({vertex_weights[j] | rag_vertex_map[j] == i})

    :param rag:
    :param vertex_weights:
    :param accumulator:
    :return:
    """
    return hg._rag_accumulate(hg.get_attribute(rag, "edge_map"), edge_weights, accumulator)
