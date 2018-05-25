import higra as hg


@hg.data_consumer(edge_weights="edge_weights")
def labelisation_watershed(graph, edge_weights):
    """
    Compute a watershed cut of the given edge weighted graph.

    The watershed cut is represented by a labelisation of the graph vertices.

    :param graph:
    :param edge_weights:
    :return:
    """
    vertex_labels = hg._labelisation_watershed(graph, edge_weights)

    hg.set_attribute(graph, "vertex_labels", vertex_labels)

    return vertex_labels
