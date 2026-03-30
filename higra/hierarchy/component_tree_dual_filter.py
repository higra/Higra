############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Wonder Alexandre Luz Alves                              #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg


def _get_attribute_map():
    return {
        "area": hg.CasfAttribute.area,
        "bounding_box_width": hg.CasfAttribute.bounding_box_width,
        "bounding_box_height": hg.CasfAttribute.bounding_box_height,
        "bounding_box_diagonal": hg.CasfAttribute.bounding_box_diagonal,
    }


def connected_alternating_sequential_filter(graph, vertex_weights, attribute, thresholds):
    """
    Connected alternating sequential filter on a vertex weighted graph.

    The filter alternates an extensive max-tree pruning step and an anti-extensive
    min-tree pruning step for each threshold in ``thresholds``.

    :param graph: input graph
    :param vertex_weights: vertex weights of the input graph
    :param attribute: one of ``"area"``, ``"bounding_box_width"``,
        ``"bounding_box_height"``, ``"bounding_box_diagonal"``, or the
        corresponding :class:`~higra.CasfAttribute` value
    :param thresholds: sequence of thresholds applied successively
    :return: filtered vertex weights
    """
    attribute_map = _get_attribute_map()

    if isinstance(attribute, str):
        try:
            attribute = attribute_map[attribute]
        except KeyError:
            raise ValueError("Unknown CASF attribute '" + attribute + "'. Expected one of " + str(tuple(attribute_map.keys())) + ".")
    elif attribute not in attribute_map.values():
        raise ValueError("attribute must be a string or a higra.CasfAttribute value.")

    vertex_weights = hg.linearize_vertex_weights(vertex_weights, graph)

    filtered_weights = hg.cpp._connected_alternating_sequential_filter(graph, vertex_weights, attribute, thresholds)

    return hg.delinearize_vertex_weights(filtered_weights, graph)
