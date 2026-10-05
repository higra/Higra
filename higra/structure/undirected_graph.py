############################################################################
# Copyright ESIEE Paris (2020)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg


def __reduce_ctr(num_vertices, sources, targets):
    graph = hg.UndirectedGraph(num_vertices)
    graph.add_edges(sources, targets)
    return graph


@hg.extend_class(hg.UndirectedGraph, method_name="__reduce__")
def ____reduce__(self):
    return __reduce_ctr, (self.num_vertices(), *self.edge_list()), self.__dict__


@hg.extend_class(hg.UndirectedGraph, method_name="sources")
def __sources(self):
    """
    Source vertex of every edge of the graph.

    Returns a zero-copy, read-only NumPy view of edge sources.
    It is not guaranteed that the view will remain valid after graph mutations (e.g., adding or removing edges).
    It is recommended to copy the array if you need to preserve its values across graph mutations.
    It is recommended to get a new view after graph mutations to ensure you have the current topology.

    :Example:

    >>> g = UndirectedGraph(3)
    >>> g.add_edges((0, 1, 0), (1, 2, 2))
    >>> g.sources()
    array([0, 1, 0])

    :return: a 1d array of size ``self.num_edges()``
    """
    return self._sources()


@hg.extend_class(hg.UndirectedGraph, method_name="targets")
def __targets(self):
    """
    Target vertex of every edge of the graph.

    Returns a zero-copy, read-only NumPy view of edge targets.
    It is not guaranteed that the view will remain valid after graph mutations (e.g., adding or removing edges).
    It is recommended to copy the array if you need to preserve its values across graph mutations.
    It is recommended to get a new view after graph mutations to ensure you have the current topology.

    :Example:

    >>> g = UndirectedGraph(3)
    >>> g.add_edges((0, 1, 0), (1, 2, 2))
    >>> g.targets()
    array([1, 2, 2])

    :return: a 1d array of size ``self.num_edges()``
    """
    return self._targets()


@hg.extend_class(hg.UndirectedGraph, method_name="edge_list")
def __edge_list(self):
    """
    Returns a tuple of two zero-copy, read-only NumPy views (sources, targets).

    It is not guaranteed that the view will remain valid after graph mutations (e.g., adding or removing edges).
    It is recommended to copy the array if you need to preserve its values across graph mutations.
    It is recommended to get a new view after graph mutations to ensure you have the current topology.

    :Example:

    >>> g = UndirectedGraph(3)
    >>> g.add_edges((0, 1, 0), (1, 2, 2))
    >>> g.edge_list()
    (array([0, 1, 0]), array([1, 2, 2]))

    :return: pair of two 1d arrays
    """
    return self.sources(), self.targets()




