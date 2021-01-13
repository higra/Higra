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


@hg.extend_class(hg.RegularGraph1d, method_name="__reduce__")
@hg.extend_class(hg.RegularGraph2d, method_name="__reduce__")
@hg.extend_class(hg.RegularGraph3d, method_name="__reduce__")
@hg.extend_class(hg.RegularGraph4d, method_name="__reduce__")
@hg.extend_class(hg.RegularGraph5d, method_name="__reduce__")
def ____reduce__(self):
    return self.__class__, (self.shape, self.neighbour_list()), self.__dict__


@hg.extend_class(hg.RegularGraph1d, method_name="as_explicit_graph")
@hg.extend_class(hg.RegularGraph2d, method_name="as_explicit_graph")
@hg.extend_class(hg.RegularGraph3d, method_name="as_explicit_graph")
@hg.extend_class(hg.RegularGraph4d, method_name="as_explicit_graph")
@hg.extend_class(hg.RegularGraph5d, method_name="as_explicit_graph")
def __as_explicit_graph(self):
    """
    Converts the current regular graph instance to an equivalent explicit undirected graph.

    :return: An :class:`~higra.UndirectedGraph` equivalent to the current graph
    """
    g = self._as_explicit_graph()
    if hg.CptGridGraph.validate(self):
        hg.CptGridGraph.link(g, hg.CptGridGraph.get_shape(self))
    return g


@hg.extend_class(hg.RegularGraph1d, method_name="__new__")
@hg.extend_class(hg.RegularGraph2d, method_name="__new__")
@hg.extend_class(hg.RegularGraph3d, method_name="__new__")
@hg.extend_class(hg.RegularGraph4d, method_name="__new__")
@hg.extend_class(hg.RegularGraph5d, method_name="__new__")
def ___new__(cls, shape, neighbour_list):
    if hg.is_iterable(shape):
        shape = hg.normalize_shape(shape)
    elif str(type(shape)).find("EmbeddingGrid") != -1:
        shape = shape.shape()
    else:
        raise ValueError("Invalid shape type.")
    g = cls._make_instance(shape, neighbour_list)
    hg.CptGridGraph.link(g, shape)
    return g


@hg.extend_class(hg.RegularGraph1d, method_name="__init__")
@hg.extend_class(hg.RegularGraph2d, method_name="__init__")
@hg.extend_class(hg.RegularGraph3d, method_name="__init__")
@hg.extend_class(hg.RegularGraph4d, method_name="__init__")
@hg.extend_class(hg.RegularGraph5d, method_name="__init__")
def ___init__(self, shape, neighbour_list):
    """
    Creates a new regular grid graph

    Example:

    >>> # construct implicit 6 adj 3D graph
    >>> adj_6 = (( 0,  0, -1),
    >>>          ( 0,  0,  1),
    >>>          ( 0, -1,  0),
    >>>          ( 0,  1,  0),
    >>>          (-1,  0,  0),
    >>>          ( 1,  0,  0))
    >>>
    >>> shape = (100, 100, 100)
    >>>
    >>> g = hg.RegularGraph3d(shape, adj_6)


    :param shape: list of int or embedding of the right dimension
    :param neighbour_list: a list of points coordinates
    :return: a regular graph instance
    """
    pass
