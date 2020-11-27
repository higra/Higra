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



