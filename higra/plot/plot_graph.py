############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) :                                                         #
#   - Giovanni Chierchia                                                   #
#   - Benjamin Perret                                                      #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import numpy as np
from .utils import COLORS, MARKERS, lighten_color

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    __matplotlib_available = True
except:
    __matplotlib_available = False


def plot_graph(graph, *, vertex_positions, vertex_labels=None):
    """
    Plot the given graph.

    Requires the ``matplotlib`` library.

    :param graph: Input graph
    :param vertex_positions: 2d array containing the coordinates of each vertex of the graph
    :param labels: Optional: vertex labels
    :return: None
    """
    assert __matplotlib_available, "The plot graph function requires matplotlib"

    sources, targets = graph.edge_list()
    segments = np.stack((vertex_positions[sources, :], vertex_positions[targets, :]), axis=1)
    lc = LineCollection(segments, zorder=0, colors='k')
    lc.set_linewidths(1)
    ax = plt.gca()
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim(segments[:, :, 0].min(), segments[:, :, 0].max())
    ax.set_ylim(segments[:, :, 1].min(), segments[:, :, 1].max())
    ax.add_collection(lc)
    plt.axis('equal')
    ax.scatter(vertex_positions[:, 0], vertex_positions[:, 1], s=20, c='w', edgecolors='k')

    if vertex_labels is not None:
        _, vertex_labels = np.unique(vertex_labels, return_inverse=True)
        ec = COLORS[vertex_labels % len(COLORS)]
        plt.scatter(vertex_positions[:, 0], vertex_positions[:, 1], s=15, linewidths=1.5, c=lighten_color(ec),
                    edgecolors=ec, alpha=0.9)
        plt.xticks(())
        plt.yticks(())

