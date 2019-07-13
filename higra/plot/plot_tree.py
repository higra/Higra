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
import higra as hg
from .utils import COLORS

try:
    from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
    __scipy_available = True
except:
    __scipy_available = False

try:
    import matplotlib.pyplot as plt
    __matplotlib_available = True
except:
    __matplotlib_available = False


def plot_partition_tree(tree, *, altitudes=None, n_clusters=0, lastp=30):
    """
    Plot the given tree as a dendrogram.

    Requires the ``matplotlib`` and the ``scipy`` libraries.

    :param tree: Input tree
    :param altitudes: Tree node altitudes (will default to :func:`~higra.attribute_regular_altitudes(tree)` if ``None``)
    :param n_clusters: Colorize the :attr:`n_clusters` largest clusters of the dendrogram with different colors
    :param lastp: Collapse subtrees containing less than :attr:`lastp` leaves.
    :return: void
    """
    assert __scipy_available, "The plot tree function requires scipy"
    assert __matplotlib_available, "The plot tree function requires matplotlib"

    if np.max(tree.num_children()) > 2:
        tree, nmap = hg.tree_2_binary_tree(tree)
        if altitudes is not None:
            altitudes = altitudes[nmap]

    linkage_matrix = hg.binary_hierarchy_to_scipy_linkage_matrix(tree, altitudes)

    extra = {} if lastp is None else dict(truncate_mode='lastp', p=lastp)
    set_link_color_palette(list(COLORS))
    dsort = np.sort(linkage_matrix[:, 2])
    dendrogram(linkage_matrix, no_labels=True, above_threshold_color="k", color_threshold=dsort[-n_clusters + 1],
               **extra)
    plt.yticks([])
