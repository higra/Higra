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


def read_tree(filename):
    """
    Read a tree stored in mixed ascii/binary format.

    Attributes are also registered as tree object attributes.

    :param filename: path to the tree file
    :return: a pair (tree, attribute_map)
    """
    tree, attribute_map = hg.cpp._read_tree(filename)

    for k in attribute_map:
        hg.set_attribute(tree, k, attribute_map[k])

    return tree, attribute_map


def print_partition_tree(tree, *,
               altitudes=None,
               attribute=None,
               float_size=4,
               ordering="area",
               scale="linear",
               return_string=False):
    """
    Print a partition tree in ASCII format.

    The tree is represented as a dendrogram oriented horizontally with the leaves on the left and the root on the right.
    Node positions are proportional to their altitudes.

    This function can be used for debugging and illustrations: it is not meant to handle large trees.

    :param tree: Input tree
    :param altitudes: Tree node altitudes (will default to :func:`~higra.attribute_regular_altitudes(tree)` if ``None``)
    :param attribute: Optional tree node attributes. If provided, the node attribute value will be printed instead
            of its altitude.
    :param float_size: Number of characters reserved for number printing.
    :param ordering: determine how the children of a node are ordered. Possible values are
            'altitudes', 'area', 'none', 'leaves'
    :param scale: scale of the x axis: 'linear' (default) or 'log'
    :param return_string: if ``True``, the string is returned instead of being printed (default ``False``)
    :return: A string if :attr:`return_string` is ``True``, ``None`` otherwise
    """
    # arbitrary !
    nleaves = tree.num_leaves()
    assert nleaves < 100, "Tree has too many leaves for pretty print!"
    if nleaves >= 10:
        leaf_format = "{:2d}"
    else:
        leaf_format = "{:1d}"

    # number printing
    float_size = max(3, int(float_size))
    half_float = max(1, float_size // 2)
    prec = max(1, float_size - 1)
    float_format = "{0:" + str(float_size) + "." + str(prec) + "g}"

    # space between two leaves
    y_spacing = 3

    # normalized altitudes determines parent/child spacing
    if altitudes is None:
        normalized_altitudes = altitudes = hg.attribute_regular_altitudes(tree)
    else:
        min_a = np.min(altitudes[tree.num_leaves():])
        normalized_altitudes = (altitudes - min_a) / (np.max(altitudes) - min_a)
        normalized_altitudes[:tree.num_leaves()] = 0

    if scale == "log":
        normalized_altitudes = np.log(1 + normalized_altitudes) / np.log(2)
    elif scale == "linear":
        pass
    else:
        raise ValueError("Invalid scale parameter '" + scale + "'")

    # attribute is what is printed
    if attribute is None:
        attribute = altitudes

    # the minimum difference of altitudes between a child and its parent will determine the total size of the graph
    diff_altitudes = normalized_altitudes[tree.parents()] - normalized_altitudes
    min_diff_altitudes = np.min(diff_altitudes[np.nonzero(diff_altitudes)])

    # spacing between two succesors cannot be less than float_size + 3
    total_width = int((float_size + 3) / min_diff_altitudes + 1)
    total_height = (1 + y_spacing) * nleaves - y_spacing

    # arbitrary !
    assert total_width < 1000, "Tree is to deep for pretty print!"

    # "drawing" area
    screen = np.full((total_height, total_width + 10), ' ')

    # y positions
    yy = np.zeros((tree.num_vertices(),))

    # area is necessary to determine how much space must be "reserved" for each child of a node
    area = hg.attribute_area(tree)

    # how leaves are sorted
    ordering_modes = {
        'altitudes': lambda cl: sorted(cl, key=lambda c: altitudes[c]),
        'area': lambda cl: sorted(cl, key=lambda c: area[c]),
        'none': lambda cl: cl,
        'leaves': None
    }

    if ordering not in ordering_modes:
        raise ValueError('Invalid ordering mode.')
    else:
        ordering = ordering_modes[ordering]

    # special case, not that the branch of the tree might self-intersect...
    if ordering is None:
        yy[:nleaves] = np.arange(0, total_height, y_spacing + 1)
        for n in tree.leaves_to_root_iterator(include_leaves=False):
            yy[n] = np.mean(yy[tree.children(n)])
    else:
        def compute_yy_rec(n, left, right):
            if tree.is_leaf(n):
                yy[n] = (left + right) / 2
            else:
                cl = ordering(tree.children(n))

                r = right - left
                ys = []
                tarea = 0
                narea = area[n]
                for i, c in enumerate(cl):
                    y = compute_yy_rec(c, left + r * tarea / narea, left + r * (tarea + area[c]) / narea)
                    ys.append(y)
                    tarea += area[c]

                yy[n] = np.mean(ys)

            return yy[n]

        compute_yy_rec(tree.root(), 0, total_height)

    # final scaling along x axis
    # because we substract the mininal non zero value in normalized altitudes,
    # we shift non leaves nodes to separate them from leaves
    xshift = half_float + 1
    x0_util = 0
    x1_util = total_width - xshift
    xr_util = x1_util - x0_util

    xx = np.round(xr_util * normalized_altitudes)
    xx[tree.num_leaves():] += xshift

    def write_string(y, x, s):
        for i, c in enumerate(s):
            screen[y, x + i] = c

    def draw_hline(y, x1, x2):
        if x1 > x2:
            x1, x2 = x2, x1
        for x in range(x1, x2 + 1):
            screen[y, x] = "-"

    def draw_vline(x, y1, y2):
        if y1 > y2:
            y1, y2 = y2, y1
        for y in range(y1, y2 + 1):
            screen[y, x] = "|"

    yy -= 1
    xx = xx.astype(np.int32)
    yy = yy.astype(np.int32)

    for n in tree.leaves_to_root_iterator(include_leaves=False):
        nx = xx[n] + half_float + 1
        ny = yy[n]
        for c in tree.children(n):
            cx = xx[c]
            if not tree.is_leaf(c):
                cx += half_float + 1
            cy = yy[c]
            draw_vline(nx, cy, ny)
            draw_hline(cy, cx, nx)

    for n in tree.leaves():
        s = leaf_format.format(n)
        write_string(yy[n], xx[n], s)

    for n in tree.leaves_to_root_iterator(include_leaves=False):
        s = float_format.format(attribute[n])
        write_string(yy[n], xx[n], s)

    r = []
    for i in range(screen.shape[0]):
        s = screen[i, :].astype('|S1').tostring().decode('utf-8')
        s = s.rstrip()
        if s != "":
            r.append(s)

    r = "\n".join(r)
    if not return_string:
        print(r)
    else:
        return r
