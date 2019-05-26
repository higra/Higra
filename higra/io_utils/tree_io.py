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
