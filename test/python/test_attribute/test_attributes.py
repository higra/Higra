############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import unittest
import numpy as np
import higra as hg


class TestAttributes(unittest.TestCase):

    def test_area(self):
        g = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray((1, 4, 6, 5, 2, 7, 3))

        ref_area = (1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 6)

        tree, altitudes = hg.bpt_canonical(edge_weights, g)
        area = hg.attribute_area(tree)
        self.assertTrue(np.all(ref_area == area))

        tree2 = hg.Tree(tree.parents())
        area2 = hg.attribute_area(tree2, data_debug=True)
        self.assertTrue(np.all(ref_area == area2))


if __name__ == '__main__':
    unittest.main()
