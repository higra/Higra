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
import higra as hg
import numpy as np


class TestAlgorithmGraphCore(unittest.TestCase):

    def test_graph_cut_2_labelisation(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0), dtype=np.int32)

        labels = hg.graph_cut_2_labelisation(edge_weights, graph)

        ref_labels = np.asarray((1, 2, 2, 1, 1, 3, 1, 3, 3), dtype=np.int32)
        self.assertTrue(hg.is_in_bijection(labels, ref_labels))

    def test_labelisation_2_graph_cut(self):
        graph = hg.get_4_adjacency_graph((3, 3))
        labels = np.asarray((1, 2, 2, 1, 1, 3, 1, 3, 3), dtype=np.int32)

        edge_weights = hg.labelisation_2_graph_cut(labels, graph)

        ref_edge_weights = np.asarray((1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0), dtype=np.int32)

        self.assertTrue(hg.is_in_bijection(edge_weights, ref_edge_weights))


if __name__ == '__main__':
    unittest.main()
